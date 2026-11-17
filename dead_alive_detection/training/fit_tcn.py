"""
Trains the TCN (model/tcn.py) on DFL and runs its heads on any match.

Each epoch uses one random augmented view per training match, cut into random windows. The loss
covers the dead head, the start and stop heads and, optionally, the stoppage type.
"""

import numpy as np
import torch
from torch import nn

from dead_alive_detection.loading.tensors import STEPS_PER_S
from dead_alive_detection.model.tcn import TCN
from dead_alive_detection.training.targets import targets, windows  # noqa: F401 (re-exported)


def standardiser(views):
    """
    Mean and std of the training inputs; a constant or empty column gets std 1.
    """
    x = np.concatenate([f for match in views for _, f in match])
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return mean, std


class Trained:
    """
    A TCN plus the standardiser of its inputs. At prediction time each half is read as one
    sequence, and outputs outside the halves are NaN.
    """

    def __init__(self, model, mean, std, model_cfg):
        self.model, self.mean, self.std, self.model_cfg = model, mean, std, model_cfg

    def z(self, f):
        """
        Standardised float32 inputs, with NaN as 0.
        """
        return np.nan_to_num((f - self.mean) / self.std).astype(np.float32)

    def _halves(self, x, t):
        for half in (1, 2):
            rows = np.flatnonzero(t["period"] == half)
            if len(rows):
                yield rows, self.model(torch.from_numpy(self.z(x[rows]).T[None].copy()))

    def heads(self, x, t):
        """
        Raw logits per step: dead, start, stop, then the types.
        """
        out = np.full((len(x), 3 + self.model.n_types), np.nan, dtype=np.float32)
        with torch.no_grad():
            for rows, logits in self._halves(x, t):
                out[rows] = logits[0].T.numpy()
        return out

    def __call__(self, x, t):
        """
        P(dead) per step.
        """
        p = np.full(len(x), np.nan)
        with torch.no_grad():
            for rows, logits in self._halves(x, t):
                p[rows] = torch.sigmoid(logits[0, 0]).numpy()
        return p

    def save(self, path, **extra):
        """
        Saves the weights, standardiser, model config and any extra values.
        """
        torch.save({"state_dict": self.model.state_dict(), "mean": self.mean, "std": self.std,
                    "model_cfg": self.model_cfg, **extra}, path)

    @classmethod
    def load(cls, path):
        """
        Loads a saved model and returns it with the extra values saved alongside.
        """
        saved = torch.load(path, weights_only=False)
        m = saved.pop("model_cfg")
        state = saved.pop("state_dict")
        n_types = state["type_out.weight"].shape[0] if "type_out.weight" in state else 0
        model = TCN(state["inp.weight"].shape[1], m["channels"], m["kernel"], m["dilations"],
                    m["dropout"], n_types)
        model.load_state_dict(state)
        model.eval()
        return cls(model, saved.pop("mean"), saved.pop("std"), m), saved


def fit_tcn(views, cfg, monitor=None):
    """
    Trains a TCN on the views of each training match and returns it as a Trained.
    If given, monitor is called every log_every epochs with that epoch's mean training losses.
    """
    m, tr = cfg["model"], cfg["train"]
    types = cfg.get("types", {})
    n_types = len(types.get("classes", []))
    type_weight = float(types.get("weight", 0.0)) if n_types else 0.0
    rng = np.random.default_rng(tr["seed"])
    torch.manual_seed(tr["seed"])
    mean, std = standardiser(views)
    sigma = tr["boundary_sigma_s"] * STEPS_PER_S
    n_inputs = views[0][0][1].shape[1]
    model = TCN(n_inputs, m["channels"], m["kernel"], m["dilations"], m["dropout"], n_types,
                types.get("seed", 1))
    trained = Trained(model, mean, std, dict(m))
    # Every view of every match, ready: standardised inputs (F, N), targets (3, N), weights (N,),
    # period, stoppage class per step (-1: none).
    prepared = [[(trained.z(f).T.copy(), *targets(t, sigma), t["period"],
                  t.get("type", np.full(len(f), -1)).astype(np.int64)) for t, f in match]
                for match in views]
    optimiser = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"],
                                  weight_decay=tr["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    length = round(tr["window_s"] * STEPS_PER_S)
    head_weight = torch.tensor([1.0, tr["boundary_weight"], tr["boundary_weight"]])  # dead, start, stop
    log_every = tr.get("log_every", 5)

    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        # This epoch's data: random windows cut from one randomly drawn view of every match.
        crops = []
        for match in prepared:
            x, y, w, period, kind = match[rng.integers(len(match))]   # one view per match per epoch
            crops += [(x[:, a:b], y[:, a:b], w[a:b], kind[a:b]) for a, b in windows(period, length, rng)]
        order = rng.permutation(len(crops))
        total, dead, batches = 0.0, 0.0, 0
        for first in range(0, len(order), tr["batch_size"]):
            batch = [crops[i] for i in order[first:first + tr["batch_size"]]]
            x, y, w, kind = (torch.from_numpy(np.stack(part)) for part in zip(*batch))
            out = model(x)
            # Mean BCE per head over the scored steps (w = 0 elsewhere), then the heads weighted.
            # (contiguous: with a type output, the slice must be reduced exactly as without one)
            heads = out[:, :3].contiguous()
            per_head = (loss_fn(heads, y) * w[:, None]).sum(dim=(0, 2)) / (w.sum() + 1e-6)
            if type_weight and (kind >= 0).any():
                type_loss = nn.functional.cross_entropy(out[:, 3:], kind, ignore_index=-1)
                loss = (((per_head * head_weight).sum() + type_weight * type_loss)
                        / (head_weight.sum() + type_weight))
            else:
                loss = (per_head * head_weight).sum() / head_weight.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total, dead, batches = total + loss.item(), dead + per_head[0].item(), batches + 1
        if monitor is not None and log_every and (epoch % log_every == 0 or epoch == tr["epochs"]):
            model.eval()
            monitor(epoch, {"train_loss": total / batches, "train_dead_loss": dead / batches},
                    trained)
    model.eval()
    return trained
