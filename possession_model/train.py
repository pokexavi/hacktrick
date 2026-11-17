"""
Training: one corrupted copy of each match per hidden-ball share, random 30 s windows, and a loss
for possession plus one for changes. Also predicts a whole match with overlapping windows.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from possession_model import config, features as fx
from possession_model.augment import Corruptor
from possession_model.model import PossessionModel

_ITEMS = {}


def device(cfg):
    want = cfg["runtime"]["device"]
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def item(cfg, corruptor, m, recipe, copy, hidden_frac=None):
    # Kept in memory across folds. On Kaggle it's read from the bundle instead of computed.
    key = (config.dumps({k: cfg[k] for k in ("data", "steps", "augment", "heads")}), m.name, recipe,
           copy, hidden_frac)
    if key not in _ITEMS:
        if cfg["runtime"]["bundle"]:
            from possession_model import bundle
            _ITEMS[key] = bundle.load_item(cfg, m.name, recipe, copy, hidden_frac)
        else:
            version, live = corruptor.corrupt(m, recipe, copy, hidden_frac)
            _ITEMS[key] = fx.build(cfg, version, live, m.target)
    return _ITEMS[key]


def training_specs(cfg):
    # (recipe, copy, hidden_frac): one training copy per hidden_frac
    return [(config.TRAIN_RECIPE, 1 + k, frac) for k, frac in enumerate(cfg["augment"]["hidden_fracs"])]


def training_items(cfg, matches):
    corruptor = Corruptor(cfg)
    return [item(cfg, corruptor, m, *spec) for m in matches for spec in training_specs(cfg)]


def scored_item(cfg, m, condition):
    # A held-out match: copy 0, with BAR's share of hidden ball
    return item(cfg, Corruptor(cfg), m, condition, 0, None)


# Windows

def window_steps(cfg):
    return int(round(cfg["model"]["temporal"]["window_s"] * cfg["steps"]["hz"]))


class Sampler:
    # Windows centred on a random live step with a label, kept inside the half. Some are mirrored
    # across the long axis (mirror_y).

    def __init__(self, items, cfg, rng):
        self.items, self.rng, self.w = items, rng, window_steps(cfg)
        self.mirror = cfg["augment"]["mirror_y"]
        self.centres = [np.flatnonzero(np.isfinite(it["y"]) & it["live"]) for it in items]
        weights = np.array([len(c) for c in self.centres], dtype=float)
        self.p = weights / weights.sum()

    def _span(self, it, c):
        rows = np.flatnonzero(it["period"] == it["period"][c])
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        a = int(np.clip(c - self.w // 2, lo, max(lo, hi - self.w)))
        return a, min(a + self.w, hi)

    def batch(self, size):
        picks = []
        for _ in range(size):
            i = int(self.rng.choice(len(self.items), p=self.p))
            c = int(self.rng.choice(self.centres[i]))
            a, b = self._span(self.items[i], c)
            picks.append((self.items[i], a, b, self.rng.random() < self.mirror))
        return stack(picks, self.w)


def stack(picks, w):
    # Builds a batch from [(item, start, stop, mirror)], padding short windows with zeros
    n = len(picks)
    it0 = picks[0][0]
    out = {"players": np.zeros((n, w) + it0["players"].shape[1:], np.float32),
           "pmask": np.zeros((n, w, it0["pmask"].shape[1]), bool),
           "ball": np.zeros((n, w, it0["ball"].shape[1]), np.float32),
           "ctx": np.zeros((n, w, it0["ctx"].shape[1]), np.float32),
           "valid": np.zeros((n, w), bool),
           "y": np.full((n, w), np.nan, np.float32), "change": np.zeros((n, w), np.float32)}
    for j, (it, a, b, mirror) in enumerate(picks):
        k = b - a
        out["players"][j, :k] = it["players"][a:b]
        out["pmask"][j, :k] = it["pmask"][a:b]
        out["ball"][j, :k] = it["ball"][a:b]
        out["ctx"][j, :k] = it["ctx"][a:b]
        out["valid"][j, :k] = True
        if "y" in it:
            out["y"][j, :k] = it["y"][a:b]
            out["change"][j, :k] = it["change"][a:b]
        if mirror:
            out["players"][j, :k][..., fx.PLAYER_MIRROR] *= -1
            out["ball"][j, :k][..., fx.BALL_MIRROR] *= -1
    return out


def _tensors(batch, dev):
    return {k: torch.from_numpy(v).to(dev) for k, v in batch.items()}


def losses(cfg, model, t):
    # Both are binary cross-entropy on the live steps; the change one is weighted by change_weight
    logit, change = model(t["players"], t["pmask"], t["ball"], t["ctx"], t["valid"])
    live = t["valid"] & (t["ctx"][..., fx.C["live"]] > 0.5)
    has_y = live & torch.isfinite(t["y"])
    poss = F.binary_cross_entropy_with_logits(logit[has_y], t["y"][has_y]) if has_y.any() \
        else logit.sum() * 0
    ch = F.binary_cross_entropy_with_logits(change[live], t["change"][live]) if live.any() \
        else change.sum() * 0
    total = poss + cfg["heads"]["change_weight"] * ch
    return total, {"possession": float(poss.detach()), "change": float(ch.detach())}


def fit(cfg, items, seed=0, log=None, watch=None):
    """
    Trains one model. watch(model, epoch) runs every log_every epochs to print held-out numbers,
    just to look at them; it doesn't affect training.
    """
    tr = cfg["train"]
    torch.manual_seed(seed)
    dev = device(cfg)
    model = PossessionModel(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    sampler = Sampler(items, cfg, np.random.default_rng([seed, 1]))
    history = []
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        start, sums = time.time(), {}
        for _ in range(tr["batches_per_epoch"]):
            total, parts = losses(cfg, model, _tensors(sampler.batch(tr["batch_size"]), dev))
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
        row = {"epoch": epoch, "seconds": round(time.time() - start, 1),
               **{k: round(v / tr["batches_per_epoch"], 4) for k, v in sums.items()}}
        if watch and tr["log_every"] and epoch % tr["log_every"] == 0:
            row.update(watch(model, epoch))
        history.append(row)
        if log:
            log("  epoch {epoch:>3}  ".format(**row) + "  ".join(
                "{} {}".format(k, v) for k, v in row.items() if k != "epoch"))
    model.history = history
    return model


@torch.no_grad()
def predict(cfg, model, it, batch_size=32):
    # Windows over each half overlapping by half a window; the overlaps are averaged
    dev = next(model.parameters()).device
    model.eval()
    w, n = window_steps(cfg), len(it["rows"])
    total, ch_total, count = np.zeros(n), np.zeros(n), np.zeros(n)
    spans = []
    for p in (1, 2):
        rows = np.flatnonzero(it["period"] == p)
        if not len(rows):
            continue
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        starts = list(range(lo, max(lo, hi - w) + 1, max(1, w // 2)))
        if starts[-1] + w < hi:
            starts.append(hi - w)
        spans += [(a, min(a + w, hi)) for a in starts]
    for i in range(0, len(spans), batch_size):
        part = spans[i:i + batch_size]
        batch = stack([(it, a, b, False) for a, b in part], w)
        logit, change = model(**{k: v for k, v in _tensors(batch, dev).items()
                                 if k in ("players", "pmask", "ball", "ctx", "valid")})
        logit, change = logit.cpu().numpy(), change.cpu().numpy()
        for j, (a, b) in enumerate(part):
            total[a:b] += logit[j, :b - a]
            ch_total[a:b] += change[j, :b - a]
            count[a:b] += 1
    count = np.maximum(count, 1)
    return total / count, ch_total / count
