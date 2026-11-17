"""
The network: attention between the players of each step, then a transformer over time.
It reads the match from both teams' side with the same weights, so swapping teams swaps the answer.

1) Tokens per step (features.py): one per player, one for the ball, one for context.
   Why: in BAR a tracking slot isn't a person and the number of players changes all the time, so
   the input is a set of "someone of this team, here, moving like this", not a fixed list.

2) Set encoder (SetEncoder), per step: attention between context, ball and players.
   Why: possession is about relations (who is closest to the ball, who is pressing him, where his
   team-mates are). Some heads only look at nearby players (distance bias) for the local duel; the
   rest look at everyone for the team shape. Attention doesn't care about order or count of tokens.

3) Two team views (views): the same encoder reads the step as is and with the pitch turned 180
   degrees and the teams swapped.
   Why: P(H) comes from the difference of the two views, so swapping the teams swaps the answer
   exactly. The model can't learn "the home team has the ball", and every match teaches it twice.

4) Temporal transformer (Temporal): attention across the steps of a 30 s window, both directions.
   Why: one frame is ambiguous (a loose ball, the ball not seen); what happens before and after
   tells who really controls it. Attention fades with time distance (ALiBi), faster in some heads
   than others. Dead steps stay in the window: how a restart is set up says who restarts.

5) Two heads: possession (logit of H) and change of possession (is there a change here).
   Why: the HSMM needs both. The possession head says who has the ball, the change head says
   where exactly a possession ends.
"""

import math

import torch
import torch.nn as nn

from possession_model import features as fx

FLIP_P = torch.tensor(fx.PLAYER_FLIP)
FLIP_B = torch.tensor(fx.BALL_FLIP)


def _swap(t, i, j):
    t = t.clone()
    t[..., [i, j]] = t[..., [j, i]]
    return t


def views(players, ball, ctx):
    # Home view as is, and the away view: the pitch turned 180 degrees and the teams swapped
    p2, b2, c2 = players.clone(), ball.clone(), ctx.clone()
    p2[..., FLIP_P] = -p2[..., FLIP_P]
    p2 = _swap(p2, fx.P["d_plus"], fx.P["d_minus"])
    p2[..., fx.P["home"]] = 1.0 - p2[..., fx.P["home"]]
    b2[..., FLIP_B] = -b2[..., FLIP_B]
    b2 = _swap(b2, fx.B["d_plus"], fx.B["d_minus"])
    c2 = _swap(c2, fx.C["n_home"], fx.C["n_away"])
    return torch.stack([players, p2]), torch.stack([ball, b2]), torch.stack([ctx, c2])


class FastDropout(nn.Module):
    # Same as nn.Dropout but about twice as fast on CPU

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        return x * (torch.rand_like(x) >= self.p) / (1 - self.p)


class Block(nn.Module):
    # A standard transformer block that takes an extra attention bias

    def __init__(self, d, heads, ff_mult, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        # No dropout on the attention weights: on CPU it took 3/4 of each training step
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), FastDropout(dropout),
                                nn.Linear(ff_mult * d, d))
        self.drop = FastDropout(dropout)

    def forward(self, x, bias):
        y = self.ln1(x)
        x = x + self.drop(self.attn(y, y, y, attn_mask=bias, need_weights=False)[0])
        return x + self.drop(self.ff(self.ln2(x)))


def _keep_self(hidden):
    # Hide the hidden tokens, but let every token see itself so no attention row is empty
    n = hidden.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=hidden.device)
    return hidden[:, None, :].expand(-1, n, -1) & ~eye


class SetEncoder(nn.Module):
    # Attention between context, ball and players of one step. Some heads look mostly at nearby
    # players (distance bias), the rest look at everyone.

    def __init__(self, cfg):
        super().__init__()
        e = cfg["model"]["encoder"]
        d, self.heads = e["d"], e["heads"]
        self.biased, self.scale = e["biased_heads"], e["dist_scale_m"]
        self.cls = nn.Linear(len(fx.CTX_COLS), d)
        self.ball = nn.Linear(len(fx.BALL_COLS), d)
        self.player = nn.Linear(len(fx.PLAYER_COLS), d)
        self.kind = nn.Parameter(torch.zeros(3, d))              # [CLS], ball, player
        nn.init.normal_(self.kind, std=0.02)
        self.rate = nn.Parameter(torch.full((max(self.biased, 1),), math.log(math.e - 1)))
        self.blocks = nn.ModuleList(Block(d, self.heads, e["ff_mult"], e["dropout"])
                                    for _ in range(e["layers"]))
        self.out = nn.LayerNorm(d)

    def forward(self, players, pmask, ball, ctx):
        m, n_p = pmask.shape
        x = torch.cat([(self.cls(ctx) + self.kind[0])[:, None],
                       (self.ball(ball) + self.kind[1])[:, None],
                       self.player(players) + self.kind[2]], dim=1)
        seen = ball[:, fx.B["seen"]] > 0.5
        valid = torch.cat([torch.ones(m, 2, dtype=torch.bool, device=x.device), pmask], dim=1)
        pos = torch.cat([torch.zeros(m, 1, 2, device=x.device),
                         ball[:, None, [fx.B["x"], fx.B["y"]]],
                         players[..., [fx.P["x"], fx.P["y"]]]], dim=1) * fx.SCALE_POS
        has_pos = torch.cat([torch.zeros(m, 1, dtype=torch.bool, device=x.device), seen[:, None], pmask], dim=1)
        n = n_p + 2
        bias = torch.zeros(m, self.heads, n, n, device=x.device)
        if self.biased:
            dist = torch.cdist(pos, pos) / self.scale
            both = has_pos[:, :, None] & has_pos[:, None, :]
            rate = nn.functional.softplus(self.rate)[: self.biased]
            bias[:, : self.biased] = -rate[None, :, None, None] * (dist * both)[:, None]
        bias = bias.masked_fill(_keep_self(~valid)[:, None], float("-inf"))
        bias = bias.reshape(m * self.heads, n, n)
        for block in self.blocks:
            x = block(x, bias)
        return self.out(x[:, 0])


class Temporal(nn.Module):
    # Transformer over time. Attention fades with distance in time (ALiBi), faster in some heads
    # than others. Dead steps are kept: how a restart is set up says who restarts.

    def __init__(self, cfg):
        super().__init__()
        t, d = cfg["model"]["temporal"], cfg["model"]["encoder"]["d"]
        self.heads = t["heads"]
        self.register_buffer("slopes", torch.tensor([2 ** (-8 * (h + 1) / self.heads)
                                                     for h in range(self.heads)]))
        self.blocks = nn.ModuleList(Block(d, self.heads, t["ff_mult"], t["dropout"])
                                    for _ in range(t["layers"]))
        self.out = nn.LayerNorm(d)

    def forward(self, e, hidden):
        m, n, _ = e.shape
        idx = torch.arange(n, device=e.device)
        gap = (idx[None, :] - idx[:, None]).abs().float()
        bias = (-self.slopes[:, None, None] * gap[None]).expand(m, -1, -1, -1).clone()
        bias = bias.masked_fill(_keep_self(hidden)[:, None], float("-inf"))
        bias = bias.reshape(m * self.heads, n, n)
        for block in self.blocks:
            e = block(e, bias)
        return self.out(e)


class PossessionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["model"]["encoder"]["d"]
        self.encoder = SetEncoder(cfg)
        self.temporal = Temporal(cfg) if cfg["model"]["temporal"]["layers"] > 0 else None
        self.possession = nn.Linear(d, 1)
        self.change = nn.Linear(d, 1)

    def forward(self, players, pmask, ball, ctx, valid):
        # Returns logit(H) = home view minus away view, and the change logit averaged over both
        b, t = valid.shape
        pv, bv, cv = views(players, ball, ctx)                       # (2, B, T, ...)
        mask = pmask[None].expand(2, -1, -1, -1)
        e = self.encoder(pv.reshape(2 * b * t, *pv.shape[3:]), mask.reshape(2 * b * t, -1),
                         bv.reshape(2 * b * t, -1), cv.reshape(2 * b * t, -1)).reshape(2 * b, t, -1)
        if self.temporal is not None:
            e = self.temporal(e, (~valid).repeat(2, 1))
        logit = self.possession(e).squeeze(-1)
        c = self.change(e).squeeze(-1)
        return logit[:b] - logit[b:], 0.5 * (c[:b] + c[b:])
