#!/usr/bin/env python3
"""
closed_loop_backtest.py  (V5: stratified explore + one-time global scan + fair dense "full" optimum)

This backtest treats an NPZ dataset as a *measurement oracle* (linear interpolation),
and runs a closed-loop amplitude calibration per qubit under a hard measurement budget.

What V5 adds (vs V4):

1) Stratified EXPLORE:
   - Force exploration proposals to include at least one point from the LOW half
     and one point from the HIGH half (relative to split_amp), then fill remaining
     proposals by farthest-from-measured. This helps avoid missing lobes/basins.

2) One-time GLOBAL SCAN measurement:
   - Once the surrogate has enough points, the first time we enter EXPLOIT we do a
     global dense-grid scan of surrogate predictions and *force one measurement* at
     the predicted global argmax (even if far). This reduces premature commitment
     when a lucky seed looks strong (q2 case).

3) Fair "full-sweep optimum" baseline:
   - Evaluate the "full best" over a dense interpolated grid using the same oracle
     interpolation method (not just discrete sweep points). This makes comparisons
     apples-to-apples when online can sample off-grid.

Run example:
  uv run closed_loop_backtest.py \
    --npz ./data/amp_sweep_bandpower_bins2_40.npz \
    --budget 10 --seed-points 5 --lambda-xtalk 0.5 \
    --norm-mode oracle \
    --explore-first-k-steps 2 --refine-last-k-steps 2 \
    --proposals 5 \
    --eval-grid 2001 \
    --global-scan-grid 401 \
    --device cpu --seed 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Robust stats helpers
# -----------------------------

def robust_med_mad(x: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad + eps
    return med, scale

def robust_med_iqr(x: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    iqr = (q75 - q25)
    scale = (iqr / 1.349) + eps
    return med, scale

def zscore(x: float, center: float, scale: float) -> float:
    return (float(x) - float(center)) / float(scale)


# -----------------------------
# Oracle: linear interpolation over sweep points
# -----------------------------

@dataclass
class DriveOracle:
    drive_k: int
    amps_sorted: np.ndarray      # [N] sorted ascending
    score_mat_sorted: np.ndarray # [N, Q]

    def query_all_scores(self, amp: float) -> np.ndarray:
        a = float(amp)
        x = self.amps_sorted
        Y = self.score_mat_sorted
        if a <= x[0]:
            return Y[0].copy()
        if a >= x[-1]:
            return Y[-1].copy()
        j = int(np.searchsorted(x, a))
        i = j - 1
        x0, x1 = float(x[i]), float(x[j])
        t = (a - x0) / (x1 - x0 + 1e-18)
        return (1.0 - t) * Y[i] + t * Y[j]

    def query_drive_xtalk(self, amp: float) -> Tuple[float, float]:
        s = self.query_all_scores(amp)  # [Q]
        drive = float(s[self.drive_k])
        if s.size <= 1:
            return drive, 0.0
        others = np.delete(s, self.drive_k)
        xt = float(np.max(others))
        return drive, xt


def build_oracles_from_npz(npz: dict) -> Dict[int, DriveOracle]:
    dq = npz["drive_qubit_idx"].astype(np.int32)
    amp = npz["amplitude"].astype(np.float64)
    score_raw = npz["score_raw"].astype(np.float64)  # [C,Q]
    Qn = score_raw.shape[1]

    oracles: Dict[int, DriveOracle] = {}
    for k in range(Qn):
        mask = (dq == k)
        if not np.any(mask):
            continue
        amps_k = amp[mask]
        scores_k = score_raw[mask, :]
        order = np.argsort(amps_k)
        amps_k = amps_k[order]
        scores_k = scores_k[order, :]
        oracles[k] = DriveOracle(drive_k=k, amps_sorted=amps_k, score_mat_sorted=scores_k)
    return oracles


# -----------------------------
# Normalization modes
# -----------------------------

@dataclass
class OracleNormStats:
    drive_center: float
    drive_scale: float
    xtalk_center: float
    xtalk_scale: float

def precompute_oracle_stats(oracle: DriveOracle, eps: float = 1e-12, use_iqr: bool = False) -> OracleNormStats:
    Y = oracle.score_mat_sorted
    k = oracle.drive_k
    drive = Y[:, k]
    if Y.shape[1] <= 1:
        xtalk = np.zeros_like(drive)
    else:
        xtalk = np.max(np.delete(Y, k, axis=1), axis=1)

    if use_iqr:
        dc, ds = robust_med_iqr(drive, eps=eps)
        xc, xs = robust_med_iqr(xtalk, eps=eps)
    else:
        dc, ds = robust_med_mad(drive, eps=eps)
        xc, xs = robust_med_mad(xtalk, eps=eps)

    return OracleNormStats(dc, ds, xc, xs)

@dataclass
class AnchorNorm:
    anchor_amps: np.ndarray
    eps: float = 1e-12
    drive_center: Optional[float] = None
    drive_scale: Optional[float] = None
    xtalk_center: Optional[float] = None
    xtalk_scale: Optional[float] = None

    def fit_from_oracle(self, oracle: DriveOracle) -> None:
        drives = []
        xtalks = []
        for a in self.anchor_amps.tolist():
            d, x = oracle.query_drive_xtalk(float(a))
            drives.append(d)
            xtalks.append(x)

        d_arr = np.asarray(drives, dtype=np.float64)
        x_arr = np.asarray(xtalks, dtype=np.float64)

        self.drive_center = float(np.median(d_arr))
        self.xtalk_center = float(np.median(x_arr))

        p10d, p90d = float(np.percentile(d_arr, 10)), float(np.percentile(d_arr, 90))
        p10x, p90x = float(np.percentile(x_arr, 10)), float(np.percentile(x_arr, 90))

        self.drive_scale = float((p90d - p10d) / 1.349 + self.eps)
        self.xtalk_scale = float((p90x - p10x) / 1.349 + self.eps)

        self.drive_scale = max(self.drive_scale, 1e-8)
        self.xtalk_scale = max(self.xtalk_scale, 1e-8)

    def z_drive(self, drive_raw: float) -> float:
        assert self.drive_center is not None and self.drive_scale is not None
        return zscore(drive_raw, self.drive_center, self.drive_scale)

    def z_xtalk(self, xtalk_raw: float) -> float:
        assert self.xtalk_center is not None and self.xtalk_scale is not None
        return zscore(xtalk_raw, self.xtalk_center, self.xtalk_scale)

@dataclass
class RollingNorm:
    window: int = 80
    eps: float = 1e-12
    abs_floor: float = 1e-8
    alpha_range: float = 0.20
    buf: List[float] = field(default_factory=list)

    def update(self, x: float) -> None:
        self.buf.append(float(x))
        if len(self.buf) > self.window:
            self.buf = self.buf[-self.window:]

    def ready(self, min_points: int) -> bool:
        return len(self.buf) >= min_points

    def stats(self) -> Tuple[float, float]:
        arr = np.asarray(self.buf, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        mad_scale = 1.4826 * mad
        p10 = float(np.percentile(arr, 10))
        p90 = float(np.percentile(arr, 90))
        range_floor = self.alpha_range * (p90 - p10)
        scale = max(mad_scale, range_floor, self.abs_floor) + self.eps
        return med, scale

    def z(self, x: float) -> float:
        med, scale = self.stats()
        return (float(x) - med) / scale


# -----------------------------
# Tiny MLP surrogate (amp -> utility)
# -----------------------------

class TinyMLP(nn.Module):
    def __init__(self, hidden: int = 32, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = 1
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class PerQubitSurrogate:
    hidden: int = 32
    depth: int = 2
    lr: float = 1e-2
    weight_decay: float = 1e-4
    steps_per_update: int = 60
    batch_size: int = 32
    recency_tau: float = 40.0
    min_points: int = 6
    device: str = "cpu"

    amps: List[float] = field(default_factory=list)
    util: List[float] = field(default_factory=list)
    amp_min_obs: Optional[float] = None
    amp_max_obs: Optional[float] = None

    model: Optional[TinyMLP] = None
    opt: Optional[optim.Optimizer] = None

    def _ensure(self) -> None:
        if self.model is None:
            self.model = TinyMLP(hidden=self.hidden, depth=self.depth).to(self.device)
            self.opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def add_point(self, amp: float, util: float) -> None:
        self.amps.append(float(amp))
        self.util.append(float(util))
        if self.amp_min_obs is None:
            self.amp_min_obs = float(amp)
            self.amp_max_obs = float(amp)
        else:
            self.amp_min_obs = float(min(self.amp_min_obs, amp))
            self.amp_max_obs = float(max(self.amp_max_obs, amp))

    def _scale_amp(self, a: np.ndarray) -> np.ndarray:
        if self.amp_min_obs is None or self.amp_max_obs is None or self.amp_max_obs <= self.amp_min_obs:
            return a.astype(np.float32)
        a01 = (a - self.amp_min_obs) / (self.amp_max_obs - self.amp_min_obs)
        return (2.0 * a01 - 1.0).astype(np.float32)

    def fit(self) -> bool:
        n = len(self.amps)
        if n < self.min_points:
            return False
        self._ensure()
        assert self.model is not None and self.opt is not None

        a = np.asarray(self.amps, dtype=np.float32).reshape(-1, 1)
        y = np.asarray(self.util, dtype=np.float32).reshape(-1, 1)

        idx = np.arange(n, dtype=np.float32)
        age = (n - 1) - idx
        w = np.exp(-age / max(self.recency_tau, 1e-6)).astype(np.float32)
        p = w / (float(np.sum(w)) + 1e-12)

        a_t = torch.from_numpy(self._scale_amp(a)).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)
        w_t = torch.from_numpy(w).to(self.device).reshape(-1, 1)

        self.model.train()
        for _ in range(self.steps_per_update):
            bs = min(self.batch_size, n)
            sel = np.random.choice(n, size=bs, replace=(bs > n), p=p)
            xb = a_t[sel]
            yb = y_t[sel]
            wb = w_t[sel]
            pred = self.model(xb)
            loss = torch.mean(wb * (pred - yb) ** 2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return True

    @torch.no_grad()
    def predict(self, amp: np.ndarray) -> np.ndarray:
        self._ensure()
        assert self.model is not None
        self.model.eval()
        a = np.asarray(amp, dtype=np.float32).reshape(-1, 1)
        x = torch.from_numpy(self._scale_amp(a)).to(self.device)
        y = self.model(x).cpu().numpy().reshape(-1)
        return y.astype(np.float64)

    def trust_region(self) -> Tuple[Optional[float], Optional[float]]:
        return self.amp_min_obs, self.amp_max_obs


# -----------------------------
# Proposal helpers
# -----------------------------

def quantile_seeds(lo: float, hi: float, seed_points: int) -> np.ndarray:
    if seed_points <= 1:
        return np.asarray([(lo + hi) / 2.0], dtype=np.float64)
    if seed_points == 3:
        qs = np.asarray([0.15, 0.50, 0.85], dtype=np.float64)
    elif seed_points == 5:
        qs = np.asarray([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float64)
    else:
        qs = np.linspace(0.10, 0.90, seed_points, dtype=np.float64)
    return lo + qs * (hi - lo)

def fit_quadratic_argmax(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return None
    try:
        A = np.vstack([x**2, x, np.ones_like(x)]).T
        coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b, _c = coeff
        if not np.isfinite(a) or not np.isfinite(b) or a >= 0:
            return None
        return float(-b / (2.0 * a))
    except Exception:
        return None


# -----------------------------
# Closed-loop per qubit
# -----------------------------

@dataclass
class BasinTracker:
    split_amp: Optional[float] = None
    best_low_amp: Optional[float] = None
    best_low_util: float = -np.inf
    best_high_amp: Optional[float] = None
    best_high_util: float = -np.inf

    def update(self, amp: float, util: float, split_amp: float) -> None:
        self.split_amp = float(split_amp)
        if amp <= split_amp:
            if util > self.best_low_util:
                self.best_low_util = float(util)
                self.best_low_amp = float(amp)
        else:
            if util > self.best_high_util:
                self.best_high_util = float(util)
                self.best_high_amp = float(amp)

    def best_other_basin_amp(self, global_best_amp: Optional[float]) -> Optional[float]:
        if self.split_amp is None or global_best_amp is None:
            return None
        if global_best_amp <= self.split_amp:
            return self.best_high_amp
        return self.best_low_amp


@dataclass
class OnlineLoopQubit:
    k: int
    lambda_xtalk: float
    norm_mode: str  # oracle|anchors|rolling
    warmup_points: int
    device: str

    rolling_window: int = 80
    abs_floor: float = 1e-8
    alpha_range: float = 0.20
    proxy_clip: float = 5e-3

    hidden: int = 32
    depth: int = 2
    lr: float = 1e-2
    steps_per_update: int = 60

    oracle_stats: Optional[OracleNormStats] = None
    anchor_norm: Optional[AnchorNorm] = None
    roll_drive: Optional[RollingNorm] = None
    roll_xtalk: Optional[RollingNorm] = None

    amps: List[float] = field(default_factory=list)
    util: List[float] = field(default_factory=list)
    drive_raw: List[float] = field(default_factory=list)
    xtalk_raw: List[float] = field(default_factory=list)

    best_amp: Optional[float] = None
    best_util: float = -np.inf

    basin: BasinTracker = field(default_factory=BasinTracker)
    surrogate: PerQubitSurrogate = field(init=False)

    # V5: one-time global scan control
    did_global_scan: bool = False

    def __post_init__(self) -> None:
        self.surrogate = PerQubitSurrogate(
            hidden=self.hidden,
            depth=self.depth,
            lr=self.lr,
            steps_per_update=self.steps_per_update,
            device=self.device,
        )
        if self.norm_mode == "rolling":
            self.roll_drive = RollingNorm(
                window=self.rolling_window, abs_floor=self.abs_floor, alpha_range=self.alpha_range
            )
            self.roll_xtalk = RollingNorm(
                window=self.rolling_window, abs_floor=self.abs_floor, alpha_range=self.alpha_range
            )

    def utility(self, d_raw: float, x_raw: float) -> Tuple[float, bool]:
        if self.norm_mode == "oracle":
            assert self.oracle_stats is not None
            dz = zscore(d_raw, self.oracle_stats.drive_center, self.oracle_stats.drive_scale)
            xz = zscore(x_raw, self.oracle_stats.xtalk_center, self.oracle_stats.xtalk_scale)
            return float(dz - self.lambda_xtalk * xz), False

        if self.norm_mode == "anchors":
            assert self.anchor_norm is not None
            dz = self.anchor_norm.z_drive(d_raw)
            xz = self.anchor_norm.z_xtalk(x_raw)
            return float(dz - self.lambda_xtalk * xz), False

        assert self.roll_drive is not None and self.roll_xtalk is not None
        self.roll_drive.update(d_raw)
        self.roll_xtalk.update(x_raw)

        if not (self.roll_drive.ready(self.warmup_points) and self.roll_xtalk.ready(self.warmup_points)):
            u = float(d_raw - self.lambda_xtalk * x_raw)
            u = float(np.clip(u, -self.proxy_clip, self.proxy_clip))
            return u, True

        dz = self.roll_drive.z(d_raw)
        xz = self.roll_xtalk.z(x_raw)
        return float(dz - self.lambda_xtalk * xz), False

    def observe(self, amp: float, d_raw: float, x_raw: float, split_amp: float) -> Tuple[float, bool]:
        util, is_proxy = self.utility(d_raw, x_raw)

        self.amps.append(float(amp))
        self.util.append(float(util))
        self.drive_raw.append(float(d_raw))
        self.xtalk_raw.append(float(x_raw))

        self.surrogate.add_point(amp, util)
        self.surrogate.fit()

        if util > self.best_util:
            self.best_util = float(util)
            self.best_amp = float(amp)

        self.basin.update(float(amp), float(util), split_amp=float(split_amp))
        return util, is_proxy

    def _trust_region(self, lo: float, hi: float) -> Tuple[float, float]:
        mlo, mhi = self.surrogate.trust_region()
        if mlo is None or mhi is None or mhi <= mlo:
            return float(lo), float(hi)
        return float(mlo), float(mhi)

    def propose_explore(self, lo: float, hi: float, n: int, split_amp: float) -> np.ndarray:
        """
        V5: Stratified explore.
        - Force one from low half and one from high half of trust region,
          then fill rest by farthest-from-measured.
        """
        tr_lo, tr_hi = self._trust_region(lo, hi)
        grid = np.linspace(tr_lo, tr_hi, 31, dtype=np.float64)

        if grid.size == 0:
            return np.asarray([0.5 * (tr_lo + tr_hi)], dtype=np.float64)

        # Low/high masks relative to split_amp but clipped to trust region
        low_mask = (grid <= split_amp)
        high_mask = (grid > split_amp)

        measured = np.asarray(self.amps, dtype=np.float64) if self.amps else None

        def farthest_pick(cands: np.ndarray) -> Optional[float]:
            if cands.size == 0:
                return None
            if measured is None or measured.size == 0:
                return float(cands[len(cands) // 2])
            d = np.min(np.abs(cands.reshape(-1, 1) - measured.reshape(1, -1)), axis=1)
            return float(cands[int(np.argmax(d))])

        picks: List[float] = []
        a_low = farthest_pick(grid[low_mask])
        a_high = farthest_pick(grid[high_mask])

        if a_low is not None:
            picks.append(a_low)
        if a_high is not None and not any(abs(a_high - p) < 1e-9 for p in picks):
            picks.append(a_high)

        # Fill remaining by global farthest-from-measured
        if measured is None or measured.size == 0:
            # Just spread them if we have no measurements yet
            extra = np.linspace(tr_lo, tr_hi, n, dtype=np.float64).tolist()
        else:
            d_all = np.min(np.abs(grid.reshape(-1, 1) - measured.reshape(1, -1)), axis=1)
            order = np.argsort(d_all)[::-1]
            extra = grid[order].tolist()

        for a in extra:
            if len(picks) >= n:
                break
            if not any(abs(a - p) < 1e-9 for p in picks):
                picks.append(float(a))

        return np.asarray(picks[:n], dtype=np.float64)

    def basin_cover_points(
        self,
        lo: float,
        hi: float,
        split_amp: float,
        m: int = 3,
        tr_only: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return m coverage points in each basin (low/high).
        Uses trust-region bounds by default.
        Points are at 20/50/80% of the basin interval (configurable via m).
        """
        if tr_only:
            tr_lo, tr_hi = self._trust_region(lo, hi)
        else:
            tr_lo, tr_hi = float(lo), float(hi)

        # Clamp split into [tr_lo, tr_hi]
        s = float(np.clip(split_amp, tr_lo, tr_hi))

        # Degenerate cases: if one basin collapses, fall back to uniform in TR
        if s <= tr_lo + 1e-15 or s >= tr_hi - 1e-15:
            g = np.linspace(tr_lo, tr_hi, m, dtype=np.float64)
            return g.copy(), g.copy()

        qs = np.linspace(0.2, 0.8, m, dtype=np.float64)

        low = tr_lo + qs * (s - tr_lo)
        high = s + qs * (tr_hi - s)
        return low.astype(np.float64), high.astype(np.float64)

    def rank_by_surrogate(self, amps: np.ndarray) -> np.ndarray:
        """
        Return amps sorted by surrogate prediction descending.
        """
        amps = np.asarray(amps, dtype=np.float64).reshape(-1)
        if amps.size == 0:
            return amps
        pred = self.surrogate.predict(amps)
        order = np.argsort(pred)[::-1]
        return amps[order]

    def global_scan_basin_argmax(self, lo: float, hi: float, split_amp: float, grid_n: int) -> Tuple[float, float]:
        g = np.linspace(lo, hi, int(grid_n), dtype=np.float64)
        pred = self.surrogate.predict(g)

        low_mask = g <= split_amp
        high_mask = g > split_amp

        # Fallbacks if split produces an empty side (shouldn't happen, but safe)
        if not np.any(low_mask):
            i = int(np.argmax(pred))
            return float(g[i]), float(g[i])
        if not np.any(high_mask):
            i = int(np.argmax(pred))
            return float(g[i]), float(g[i])

        i_low = int(np.argmax(pred[low_mask]))
        i_high = int(np.argmax(pred[high_mask]))

        g_low = g[low_mask]
        g_high = g[high_mask]

        return float(g_low[i_low]), float(g_high[i_high])

    def propose_exploit(self, lo: float, hi: float, n: int) -> np.ndarray:
        tr_lo, tr_hi = self._trust_region(lo, hi)
        grid = np.linspace(tr_lo, tr_hi, 81, dtype=np.float64)
        pred = self.surrogate.predict(grid)
        best_pred_amp = float(grid[int(np.argmax(pred))])

        r = 0.12 * (tr_hi - tr_lo)
        lo2 = max(tr_lo, best_pred_amp - r)
        hi2 = min(tr_hi, best_pred_amp + r)
        local = np.linspace(lo2, hi2, 27, dtype=np.float64)
        local_pred = self.surrogate.predict(local)
        cand = local[np.argsort(local_pred)[::-1]]

        out: List[float] = []
        if self.best_amp is not None:
            out.append(self.best_amp)

        other = self.basin.best_other_basin_amp(self.best_amp)
        if other is not None and not any(abs(other - b) < 1e-6 for b in out):
            out.append(other)

        for a in cand.tolist():
            if len(out) >= n:
                break
            if not any(abs(a - b) < 1e-6 for b in out):
                out.append(a)

        if len(out) < n:
            pad = np.linspace(tr_lo, tr_hi, n, dtype=np.float64)
            for a in pad.tolist():
                if len(out) >= n:
                    break
                if not any(abs(a - b) < 1e-6 for b in out):
                    out.append(a)

        return np.asarray(out[:n], dtype=np.float64)

    def propose_refine(self, lo: float, hi: float, n: int) -> np.ndarray:
        tr_lo, tr_hi = self._trust_region(lo, hi)
        if self.best_amp is None:
            return self.propose_exploit(lo, hi, n=n)

        a0 = float(self.best_amp)
        delta = 0.06 * (tr_hi - tr_lo)
        left = float(np.clip(a0 - delta, tr_lo, tr_hi))
        right = float(np.clip(a0 + delta, tr_lo, tr_hi))

        props: List[float] = [a0]

        amps = np.asarray(self.amps, dtype=np.float64)
        util = np.asarray(self.util, dtype=np.float64)
        win = max(2.5 * delta, 1e-12)
        mask = np.abs(amps - a0) <= win
        if np.sum(mask) >= 3:
            a_hat = fit_quadratic_argmax(amps[mask], util[mask])
            if a_hat is not None:
                a_hat = float(np.clip(a_hat, tr_lo, tr_hi))
                props.append(a_hat)

        props.extend([left, right])

        other = self.basin.best_other_basin_amp(self.best_amp)
        if other is not None:
            props.append(float(other))

        uniq: List[float] = []
        for a in props:
            if not any(abs(a - b) < 1e-6 for b in uniq):
                uniq.append(a)

        if len(uniq) < n:
            extra = self.propose_exploit(lo, hi, n=n)
            for a in extra.tolist():
                if len(uniq) >= n:
                    break
                if not any(abs(a - b) < 1e-6 for b in uniq):
                    uniq.append(a)

        return np.asarray(uniq[:n], dtype=np.float64)

    def pick_unmeasured(
        self,
        candidates: np.ndarray,
        lo: float,
        hi: float,
        prefer: str = "first",  # "first" | "farthest" | "surrogate"
    ) -> float:
        """
        Pick an unmeasured candidate.

        prefer="first"    : old behavior
        prefer="farthest" : maximize distance to nearest measured (best for EXPLORE)
        prefer="surrogate": maximize surrogate prediction (best for EXPLOIT)
        """
        c = np.asarray(candidates, dtype=np.float64).reshape(-1)

        # filter unmeasured
        un = []
        for a in c.tolist():
            if not any(abs(a - b) < 1e-6 for b in self.amps):
                un.append(float(np.clip(a, lo, hi)))

        if len(un) == 0:
            tr_lo, tr_hi = self._trust_region(lo, hi)
            return float(np.random.uniform(tr_lo, tr_hi))

        if prefer == "first":
            return float(un[0])

        if prefer == "surrogate":
            u = np.asarray(un, dtype=np.float64)
            pred = self.surrogate.predict(u)
            return float(u[int(np.argmax(pred))])

        # prefer == "farthest"
        measured = np.asarray(self.amps, dtype=np.float64) if self.amps else None
        if measured is None or measured.size == 0:
            return float(un[len(un) // 2])

        u = np.asarray(un, dtype=np.float64)
        d = np.min(np.abs(u.reshape(-1, 1) - measured.reshape(1, -1)), axis=1)
        return float(u[int(np.argmax(d))])



# -----------------------------
# Full-sweep evaluation utility (dense interpolated grid)
# -----------------------------

def full_sweep_best_amp_util_dense(
    oracle: DriveOracle,
    lambda_xtalk: float,
    grid_n: int,
    use_iqr: bool = False,
) -> Tuple[float, float]:
    stats = precompute_oracle_stats(oracle, use_iqr=use_iqr)
    lo = float(oracle.amps_sorted[0])
    hi = float(oracle.amps_sorted[-1])

    g = np.linspace(lo, hi, int(grid_n), dtype=np.float64)
    drive = np.empty_like(g)
    xtalk = np.empty_like(g)
    for i, a in enumerate(g):
        d, x = oracle.query_drive_xtalk(float(a))
        drive[i] = d
        xtalk[i] = x

    util = (drive - stats.drive_center) / stats.drive_scale - lambda_xtalk * (xtalk - stats.xtalk_center) / stats.xtalk_scale
    j = int(np.argmax(util))
    return float(g[j]), float(util[j])

def full_sweep_util_at_amp(
    oracle: DriveOracle,
    lambda_xtalk: float,
    amp: float,
    use_iqr: bool = False,
) -> float:
    stats = precompute_oracle_stats(oracle, use_iqr=use_iqr)
    d, x = oracle.query_drive_xtalk(float(amp))
    dz = (d - stats.drive_center) / stats.drive_scale
    xz = (x - stats.xtalk_center) / stats.xtalk_scale
    return float(dz - lambda_xtalk * xz)


# -----------------------------
# Main backtest
# -----------------------------

def run_backtest(
    npz_path: str,
    budget: int,
    seed_points: int,
    lambda_xtalk: float,
    norm_mode: str,
    warmup_points: int,
    abs_floor: float,
    alpha_range: float,
    explore_first_k_steps: int,
    refine_last_k_steps: int,
    proposals: int,
    anchor_count: int,
    anchor_strategy: str,
    use_iqr_stats: bool,
    eval_grid: int,
    global_scan_grid: int,
    device: str,
    seed: int,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    npz = dict(np.load(npz_path, allow_pickle=False))
    oracles = build_oracles_from_npz(npz)

    amp_all = npz["amplitude"].astype(np.float64)
    global_lo = float(np.min(amp_all))
    global_hi = float(np.max(amp_all))

    print(f"\nClosed-loop backtest (V5): {npz_path}")
    print(f"budget_per_qubit={budget}  seed_points={seed_points}  lambda_xtalk={lambda_xtalk}")
    print(f"norm_mode={norm_mode}  warmup_points={warmup_points}  proposals={proposals}")
    print(f"use_iqr_stats={use_iqr_stats}")
    print(f"explore_first_k_steps={explore_first_k_steps}  refine_last_k_steps={refine_last_k_steps}")
    print(f"eval_grid={eval_grid}  global_scan_grid={global_scan_grid}")
    if norm_mode == "rolling":
        print(f"rolling abs_floor={abs_floor}  alpha_range={alpha_range}")
    if norm_mode == "anchors":
        print(f"anchors: count={anchor_count} strategy={anchor_strategy}")
    print(f"device={device}  global_amp_range=[{global_lo:.6f}, {global_hi:.6f}]\n")

    results = {}

    for k, oracle in oracles.items():
        lo_k, hi_k = float(oracle.amps_sorted[0]), float(oracle.amps_sorted[-1])
        split_amp = float(np.median(oracle.amps_sorted))

        loop = OnlineLoopQubit(
            k=k,
            lambda_xtalk=lambda_xtalk,
            norm_mode=norm_mode,
            warmup_points=warmup_points,
            device=device,
            abs_floor=abs_floor,
            alpha_range=alpha_range,
        )

        if norm_mode == "oracle":
            loop.oracle_stats = precompute_oracle_stats(oracle, use_iqr=use_iqr_stats)
        elif norm_mode == "anchors":
            if anchor_strategy == "quantiles":
                anchors = quantile_seeds(lo_k, hi_k, anchor_count)
            elif anchor_strategy == "ends_mid":
                if anchor_count <= 1:
                    anchors = np.asarray([(lo_k + hi_k) / 2.0], dtype=np.float64)
                elif anchor_count == 2:
                    anchors = np.asarray([lo_k, hi_k], dtype=np.float64)
                else:
                    anchors = np.linspace(lo_k, hi_k, anchor_count, dtype=np.float64)
            else:
                raise ValueError(f"Unknown anchor_strategy: {anchor_strategy}")

            loop.anchor_norm = AnchorNorm(anchor_amps=anchors)
            loop.anchor_norm.fit_from_oracle(oracle)

        print(f"==================== Drive q{k} ====================")

        seeds = quantile_seeds(lo_k, hi_k, seed_points)
        step = 0

        for a in seeds:
            if step >= budget:
                break
            d, x = oracle.query_drive_xtalk(float(a))
            util, is_proxy = loop.observe(float(a), d, x, split_amp=split_amp)
            step += 1
            if is_proxy:
                print(f"[seed {step:2d}/{budget}] amp={a:.6f}  score_raw={d:.3e}  xtalk_raw={x:.3e}  util_proxy={util:+.3e}")
            else:
                print(f"[seed {step:2d}/{budget}] amp={a:.6f}  score_raw={d:.3e}  xtalk_raw={x:.3e}  util={util:+.3f}")

        while step < budget:
            remaining = budget - step

            if step < seed_points + explore_first_k_steps:
                mode = "EXPLORE"
                props = loop.propose_explore(lo_k, hi_k, n=proposals, split_amp=split_amp)
                a_next = loop.pick_unmeasured(props, lo_k, hi_k, prefer="farthest")

            elif remaining <= refine_last_k_steps:
                mode = "REFINE"
                props = loop.propose_refine(lo_k, hi_k, n=proposals)
                a_next = loop.pick_unmeasured(props, lo_k, hi_k, prefer="first")

            else:
                mode = "EXPLOIT"

                if (not loop.did_global_scan) and (len(loop.amps) >= loop.surrogate.min_points):
                    # V6-style BASINSCAN: basin coverage + surrogate ranking
                    # 1) Build a small candidate set in each basin (coverage points)
                    low_cov, high_cov = loop.basin_cover_points(lo_k, hi_k, split_amp=split_amp, m=3, tr_only=True)

                    # 2) Rank candidates within each basin by surrogate prediction
                    low_ranked = loop.rank_by_surrogate(low_cov)
                    high_ranked = loop.rank_by_surrogate(high_cov)

                    # 3) Choose which basin to "check" first: the OTHER basin from current best
                    if loop.best_amp is None:
                        # If we don't have a best yet, just interleave
                        other_first = True
                        best_is_low = True
                    else:
                        best_is_low = (loop.best_amp <= split_amp)
                        other_first = True  # always check the other basin first

                    if best_is_low:
                        other_list = high_ranked
                        same_list = low_ranked
                    else:
                        other_list = low_ranked
                        same_list = high_ranked

                    # 4) Also include the surrogate global argmax per basin (dense scan)
                    a_low, a_high = loop.global_scan_basin_argmax(lo_k, hi_k, split_amp=split_amp, grid_n=global_scan_grid)

                    # 5) Assemble proposal list:
                    #    - try multiple OTHER-basin points first (coverage-ranked)
                    #    - include basin argmaxes as additional candidates
                    #    - include current best to keep it "in play"

                    # Decide which basin is "other"
                    if loop.best_amp is None:
                        best_is_low = True
                    else:
                        best_is_low = (loop.best_amp <= split_amp)

                    # Candidate pools
                    if best_is_low:
                        other_list = high_ranked
                        other_argmax = a_high
                        same_list = low_ranked
                    else:
                        other_list = low_ranked
                        other_argmax = a_low
                        same_list = high_ranked

                    # --- FORCE: pick ONLY from the other basin on this one step ---
                    other_props_list: List[float] = []
                    for a in other_list.tolist():
                        other_props_list.append(float(a))
                    other_props_list.append(float(other_argmax))

                    # Dedupe other-basin list
                    other_uniq: List[float] = []
                    for a in other_props_list:
                        if not any(abs(a - b) < 1e-6 for b in other_uniq):
                            other_uniq.append(a)

                    other_props = np.asarray(other_uniq, dtype=np.float64)

                    # Pick best predicted *within the other basin*
                    a_next = loop.pick_unmeasured(other_props, lo_k, hi_k, prefer="surrogate")

                    # Still print a richer props list for debugging visibility
                    props_list: List[float] = []
                    if loop.best_amp is not None:
                        props_list.append(float(loop.best_amp))
                    props_list.extend(other_props.tolist())
                    props_list.append(float(a_low))
                    props_list.append(float(a_high))
                    props_list.extend(same_list.tolist())

                    uniq: List[float] = []
                    for a in props_list:
                        if not any(abs(a - b) < 1e-6 for b in uniq):
                            uniq.append(a)
                    props = np.asarray(uniq, dtype=np.float64)

            d, x = oracle.query_drive_xtalk(a_next)
            util, is_proxy = loop.observe(a_next, d, x, split_amp=split_amp)
            step += 1

            # Enable this to force localality in the next steps. For now, global scanning is simply performing better.
            #loop.did_global_scan = True

            if is_proxy:
                print(f"[step {step:2d}/{budget} {mode}] amp={a_next:.6f}  util_proxy={util:+.3e}  (props={np.array2string(props, precision=6)})")
            else:
                print(f"[step {step:2d}/{budget} {mode}] amp={a_next:.6f}  util={util:+.3f}  (props={np.array2string(props, precision=6)})")

        # --- Evaluation: fair dense-grid full optimum ---
        full_best_amp, full_best_util = full_sweep_best_amp_util_dense(
            oracle, lambda_xtalk, grid_n=eval_grid, use_iqr=use_iqr_stats
        )

        online_best_amp = loop.best_amp if loop.best_amp is not None else float(seeds[0])
        util_full_at_online = full_sweep_util_at_amp(oracle, lambda_xtalk, online_best_amp, use_iqr=use_iqr_stats)
        util_full_at_full = full_sweep_util_at_amp(oracle, lambda_xtalk, full_best_amp, use_iqr=use_iqr_stats)

        amp_err = abs(online_best_amp - full_best_amp)

        print("\n--- Summary ---")
        print(f"online best_amp={online_best_amp:.6f}   online best_util(internal)={loop.best_util:+.3f}")
        print(f"full-best (dense) amp={full_best_amp:.6f} best_util(full-norm)={full_best_util:+.3f}")
        print(f"utility(full-norm) @ online_amp = {util_full_at_online:+.3f}")
        print(f"utility(full-norm) @ full_amp   = {util_full_at_full:+.3f}")
        print(f"|amp_online - amp_full| = {amp_err:.6f}\n")

        results[k] = dict(
            online_amp=online_best_amp,
            full_amp=full_best_amp,
            amp_err=amp_err,
            util_full_online=util_full_at_online,
            util_full_full=util_full_at_full,
        )

    print("==================== Overall Summary ====================")
    for k in sorted(results.keys()):
        r = results[k]
        print(
            f"q{k}: online_amp={r['online_amp']:.6f}  "
            f"full_amp={r['full_amp']:.6f}  "
            f"amp_err={r['amp_err']:.6f}  "
            f"util_full(online)={r['util_full_online']:+.3f}  "
            f"util_full(full)={r['util_full_full']:+.3f}"
        )
    print("")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", required=True)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--seed-points", type=int, default=5)
    ap.add_argument("--lambda-xtalk", type=float, default=0.5)
    ap.add_argument("--proposals", type=int, default=5)

    ap.add_argument("--norm-mode", choices=["oracle", "anchors", "rolling"], default="oracle",
                    help="oracle: full-sweep stats (backtest alignment); anchors/rolling: hardware-feasible.")
    ap.add_argument("--use-iqr-stats", action="store_true",
                    help="Use median+IQR-derived scale instead of median+MAD (oracle eval & oracle norm).")

    ap.add_argument("--warmup-points", type=int, default=5,
                    help="Only used for rolling mode readiness.")

    ap.add_argument("--abs-floor", type=float, default=1e-8)
    ap.add_argument("--alpha-range", type=float, default=0.20)

    ap.add_argument("--anchor-count", type=int, default=3)
    ap.add_argument("--anchor-strategy", choices=["quantiles", "ends_mid"], default="quantiles")

    ap.add_argument("--explore-first-k-steps", type=int, default=2)
    ap.add_argument("--refine-last-k-steps", type=int, default=2)

    # V5: fair evaluation + global scan grids
    ap.add_argument("--eval-grid", type=int, default=2001,
                    help="Dense grid size for computing the 'full' optimum via oracle interpolation.")
    ap.add_argument("--global-scan-grid", type=int, default=401,
                    help="Dense grid size for the one-time surrogate global-scan measurement.")

    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    run_backtest(
        npz_path=args.npz,
        budget=args.budget,
        seed_points=args.seed_points,
        lambda_xtalk=args.lambda_xtalk,
        norm_mode=args.norm_mode,
        warmup_points=args.warmup_points,
        abs_floor=args.abs_floor,
        alpha_range=args.alpha_range,
        explore_first_k_steps=args.explore_first_k_steps,
        refine_last_k_steps=args.refine_last_k_steps,
        proposals=args.proposals,
        anchor_count=args.anchor_count,
        anchor_strategy=args.anchor_strategy,
        use_iqr_stats=args.use_iqr_stats,
        eval_grid=args.eval_grid,
        global_scan_grid=args.global_scan_grid,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

