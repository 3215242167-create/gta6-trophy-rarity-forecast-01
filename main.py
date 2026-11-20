"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads one case (schema makefun/gta6-trophies-case/v1): every valid read the
house made of Exophase's GTA VI PS5 trophy page before the checkpoint, and
forecasts each trophy's earn rate (percent) at the case's target as five
quantiles (5/25/50/75/95%). Python 3.13, standard library only, no network,
deterministic: two runs on the same case write the same bytes.

Method
------
Per trophy, work on the logit scale, logit(p) = ln(p / (100 - p)) with p in
percent clamped to [0.01, 99.99] (the scorer's own clamp), and model the rate
against log time since release, x = ln(1 + hours since 2026-11-19T00:00:00Z):

* The history is first reduced to distinct snapshots: Exophase refreshes on
  its own schedule, so long stretches of reads are byte-identical and carry
  no information; a repeated snapshot must not outweigh a real move.
* On the distinct snapshots, fit a recency-weighted least-squares line
  logit(rate) = a + b * x (half-life 24 h), using only the last 24 h of
  snapshots when there are at least three of them. The slope is clamped to
  [0, 1.2]: earn rates rise as more players play and progress, and a negative
  slope fitted across an Exophase refresh (a step drop from players-tracked
  dilution) would wrongly extrapolate a decline. With fewer than three
  distinct snapshots, or no movement at all, the median is the last read's
  rate (persistence): that is exactly the practice cases' situation, where
  every read is the same unrefreshed page.
* The median at the target is the fitted line read at the target's x,
  clamped so the rate stays within [0.2x, 25x] its last read (and in
  [0.01, 99.99]); a trophy at 100% stays at ~100%.
* When the history has never moved (a single distinct snapshot), no trend
  can be fitted, but the median still sits 0.8 logit below the last read:
  the next Exophase refresh is expected to dilute, because players-tracked
  only grows and newly tracked players have earned little yet (launch
  night: 638 -> 3,868 tracked, every rate but the 100% one lower). The
  100% trophy is exempt. This only fires on a flat history; the final
  case's moving history is governed by the trend fit instead.

Spread (logit sigma) combines three honest sources of uncertainty:

* a random walk, 0.5 * sqrt(horizon in days) -- the house baseline's scale;
* the fit's residual scatter (floored at 0.15, capped at 1.0);
* an unseen-refresh term of 1.0 logit: Exophase can refresh between the
  checkpoint and the target, and a refresh reprices every rate at once
  (launch night: players tracked 638 -> 3,868, every rate but the 100% one
  lower). The practice cases straddle exactly such a refresh, and the house
  baseline's narrow band (sigma 0.125 logit over 90 minutes) scores ~1.51
  there because the whole answer falls outside it.

Quantiles are median +/- z_tau * sigma on the logit scale, mapped back to
percent and rounded to 4 decimals half-up (as the starter rounds).

Fallbacks keep every case valid: a trophy with no usable rate falls back to
a tier prior (platinum 0.5%, gold 2%, silver 5%, bronze 10%) with a wide
spread; a case with no history at all uses the tier priors for every trophy;
any unexpected error while forecasting one case still writes a persistence
forecast for it instead of crashing.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Spread components, in logit units.
SIGMA_RW = 0.5        # random-walk scale per sqrt(day), as the house baseline
SIGMA_REFRESH = 1.0   # unseen Exophase refresh between checkpoint and target
RESID_FLOOR = 0.15
RESID_CAP = 1.0
SIGMA_FALLBACK = 2.0  # no usable history at all for the trophy

# Trend-fit guards.
SLOPE_MAX = 1.2
FIT_HALF_LIFE_H = 24.0
FIT_WINDOW_H = 24.0
MEDIAN_MIN_RATIO = 0.2   # median rate vs last read
MEDIAN_MAX_RATIO = 25.0
FLAT_SHIFT = 0.8  # median logit shift down when history never moved

TIER_PRIOR = {"platinum": 0.5, "gold": 2.0, "silver": 5.0, "bronze": 10.0}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(y: float) -> float:
    if y >= 0:
        e = math.exp(-y)
        return 100.0 / (1.0 + e)
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(when: datetime) -> float:
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"level {p} outside (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def distinct_snapshots(history: list) -> list:
    """History reads reduced to snapshots whose rates actually changed.

    Returns [(hours_since_release, rates_tuple)] in time order. A snapshot
    whose rates equal the previous kept one carries no information.
    """
    out = []
    prev = None
    for read in history:
        try:
            when = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            continue
        key = tuple(rates)
        if key == prev:
            continue
        prev = key
        out.append((hours_since_release(when), tuple(rates)))
    return out


def fit_trophy(points: list, x_target: float, last_rate: float):
    """Fit logit(rate) = a + b * ln(1 + t) on distinct snapshots.

    points: [(x, y_logit, weight)] in time order. Returns (median_logit,
    resid_sigma). Falls back to persistence when the fit is not trustworthy.
    """
    y_last = logit(last_rate)
    if len(points) < 3:
        return y_last, RESID_FLOOR
    xs = [p[0] for p in points]
    if max(xs) - min(xs) < 0.05:
        return y_last, RESID_FLOOR
    # Weighted least squares for the slope only; the level is anchored at
    # the last read, so a refresh step inside the window cannot drag the
    # median below where the trophy actually stands.
    sw = sum(p[2] for p in points)
    mx = sum(p[2] * p[0] for p in points) / sw
    my = sum(p[2] * p[1] for p in points) / sw
    sxx = sum(p[2] * (p[0] - mx) ** 2 for p in points)
    if sxx <= 1e-12:
        return y_last, RESID_FLOOR
    sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in points)
    slope = sxy / sxx
    slope = min(SLOPE_MAX, max(0.0, slope))
    x_last = max(xs)
    resid2 = sum(p[2] * (p[1] - (y_last + slope * (p[0] - x_last))) ** 2 for p in points) / sw
    resid = min(RESID_CAP, max(RESID_FLOOR, math.sqrt(max(0.0, resid2))))
    y_med = y_last + slope * (x_target - x_last)
    # Keep the median within a sane band around the last read rate.
    lo = logit(last_rate * MEDIAN_MIN_RATIO)
    hi = logit(last_rate * MEDIAN_MAX_RATIO)
    y_med = min(hi, max(lo, y_med))
    return y_med, resid


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zs = [norm_ppf(float(lv)) for lv in levels]
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))

    snaps = distinct_snapshots(case.get("history") or [])
    t_last = snaps[-1][0] if snaps else 0.0
    # Fit window: last 24 h of snapshots when there are at least three.
    window = [s for s in snaps if t_last - s[0] <= FIT_WINDOW_H]
    fit_snaps = window if len(window) >= 3 else snaps

    rows = []
    for i, trophy in enumerate(trophies):
        tier = str(trophy.get("tier", "")).lower()
        prior = TIER_PRIOR.get(tier, 5.0)
        # Last non-null rate for this trophy, and its fit points.
        last_rate = None
        points = []
        for t, rates in fit_snaps:
            if i >= len(rates):
                continue
            r = rates[i]
            if r is None:
                continue
            last_rate = r
            w = 0.5 ** ((t_last - t) / FIT_HALF_LIFE_H)
            points.append((math.log1p(t), logit(float(r)), w))
        if last_rate is None:
            y_med, sigma = logit(prior), SIGMA_FALLBACK
        else:
            y_med, resid = fit_trophy(points, x_target, float(last_rate))
            if len(snaps) <= 1 and float(last_rate) < 99.0:
                # No movement ever observed: the next refresh is expected
                # to dilute (see module docstring); the trend fit cannot
                # say so, because there is no trend to fit.
                y_med -= FLAT_SHIFT
            sigma = math.sqrt((SIGMA_RW * math.sqrt(max(0.0, horizon_days))) ** 2
                              + resid ** 2 + SIGMA_REFRESH ** 2)
        qs = []
        prev_q = -1.0
        for z in zs:
            q = round4(clamp_pct(inv_logit(y_med + z * sigma)))
            q = float(q)
            if q < prev_q:  # rounding can never decrease this, but stay safe
                q = prev_q
            qs.append(q)
            prev_q = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case: dict) -> dict:
    """Persistence forecast: every quantile equals the last read's rate.

    Used when forecasting a case raises, so the case stays valid.
    """
    rows = []
    history = case.get("history") or []
    last_rates = history[-1].get("rates") if history else None
    for i, trophy in enumerate(case["trophies"]):
        rate = None
        if last_rates and i < len(last_rates):
            rate = last_rates[i]
        if rate is None:
            rate = TIER_PRIOR.get(str(trophy.get("tier", "")).lower(), 5.0)
        rate = round4(clamp_pct(float(rate)))
        rows.append({"id": trophy["id"], "q": [rate] * 5})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = fallback_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
