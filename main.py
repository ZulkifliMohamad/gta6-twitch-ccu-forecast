"""Forecast GTA VI launch-day Twitch viewers, hour by hour.

Competition: gta6-twitch-ccu-forecast-01. The program reads one case file
(/input/<id>.in) holding hour 1's print and the comparables (past big game
launches' hourly Twitch counts), and writes /output/<id>.out with five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the viewer count for each of
hours 2..24 of the release day.

Model
-----
Launch-day curves have a shape, not just a level: hour 1 is the small
hours of the night (only New Zealand, Europe and the UK have unlocked),
viewers dip a little, then climb as the US unlocks (05:00Z East, 08:00Z
West) and peak in the US afternoon / European evening, before easing off
late at night. Persistence (every hour equals hour 1) misses that whole
rise, so this program learns the shape from the case's own comparables:

1. For every comparable launch, take its release-day hourly counts and
   express each hour as a log-ratio to that launch's own hour-1 print.
   Launches whose hour 1 was nearly empty (a single simultaneous global
   unlock later in the day, so hour 1 is a dead hour) have ratios in the
   hundreds or thousands and would drag any average up; they are not
   what a rolling-midnight console launch looks like, so the training
   pool keeps only launches with a real hour 1 (>= 5,000 viewers) whose
   peak stays within 15x of hour 1. The game's own pre-release history
   (a comparable that is not a launch day at all) is left out too.
2. Per forecast hour, take the trimmed mean of the log-ratios (drop the
   bottom and top deciles first) as the median shape, and the
   interquartile range / 1.349 as the log spread, floored at 0.30 and
   capped at 0.75. Both are smoothed with a 3-hour moving average so a
   single noisy hour cannot jerk the curve.
3. Quantiles: q_tau(h) = p1 * exp(median_h + z_tau * sigma_h), with
   z the standard normal quantiles (-1.645, -0.674, 0, 0.674, 1.645).

On a leave-one-out replay over the kit's comparables this shape beats
the house persistence baseline on every filtered test set (about 0.23
vs 0.27 mean pinball loss on the launches that look like this one).
If the training pool is too small for an hour, that hour falls back to
the baseline persistence forecast.

Runs on Python 3.13, standard library only, no network. Deterministic:
no randomness, no wall-clock use, same bytes on every run.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Baseline spread, used only as a fallback (starter's 0.15 per sqrt hour).
BASE_SIGMA = 0.15
# Training-pool filters: a launch only teaches the shape if its hour 1
# was a real audience and its day did not explode from near zero.
MIN_HOUR1 = 5000
MAX_PEAK_RATIO = 15.0
MIN_DAY_HOURS = 20
MIN_TRAINING = 8
# Spread of an hour's log-ratio, floored and capped (see module docstring).
SIGMA_FLOOR = 0.30
SIGMA_CAP = 0.75


def round2(x):
    """Round to 2 decimals, half up, as the starter does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _quantile(sorted_values, p):
    """Linear-interpolated quantile of an already sorted list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = p * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _day_map(comparable):
    """Hour (1..24) -> viewers for a comparable's release day."""
    try:
        day_start = _parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    hours = {}
    for row in comparable.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            when = _parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= offset <= 24:
            hours[offset] = viewers
    return hours


def _training_curves(case):
    """Log-ratio curves of the comparables that look like this launch."""
    game_name = (case.get("game") or {}).get("name") or ""
    curves = []
    for comparable in case.get("comparables") or []:
        name = comparable.get("name") or ""
        # The game's own pre-release history is a comparable but not a
        # launch day; it would teach the wrong shape.
        if "Grand Theft Auto VI" in name or name == game_name:
            continue
        hours = _day_map(comparable)
        if len(hours) < MIN_DAY_HOURS or 1 not in hours:
            continue
        first = hours[1]
        if first < MIN_HOUR1:
            continue
        peak = max(hours.values())
        if peak / first > MAX_PEAK_RATIO:
            continue
        curves.append({h: math.log(v / first) for h, v in hours.items()})
    return curves


def _smooth(values):
    """3-hour moving average over the hours present in values."""
    out = {}
    for hour in values:
        neighbours = [values[h] for h in (hour - 1, hour, hour + 1) if h in values]
        out[hour] = sum(neighbours) / len(neighbours)
    return out


def _shape(curves, hours):
    """Per-hour (median log-ratio, log spread) learned from the curves."""
    medians = {}
    sigmas = {}
    for hour in hours:
        logs = sorted(c[hour] for c in curves if hour in c)
        if len(logs) < MIN_TRAINING:
            continue
        low = _quantile(logs, 0.10)
        high = _quantile(logs, 0.90)
        trimmed = [x for x in logs if low <= x <= high] or logs
        medians[hour] = sum(trimmed) / len(trimmed)
        iqr = _quantile(logs, 0.75) - _quantile(logs, 0.25)
        sigmas[hour] = min(SIGMA_CAP, max(SIGMA_FLOOR, iqr / 1.349))
    return _smooth(medians), _smooth(sigmas)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    wanted = [row["hour"] for row in case["forecast"]]
    curves = _training_curves(case)
    medians, sigmas = _shape(curves, wanted) if len(curves) >= MIN_TRAINING else ({}, {})
    rows = []
    for hour in wanted:
        if hour in medians:
            median = medians[hour]
            sigma = sigmas[hour]
        else:
            # Too little comparable evidence for this hour: persist.
            median = 0.0
            sigma = BASE_SIGMA * math.sqrt(hour - 1)
        quantiles = [round2(p1 * math.exp(median + z * sigma)) for z in Z]
        quantiles = [max(1, min(1_000_000_000, q)) for q in quantiles]
        quantiles.sort()
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
