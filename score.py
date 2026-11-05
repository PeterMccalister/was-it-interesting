"""
WII scoring, v1 (draft).

Usage (run from the project root):
  python scripts/score.py fetch 2018 2019 2020 2021 2024
      Downloads each race via FastF1, computes the raw factors and caches them
      in data/raw_factors.json. Races already cached are skipped.

  python scripts/score.py rank
      Percentile-normalises every cached race against all cached races,
      combines the factors with the v1 weights and prints a ranked list,
      followed by a report on where your calibration anchors landed.

  python scripts/score.py rank --season 2021 --top 30
      Same pool for the percentiles, but only show one season / the top N.

Nothing here writes to Supabase.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_PATH = DATA / "raw_factors.json"
CACHE_DIR = DATA / "fastf1_cache"

SCORE_VERSION = "v1-draft"

# factor: (weight, higher_raw_value_is_more_exciting)
FACTORS = {
    "overtakes": (0.25, True),
    "close_finish": (0.20, False),  # raw = avg P1-P2 gap in seconds, smaller is better
    "lead_changes": (0.15, True),
    "grid_shuffle": (0.15, True),
    "neutralisations": (0.15, True),
    "weather": (0.10, True),
}
NEUTRALISATION_CAP = 3
FINAL_GAP_LAPS = 5

# Calibration anchors: (season, keyword matched against event/location/country, label)
ANCHORS = [
    (2018, "azerbaijan", "classic"),
    (2019, "germany", "classic"),
    (2019, "brazil", "classic"),
    (2020, "monza", "classic"),
    (2021, "brazil", "classic"),
    (2021, "hungary", "classic"),
    (2024, "brazil", "classic"),
    (2019, "spain", "processional"),
    (2020, "spain", "processional"),
    (2021, "spain", "processional"),
    (2024, "monaco", "processional"),
    (2018, "abu dhabi", "processional"),
]


# --------------------------------------------------------------------------
# Factor computation. These take plain DataFrames so they can be tested
# without any network access.
# --------------------------------------------------------------------------

def _num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _position_table(laps):
    """Rows = lap number, columns = driver, values = position at the end of that lap."""
    return laps.pivot_table(index="LapNumber", columns="Driver", values="Position", aggfunc="first")


def _pit_laps(laps):
    """Set of (driver, lap) where the driver entered or left the pits."""
    pitted = laps[laps["PitInTime"].notna() | laps["PitOutTime"].notna()]
    return set(zip(pitted["Driver"], pitted["LapNumber"]))


def _neutral_laps(laps):
    """Laps run (even partly) under SC, VSC or red flag."""
    mask = laps["TrackStatus"].astype(str).str.contains("[4567]", regex=True)
    return set(laps.loc[mask, "LapNumber"])


def count_overtakes(laps):
    """
    On-track passes between consecutive laps. For each lap, take the cars that
    did not touch the pit lane on that lap or the one before, and count the pairs
    whose relative order swapped. Lap 1 and neutralised laps are skipped.
    """
    pos = _position_table(laps)
    pitted = _pit_laps(laps)
    neutral = _neutral_laps(laps)
    total = 0
    for lap in pos.index:
        if lap < 2 or lap in neutral or (lap - 1) not in pos.index:
            continue
        prev, cur = pos.loc[lap - 1], pos.loc[lap]
        drivers = [
            d for d in pos.columns
            if pd.notna(prev[d]) and pd.notna(cur[d])
            and (d, lap) not in pitted and (d, lap - 1) not in pitted
        ]
        for i, a in enumerate(drivers):
            for b in drivers[i + 1:]:
                if (prev[a] < prev[b]) != (cur[a] < cur[b]):
                    total += 1
    return total


def count_lead_changes(laps):
    """Changes of race leader that were not caused by the old leader pitting or retiring."""
    pos = _position_table(laps)
    pitted = _pit_laps(laps)
    leaders = {}
    for lap in pos.index:
        p1 = pos.loc[lap][pos.loc[lap] == 1]
        if len(p1) == 1:
            leaders[lap] = p1.index[0]
    changes = 0
    for lap in sorted(leaders):
        if (lap - 1) not in leaders:
            continue
        old, new = leaders[lap - 1], leaders[lap]
        if old == new:
            continue
        if (old, lap) in pitted or (old, lap - 1) in pitted:
            continue
        if pd.isna(pos.loc[lap].get(old)):  # old leader retired
            continue
        changes += 1
    return changes


def average_final_gap(laps, results, n_laps=FINAL_GAP_LAPS):
    """Mean absolute gap (seconds) between the classified P1 and P2 over the last n laps."""
    fin = results.copy()
    fin["Position"] = pd.to_numeric(fin["Position"], errors="coerce")
    fin = fin.dropna(subset=["Position"]).sort_values("Position")
    if len(fin) < 2:
        return None
    d1, d2 = fin.iloc[0]["Abbreviation"], fin.iloc[1]["Abbreviation"]
    t = laps.assign(_t=laps["Time"].dt.total_seconds()).pivot_table(
        index="LapNumber", columns="Driver", values="_t", aggfunc="first"
    )
    if d1 not in t.columns or d2 not in t.columns:
        return None
    last = int(laps.loc[laps["Driver"] == d1, "LapNumber"].max())
    gaps = []
    for lap in range(last - n_laps + 1, last + 1):
        if lap in t.index and pd.notna(t.loc[lap, d1]) and pd.notna(t.loc[lap, d2]):
            gaps.append(max(abs(t.loc[lap, d2] - t.loc[lap, d1]), 0.05))
    return float(np.mean(gaps)) if gaps else None


def grid_shuffle(results):
    """1 - Spearman rank correlation between grid and finish, classified finishers only."""
    r = results.copy()
    r["Position"] = pd.to_numeric(r["Position"], errors="coerce")
    r["GridPosition"] = pd.to_numeric(r["GridPosition"], errors="coerce")
    if "ClassifiedPosition" in r:
        r = r[pd.to_numeric(r["ClassifiedPosition"], errors="coerce").notna()]
    r = r.dropna(subset=["Position", "GridPosition"])
    if len(r) < 5:
        return None
    grid = r["GridPosition"].where(r["GridPosition"] > 0, r["GridPosition"].max() + 1)  # pit-lane starts go last
    rho = np.corrcoef(grid.rank(), r["Position"].rank())[0, 1]
    return None if np.isnan(rho) else float(1 - rho)


def count_neutralisations(track_status, race_start=None):
    """Number of SC (4), red flag (5) and VSC (6) periods during the race."""
    ts = track_status
    if race_start is not None:
        ts = ts[ts["Time"] >= race_start]
    count, prev = 0, None
    for s in ts["Status"].astype(str):
        if s in ("4", "5", "6") and s != prev:
            count += 1
        prev = s
    return count


def rain_share(weather, race_start=None, race_end=None):
    """Share of weather samples during the race with rainfall."""
    if weather is None or len(weather) == 0 or "Rainfall" not in weather:
        return None
    w = weather
    if race_start is not None and race_end is not None:
        w = w[(w["Time"] >= race_start) & (w["Time"] <= race_end)]
    if len(w) == 0:
        return None
    return float(w["Rainfall"].astype(bool).mean())


def raw_factors(laps, results, track_status, weather):
    laps = laps.dropna(subset=["LapNumber"]).copy()
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    start = laps["LapStartTime"].min() if "LapStartTime" in laps else pd.NaT
    end = laps["Time"].max()
    start = None if pd.isna(start) else start
    end = None if pd.isna(end) else end
    return {
        "overtakes": _num(count_overtakes(laps)),
        "close_finish": _num(average_final_gap(laps, results)),
        "lead_changes": _num(count_lead_changes(laps)),
        "grid_shuffle": _num(grid_shuffle(results)),
        "neutralisations": _num(count_neutralisations(track_status, start)),
        "weather": _num(rain_share(weather, start, end)),
    }


# --------------------------------------------------------------------------
# Fetching (needs network access to FastF1's data sources)
# --------------------------------------------------------------------------

def load_raw():
    if RAW_PATH.exists():
        return json.loads(RAW_PATH.read_text())
    return {}


def save_raw(raw):
    DATA.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False))


def fetch_season(season, raw):
    import fastf1  # imported here so `rank` works without it

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    now = pd.Timestamp.now()
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        key = f"{season}-{rnd:02d}"
        if key in raw and "error" not in raw[key]:
            continue
        if ev["EventDate"] > now:
            continue
        meta = {
            "season": season,
            "round": rnd,
            "event": str(ev["EventName"]),
            "location": str(ev["Location"]),
            "country": str(ev["Country"]),
        }
        try:
            session = fastf1.get_session(season, rnd, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            meta["raw"] = raw_factors(session.laps, session.results, session.track_status, session.weather_data)
            print(f"{key}  {meta['event']}: ok")
        except Exception as exc:  # keep going, retry failed races on the next run
            meta["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{key}  {meta['event']}: FAILED ({meta['error']})")
        raw[key] = meta
        save_raw(raw)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _transform(factor, value):
    # Percentiles ignore monotone transforms (so no log is needed for the gap),
    # but the neutralisation cap does change the ranking, so it lives here.
    if factor == "neutralisations":
        return min(value, NEUTRALISATION_CAP)
    return value


def _percentile(value, reference, higher_is_better):
    ref = np.asarray(reference, dtype=float)
    return float((ref < value).mean() if higher_is_better else (ref > value).mean())


def score_races(raw):
    ok = {k: v for k, v in raw.items() if "error" not in v and "raw" in v}
    reference = {
        f: [_transform(f, r["raw"][f]) for r in ok.values() if r["raw"].get(f) is not None]
        for f in FACTORS
    }
    rows = []
    for key, r in ok.items():
        norms = {}
        for f, (_, higher) in FACTORS.items():
            v = r["raw"].get(f)
            if v is None or len(reference[f]) < 2:
                continue
            norms[f] = _percentile(_transform(f, v), reference[f], higher)
        if not norms:
            continue
        wsum = sum(FACTORS[f][0] for f in norms)  # re-spread weight if a factor is missing
        score = 10 * sum(FACTORS[f][0] * n for f, n in norms.items()) / wsum
        rows.append({
            "key": key,
            "season": r["season"],
            "round": r["round"],
            "event": r["event"],
            "location": r["location"],
            "country": r["country"],
            "score": score,
            "norms": norms,
            "raw": r["raw"],
            "missing": [f for f in FACTORS if f not in norms],
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    for r in rows:
        r["anchor"] = anchor_label(r)
    return rows


def anchor_label(r):
    hay = f"{r['event']} {r['location']} {r['country']}".lower()
    for season, keyword, label in ANCHORS:
        if r["season"] == season and keyword in hay:
            return label
    return None


def print_ranking(rows, season=None, top=None):
    shown = [(i + 1, r) for i, r in enumerate(rows) if season is None or r["season"] == season]
    if top:
        shown = shown[:top]
    print(f"{'#':>4}  {'race':<7} {'event':<28} {'score':>5}  note")
    for rank, r in shown:
        note = r["anchor"] or ""
        if r["missing"]:
            note += f" (missing: {', '.join(r['missing'])})"
        print(f"{rank:>4}  {r['key']:<7} {r['event'][:28]:<28} {r['score']:>5.2f}  {note}".rstrip())


def print_anchor_report(rows):
    n = len(rows)
    print(f"\nCalibration anchors ({n} races scored, score {SCORE_VERSION})")
    means = {}
    for label in ("classic", "processional"):
        sel = [(i + 1, r) for i, r in enumerate(rows) if r["anchor"] == label]
        if not sel:
            continue
        means[label] = float(np.mean([r["score"] for _, r in sel]))
        print(f"  {label}: mean score {means[label]:.2f}")
        for rank, r in sel:
            print(f"    rank {rank:>3}/{n}  {r['key']}  {r['event']}  ({r['score']:.2f})")
    if len(means) == 2:
        print(f"  separation (classic - processional): {means['classic'] - means['processional']:+.2f}")


def cmd_fetch(args):
    raw = load_raw()
    for season in args.seasons:
        fetch_season(season, raw)
    print(f"\nCached {sum('raw' in v for v in raw.values())} races in {RAW_PATH}")


def cmd_rank(args):
    raw = load_raw()
    if not raw:
        raise SystemExit("No cached races yet. Run: python scripts/score.py fetch 2021")
    rows = score_races(raw)
    print_ranking(rows, season=args.season, top=args.top)
    print_anchor_report(rows)
    failed = [k for k, v in raw.items() if "error" in v]
    if failed:
        print(f"\n{len(failed)} race(s) failed to load and are excluded: {', '.join(sorted(failed))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download races and cache raw factors")
    f.add_argument("seasons", type=int, nargs="+")
    f.set_defaults(func=cmd_fetch)
    r = sub.add_parser("rank", help="score cached races and print a ranked list")
    r.add_argument("--season", type=int)
    r.add_argument("--top", type=int)
    r.set_defaults(func=cmd_rank)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
