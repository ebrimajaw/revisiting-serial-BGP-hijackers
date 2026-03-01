from __future__ import annotations
import argparse
import gzip
import math
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

"""
- ASN list + (optional) label source. Must contain column "asn".
- If it also contains "class" or "predicted_class", that label is carried along.
- v4/v6 per-ASN daily prefix peer-count dictionaries produced by process_asn
- RIR mapping produced by calculateRIR
- MOAS consolidation produced by consolidated_moas
"""

ASN_LIST_CSV = "/home/grountruth_updated.csv"  # or "predictions set file"
V4_PKL_GZ = "/home/processed4/asn_all_day_pc.pkl.gz"
V6_PKL_GZ = "/home/processed6/asn_all_day_pc.pkl.gz"
ASN_RIR_PKL_GZ = "/home/asn_rir_all.pkl.gz"
MOAS_CONSOLIDATED_PKL_GZ = "moas_consolidated.pkl.gz"

OUTPUT_CSV = "/home/computed_features.csv"


FEATURE_COLUMNS = ["asn","zero_drops", "up_time", "sixmonths_percent", "p10_adTime", "q1_adTime", "median_adTime",
                   "q3_adTime", "p90_adTime","p10_pfx_medianVis", "q1_pfx_medianVis", "median_pfx_medianVis","q3_pfx_medianVis",
                   "p90_pfx_medianVis","highMedVis_percent", "lowMedVis_percent","highMaxVis_percent", "lowMaxVis_percent",
                   "q1_adTime_highMaxVis", "median_adTime_highMaxVis", "q3_adTime_highMaxVis","q1_adTime_lowMaxVis", "median_adTime_lowMaxVis",
                   "q3_adTime_lowMaxVis","q1_adTime_highMedVis", "median_adTime_highMedVis", "q3_adTime_highMedVis","q1_adTime_lowMedVis",
                   "median_adTime_lowMedVis", "q3_adTime_lowMedVis","median_unique_pfx","rir_count", "frac_unassigned_add", "frac_unassigned_pref",
                   "top_rir_add", "top_rir_pref", "gini_rir_add", "gini_rir_pref","frac_moas_add", "frac_moas_pref","std_moas_len",
                   "var_moas_len", "range_moas_len","p1_moas_len", "p99_moas_len"
                   ]


def load_pickle_gz(path: str) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den not in (0, 0.0) else 0.0

def gini_from_counts(counts: Iterable[int]) -> float:
    """
    Gini coefficient for a multiset of non-negative values.
    Returns 0.0 for empty or all-zero vectors.
    """
    x = np.array(list(counts), dtype=float)
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    # Standard Gini formula based on Lorenz curve
    cum = np.cumsum(x_sorted)
    g = (n + 1 - 2 * (cum.sum() / s)) / n
    return float(g)


def percentile_or_zero(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def median_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(values))


def std_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(values))


def var_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.var(values))


def compute_daywise_max_visibility(
    v4: Dict[int, Dict[int, Dict[str, int]]],
    v6: Dict[int, Dict[int, Dict[str, int]]],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    day_max_v4: Dict[int, int] = {}
    day_max_v6: Dict[int, int] = {}

    for asn, days in v4.items():
        for day, pfx_map in days.items():
            if not pfx_map:
                continue
            mx = max(pfx_map.values())
            cur = day_max_v4.get(day, 0)
            if mx > cur:
                day_max_v4[day] = mx

    for asn, days in v6.items():
        for day, pfx_map in days.items():
            if not pfx_map:
                continue
            mx = max(pfx_map.values())
            cur = day_max_v6.get(day, 0)
            if mx > cur:
                day_max_v6[day] = mx
    return day_max_v4, day_max_v6


def compute_features_for_asn(
    asn: int,
    v4: Dict[int, Dict[int, Dict[str, int]]],
    v6: Dict[int, Dict[int, Dict[str, int]]],
    day_max_v4: Dict[int, int],
    day_max_v6: Dict[int, int],
    rir_data: Dict[int, Any],
    moas_data: Any,
) -> List[Any]:
   
    if asn not in v4 or asn not in v6:
        return []
    days4 = v4.get(asn, {})
    days6 = v6.get(asn, {})
    if not days4 or not days6:
        return []
    
    adv_times: List[int] = []
    unique_pfx_per_day: List[int] = []
    zero_drops = 0

    # We build adv_times by collecting days where ASN appears either v4 or v6
    all_days = sorted(set(days4.keys()) | set(days6.keys()))
    if not all_days:
        return []

    # Determine "drops" based on gaps between consecutive observed days.
    # If there is a gap > 1, count missing days as drops.
    prev = all_days[0]
    adv_times.append(prev)
    for d in all_days[1:]:
        if d == prev:
            continue
        if d - prev > 1:
            zero_drops += (d - prev - 1)
        prev = d
        adv_times.append(d)

    # Up-time: fraction of days observed in full span 1..max_day 
    max_day = max(all_days)
    up_time = safe_div(len(set(all_days)), max_day)
    second_half = [d for d in all_days if d >= 183 and d <= 365]
    first_year_days = [d for d in all_days if d >= 1 and d <= 365]
    sixmonths_percent = safe_div(len(second_half), max(1, len(first_year_days)))

    for d in all_days:
        pfx4 = set(days4.get(d, {}).keys())
        pfx6 = set(days6.get(d, {}).keys())
        unique_pfx_per_day.append(len(pfx4 | pfx6))

    p10_adTime = percentile_or_zero(adv_times, 10)
    q1_adTime = percentile_or_zero(adv_times, 25)
    median_adTime = median_or_zero(adv_times)
    q3_adTime = percentile_or_zero(adv_times, 75)
    p90_adTime = percentile_or_zero(adv_times, 90)

    median_vis: List[float] = []
    max_vis: List[float] = []

    for d in all_days:
        vals: List[float] = []

        # v4 and v6 normalization
        denom4 = float(day_max_v4.get(d, 0))
        if denom4 > 0:
            vals.extend([pc / denom4 for pc in days4.get(d, {}).values()])

        denom6 = float(day_max_v6.get(d, 0))
        if denom6 > 0:
            vals.extend([pc / denom6 for pc in days6.get(d, {}).values()])
        if not vals:
            continue

        median_vis.append(float(np.median(vals)))
        max_vis.append(float(np.max(vals)))

    p10_pfx_medianVis = percentile_or_zero(median_vis, 10)
    q1_pfx_medianVis = percentile_or_zero(median_vis, 10)
    median_pfx_medianVis = median_or_zero(median_vis)
    q3_pfx_medianVis = percentile_or_zero(median_vis, 75)
    p90_pfx_medianVis = percentile_or_zero(median_vis, 90)

    highMedVis_percent = safe_div(sum(1 for x in median_vis if x >= 0.5), len(median_vis))
    lowMedVis_percent = safe_div(sum(1 for x in median_vis if x < 0.5), len(median_vis))
    highMaxVis_percent = safe_div(sum(1 for x in max_vis if x >= 0.5), len(max_vis))
    lowMaxVis_percent = safe_div(sum(1 for x in max_vis if x < 0.5), len(max_vis))

    # Conditional ad-time percentiles (high/low max vis)
    high_max_vis_adv = [t for t, mv in zip(adv_times, max_vis[: len(adv_times)]) if mv >= 0.5]
    low_max_vis_adv = [t for t, mv in zip(adv_times, max_vis[: len(adv_times)]) if mv < 0.5]
    high_med_vis_adv = [t for t, md in zip(adv_times, median_vis[: len(adv_times)]) if md >= 0.5]
    low_med_vis_adv = [t for t, md in zip(adv_times, median_vis[: len(adv_times)]) if md < 0.5]

    q1_adTime_highMaxVis = percentile_or_zero(high_max_vis_adv, 25)
    median_adTime_highMaxVis = median_or_zero(high_max_vis_adv)
    q3_adTime_highMaxVis = percentile_or_zero(high_max_vis_adv, 75)

    q1_adTime_lowMaxVis = percentile_or_zero(low_max_vis_adv, 25)
    median_adTime_lowMaxVis = median_or_zero(low_max_vis_adv)
    q3_adTime_lowMaxVis = percentile_or_zero(low_max_vis_adv, 75)

    q1_adTime_highMedVis = percentile_or_zero(high_med_vis_adv, 25)
    median_adTime_highMedVis = median_or_zero(high_med_vis_adv)
    q3_adTime_highMedVis = percentile_or_zero(high_med_vis_adv, 75)

    q1_adTime_lowMedVis = percentile_or_zero(low_med_vis_adv, 25)
    median_adTime_lowMedVis = median_or_zero(low_med_vis_adv)
    q3_adTime_lowMedVis = percentile_or_zero(low_med_vis_adv, 75)
    median_unique_pfx = median_or_zero(unique_pfx_per_day)

    rir_count = 0
    frac_unassigned_add = 0.0
    frac_unassigned_pref = 0.0
    top_rir_add = 0
    top_rir_pref = 0
    gini_rir_add = 0.0
    gini_rir_pref = 0.0

    all_pref = set()
    for d in all_days:
        all_pref |= set(days4.get(d, {}).keys())
        all_pref |= set(days6.get(d, {}).keys())
    all_pref = set(all_pref)

    asn_rir_map = rir_data.get(asn, {})
    if all_pref:
        rir_per_prefix: List[int] = []
        for p in all_pref:
            try:
                rir_per_prefix.append(int(asn_rir_map[p][0]))
            except Exception:
                # Unknown prefix in rir mapping -> treat as unassigned (0)
                rir_per_prefix.append(0)

        c_pref = Counter(rir_per_prefix)
        rir_count = len(c_pref)
        addr_weights: Counter = Counter()
        total_add = 0.0
        unassigned_add = 0.0
        for p, rir_code in zip(all_pref, rir_per_prefix):
            try:
                plen = int(p.split("/")[1])
            except Exception:
                # If malformed, ignore (rare)
                continue
            if ":" in p:
                maxlen = 128
            else:
                maxlen = 32
            w = float(2 ** (maxlen - plen))
            total_add += w
            addr_weights[rir_code] += w
            if rir_code == 0:
                unassigned_add += w

        total_pref = float(len(all_pref))
        unassigned_pref = float(c_pref.get(0, 0))
        frac_unassigned_add = safe_div(unassigned_add, total_add)
        frac_unassigned_pref = safe_div(unassigned_pref, total_pref)

        # top RIR fractions (exclude unassigned? original prints top_rir_* as max count fraction overall)
        top_rir_add = 0
        top_rir_pref = 0
        if addr_weights:
            top_rir_add = int(max(addr_weights.values()))
        if c_pref:
            top_rir_pref = int(max(c_pref.values()))
        # Convert to fractions
        top_rir_add = safe_div(top_rir_add, total_add) if total_add > 0 else 0.0
        top_rir_pref = safe_div(top_rir_pref, total_pref) if total_pref > 0 else 0.0
        gini_rir_add = gini_from_counts(addr_weights.values())
        gini_rir_pref = gini_from_counts(c_pref.values())

    frac_moas_add = 0.0
    frac_moas_pref = 0.0
    std_moas_len = 0.0
    var_moas_len = 0.0
    range_moas_len = 0.0
    p1_moas_len = 0.0
    p99_moas_len = 0.0

    asn_moas = moas_data.get(asn, [])
    if all_pref:
        moas_prefixes: set = set()
        moas_sizes: List[int] = []

        for item in asn_moas:
            try:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    _day, origins = item
                    if origins is None:
                        continue
                    moas_sizes.append(len(origins))
                elif isinstance(item, (tuple, list)) and len(item) == 3:
                    prefix, _day, origins = item
                    if isinstance(prefix, str) and "/" in prefix:
                        moas_prefixes.add(prefix)
                    if origins is None:
                        continue
                    moas_sizes.append(len(origins))
            except Exception:
                continue

        try:
            if isinstance(asn_moas, (set, list)) and asn_moas and isinstance(next(iter(asn_moas)), str):
                moas_prefixes = set(asn_moas)
            elif isinstance(asn_moas, dict):
                moas_prefixes = set(asn_moas.keys())
            else:
                for it in asn_moas:
                    if isinstance(it, (tuple, list)) and it:
                        if isinstance(it[0], str) and "/" in it[0]:
                            moas_prefixes.add(it[0])
        except Exception:
            moas_prefixes = set()

        moas_prefixes = moas_prefixes & all_pref
        moas_pref = float(len(moas_prefixes))
        frac_moas_pref = safe_div(moas_pref, float(len(all_pref)))

        total_add = 0.0
        moas_add = 0.0
        for p in all_pref:
            try:
                plen = int(p.split("/")[1])
            except Exception:
                continue
            maxlen = 128 if ":" in p else 32
            w = float(2 ** (maxlen - plen))
            total_add += w
            if p in moas_prefixes:
                moas_add += w
        frac_moas_add = safe_div(moas_add, total_add)

        if moas_sizes:
            std_moas_len = std_or_zero(moas_sizes)
            var_moas_len = var_or_zero(moas_sizes)
            range_moas_len = float(max(moas_sizes) - min(moas_sizes))
            p1_moas_len = percentile_or_zero(moas_sizes, 1)
            p99_moas_len = percentile_or_zero(moas_sizes, 99)

    row = [asn,zero_drops,up_time,sixmonths_percent,p10_adTime,q1_adTime,median_adTime,q3_adTime,
           p90_adTime,p10_pfx_medianVis,q1_pfx_medianVis,median_pfx_medianVis,q3_pfx_medianVis,p90_pfx_medianVis,
           highMedVis_percent,lowMedVis_percent,highMaxVis_percent,lowMaxVis_percent,q1_adTime_highMaxVis,
           median_adTime_highMaxVis,q3_adTime_highMaxVis,q1_adTime_lowMaxVis,median_adTime_lowMaxVis,q3_adTime_lowMaxVis,
           q1_adTime_highMedVis,median_adTime_highMedVis,q3_adTime_highMedVis,q1_adTime_lowMedVis,median_adTime_lowMedVis,
           q3_adTime_lowMedVis,median_unique_pfx,rir_count,frac_unassigned_add,frac_unassigned_pref,top_rir_add,top_rir_pref,gini_rir_add,
           gini_rir_pref,frac_moas_add,frac_moas_pref,std_moas_len,var_moas_len,range_moas_len,p1_moas_len,p99_moas_len,
           ]
    return row

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asn-csv", default=ASN_LIST_CSV, help="CSV with column 'asn' (optional 'label' col).")
    ap.add_argument("--v4", default=V4_PKL_GZ, help="v4 pickle.gz path")
    ap.add_argument("--v6", default=V6_PKL_GZ, help="v6 pickle.gz path")
    ap.add_argument("--rir", default=ASN_RIR_PKL_GZ, help="asn_rir_all pickle.gz path")
    ap.add_argument("--moas", default=MOAS_CONSOLIDATED_PKL_GZ, help="moas_consolidated pickle.gz path")
    ap.add_argument("--out", default=OUTPUT_CSV, help="output CSV path")
    args = ap.parse_args()
    asn_df = pd.read_csv(args.asn_csv)
    if "asn" not in asn_df.columns:
        raise ValueError(f"ASN list file must contain column 'asn'. Found: {list(asn_df.columns)}")

    label_col: Optional[str] = None
    if "class" in asn_df.columns:
        label_col = "class"
    elif "predicted_class" in asn_df.columns:
        label_col = "predicted_class"
    asns = [int(x) for x in asn_df["asn"].tolist()]
    v4 = load_pickle_gz(args.v4)
    v6 = load_pickle_gz(args.v6)
    rir_data = load_pickle_gz(args.rir)
    moas_data = load_pickle_gz(args.moas)
    day_max_v4, day_max_v6 = compute_daywise_max_visibility(v4, v6)

    rows: List[List[Any]] = []
    for asn in asns:
        row = compute_features_for_asn(
            asn=asn,
            v4=v4,
            v6=v6,
            day_max_v4=day_max_v4,
            day_max_v6=day_max_v6,
            rir_data=rir_data,
            moas_data=moas_data,)
        if not row:
            continue
        rows.append(row)
    out_df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    if label_col is not None:
        out_df = out_df.merge(asn_df[["asn", label_col]], on="asn", how="left")

    out_df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()