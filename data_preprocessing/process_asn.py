import gzip
import pickle
from ast import literal_eval as lv
from collections import defaultdict

"""
process_asns:
  - Reads focus ASNs from ground_truth or prediction set file 
  - Iterates day1..day2881 for:
      ipv4  & ipv6 files
  - Each line must be literal_eval(...) of length 3:
      [prefix, asn, peer_cnt]
  - Keeps only rows where asn ∈ focus_asns (ground truth or predctions sets)
  - Writes pickles consumed by compute.features for the feature extraction step:
"""

def load_focus_asns(gt_path: str) -> set:
    focus = set()
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            focus.add(int(lv(s)[0]))
    return focus

def parse_day_file(path: str):
    with gzip.open(path, "rb") as f:
        for raw in f:
            s = raw.decode("utf-8", errors="strict").strip()
            if not s:
                continue
            yield lv(s)  

def ingest(afi_dir: str, focus_asns: set, day_max: int = 2881):
    out = defaultdict(lambda: defaultdict(dict))
    for day in range(1, day_max + 1):
        path = f"{afi_dir}/day{day}.txt.gz"
        try:
            for ls in parse_day_file(path):
                if len(ls) != 3:
                    continue  
                prefix, asn, pc = ls[0], int(ls[1]), int(ls[2])
                if asn not in focus_asns:
                    continue
                prev = out[asn][day].get(prefix)
                if prev is not None and prev != pc:
                    raise ValueError(
                        f"Conflicting duplicate detected in {path}: "
                        f"(asn={asn}, day={day}, prefix={prefix}) prev_pc={prev}, new_pc={pc}")
                out[asn][day][prefix] = pc
        except FileNotFoundError:
            continue

    return out


def main():
    ground_truth = "/home/ground_truth.csv"
    ipv4_dir = "/home/ipv4"
    ipv6_dir = "/home/ipv6"

    out4 = "/home/ipv4/asn_all_day_pc.pkl.gz"
    out6 = "/home/ipv6/asn_all_day_pc.pkl.gz"  

    focus_asns = load_focus_asns(ground_truth)

    v4 = ingest(ipv4_dir, focus_asns, day_max=2881)
    v6 = ingest(ipv6_dir, focus_asns, day_max=2881)

    with gzip.open(out4, "wb") as f:
        pickle.dump(dict(v4), f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(out6, "wb") as f:
        pickle.dump(dict(v6), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Saved v4: {out4}")
    print(f"[OK] Saved v6: {out6}")

if __name__ == "__main__":
    main()