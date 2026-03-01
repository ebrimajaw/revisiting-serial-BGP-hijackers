import gzip
import pickle

"""
This script builds:
  - expanded_rir_v6.pkl.gz

Input expected:
  - expanded_rir_v6.txt.gz
Output:
  - expanded_rir_v6.pkl.gz
"""

INPUT_V6_TXT_GZ = "/home/expanded_rir_v6.txt.gz"
OUTPUT_V6_PKL_GZ = "/home/expanded_rir_v6.pkl.gz"

def build_v6_pkl(input_v6_txt_gz: str = INPUT_V6_TXT_GZ,
                 output_v6_pkl_gz: str = OUTPUT_V6_PKL_GZ) -> None:
    """
    Builds expanded_rir_v6.pkl.gz from expanded_rir_v6.txt.gz.
    expanded_rir_v6.txt.gz is build from the combined RIR allocation files
    """
    rir_dict = {}
    count = 0

    with gzip.open(input_v6_txt_gz, "rt", encoding="utf-8", errors="strict") as fp:
        for line in fp:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            if len(parts) < 3:
                continue
            ip, cat, rir = parts[0], parts[1], parts[2]
            if rir not in rir_dict:
                rir_dict[rir] = {}
            if cat not in rir_dict[rir]:
                rir_dict[rir][cat] = []
            rir_dict[rir][cat].append(ip)
            count += 1
    with gzip.open(output_v6_pkl_gz, "wb") as fo:
        pickle.dump(rir_dict, fo, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Built {output_v6_pkl_gz} from {input_v6_txt_gz} | rows={count} | rirs={len(rir_dict)}")

def main() -> None:
    build_v6_pkl()

if __name__ == "__main__":
    main()