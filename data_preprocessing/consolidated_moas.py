import pandas as pd
import pickle, gzip, glob
from ast import literal_eval as lv

focus_asns = []
with open("ground_truth.csv", "r", encoding="utf-8") as fp:
    for line in fp:
        s = line.strip()
        if not s:
            continue
        focus_asns.append(str(lv(s)[0]))

focus_asns = set(focus_asns)


def consolidate_moas():
    files = glob.glob("/home/moas/*.csv")
    files.sort()
    moas_by_asn = {}

    for i, fpath in enumerate(files):
        day = i + 1
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            prefix = row["moas_prefix"]
            asns = row["moas_origins"].split(",")
            asns_set = set(asns)
            involved = list(asns_set & focus_asns)
            if not involved:
                continue
            for a in involved:
                if a not in moas_by_asn:
                    moas_by_asn[a] = []
                moas_by_asn[a].append((prefix, day, asns))

    with gzip.open("/home/moas_consolidated.pkl.gz", "wb") as fo:
        pickle.dump(moas_by_asn, fo, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    consolidate_moas()