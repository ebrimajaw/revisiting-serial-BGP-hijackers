
import gzip
import pickle
import numpy as np

ALL_RIR_TXT = "/home/all_rir.txt"
V4_PKL_GZ = "/home/processed4/asn_all_day_pc.pkl.gz"
V6_PKL_GZ = "/home/processed6/asn_all_day_pc.pkl.gz"
V6_REF_PKL_GZ = "/home/expanded_rir_v6.pkl.gz"
OUT_ASN_RIR = "/home/asn_rir_all.pkl.gz"
def load_pickle_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def build_v4_matrix():
    cats = ["unknown"]
    rirs = ["unknown"]
    Mv4 = np.zeros((256**4, 2), dtype=np.uint16)
    with open(ALL_RIR_TXT, "r", encoding="utf-8") as fp:
        for line in fp:
            l_data = line.strip().split(",")
            if len(l_data) < 7:
                continue
            rir = l_data[0]
            cat = l_data[6]
            if rir not in rirs:
                rirs.append(rir)
            if cat not in cats:
                cats.append(cat)
            if l_data[2] != "ipv4":
                continue
            try:
                value = int(l_data[4])
                ip_oct = [int(i) for i in l_data[3].split(".")]
                rowno = (
                    ip_oct[0]*(256**3)
                    + ip_oct[1]*(256**2)
                    + ip_oct[2]*256
                    + ip_oct[3])
            except Exception:
                continue
            Mv4[rowno:rowno+value, 0] = cats.index(cat)
            Mv4[rowno:rowno+value, 1] = rirs.index(rir)
    return Mv4, rirs

def load_v6_ref():
    rirv6 = load_pickle_gz(V6_REF_PKL_GZ)
    v6ref = {}
    for r in rirv6:
        for c in rirv6[r]:
            for ip in rirv6[r][c]:
                if ip not in v6ref:
                    v6ref[ip] = r
    return v6ref

def v4_rowno(ip):
    o = [int(i) for i in ip.split(".")]
    return o[0]*(256**3) + o[1]*(256**2) + o[2]*256 + o[3]

def collect_prefixes(data):
    for asn, days in data.items():
        prefs = set()
        for _, pfx_map in days.items():
            prefs.update(pfx_map.keys())
        yield int(asn), prefs

def main():
    Mv4, rirs = build_v4_matrix()
    v4data = load_pickle_gz(V4_PKL_GZ)
    v6data = load_pickle_gz(V6_PKL_GZ)
    v6ref = load_v6_ref()
    asn_rir_all = {}

    # IPv4
    for asn, prefs in collect_prefixes(v4data):
        mp = asn_rir_all.setdefault(asn, {})
        for pref in prefs:
            try:
                base_ip = pref.split("/")[0]
                row = v4_rowno(base_ip)
                rir_index = int(Mv4[row, 1])
            except Exception:
                rir_index = 0
            mp[pref] = (rir_index,)

    # IPv6
    for asn, prefs in collect_prefixes(v6data):
        mp = asn_rir_all.setdefault(asn, {})
        for pref in prefs:
            base_ip = pref.split("/")[0]
            rir_name = v6ref.get(base_ip)
            if rir_name and rir_name in rirs:
                rir_index = rirs.index(rir_name)
            else:
                rir_index = 0
            mp[pref] = (rir_index,)

    with gzip.open(OUT_ASN_RIR, "wb") as f:
        pickle.dump(asn_rir_all, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()