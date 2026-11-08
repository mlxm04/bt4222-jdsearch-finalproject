import os, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

ROOT = "/content/drive/MyDrive/BT4222_Group10_JDSearch" if "google.colab" in sys.modules else os.path.dirname(os.path.abspath(__file__))
GAIN = np.array([0.0, 1.0, 3.0, 7.0])
SPLITS = {"train": 0, "valid": 1, "test": 2}

def ndcg_lists(sid, label, score, k=10, seed=0):
    """NDCG@k of each search, as in step4_lightgbm.ipynb. Ties at random, NaN last, all-zero searches dropped."""
    d = pd.DataFrame({"sid": sid, "label": label, "score": score})
    d["tie"] = np.random.default_rng(seed).random(len(d))
    d = d.sort_values(["sid", "score", "tie"], ascending=[True, False, True], na_position="last")
    pos = d.groupby("sid").cumcount().to_numpy()
    dcg = pd.Series(np.where(pos < k, GAIN[d.label.to_numpy()] / np.log2(pos + 2), 0.0)).groupby(d.sid.to_numpy()).sum()
    best = d.sort_values(["sid", "label"], ascending=[True, False])
    bpos = best.groupby("sid").cumcount().to_numpy()
    idcg = pd.Series(np.where(bpos < k, GAIN[best.label.to_numpy()] / np.log2(bpos + 2), 0.0)).groupby(best.sid.to_numpy()).sum()
    ok = idcg > 0
    return dcg[ok] / idcg[ok]

def paired_test(a, b):
    """a - b on the same searches: mean gap, 95% interval, paired t-test p-value, share of searches better / worse."""
    d = a - b
    half = 1.96 * d.std() / np.sqrt(len(d))
    return {"difference": d.mean(), "95% low": d.mean() - half, "95% high": d.mean() + half,
            "p-value": stats.ttest_rel(a, b).pvalue, "share better": (d > 0).mean(), "share worse": (d < 0).mean()}

ex = pd.read_parquet(f"{ROOT}/exposures.parquet")
SID, LAB = ex.sid.to_numpy(), ex.label.to_numpy()

FILES = {"lightgbm": "lightgbm/score_tuned.parquet",
         "kgin + features": "score_kgin_features.parquet",
         "lstm + attention": "score_lstm_attn_nolr.parquet"}
S = {}
for name, f in FILES.items():
    s = pd.read_parquet(f"{ROOT}/{f}")
    assert (s.sid.to_numpy() == SID).all() and (s.wid.to_numpy() == ex.wid.to_numpy()).all(), f"{f} rows don't line up with exposures"
    S[name] = s.score.to_numpy()
S["popularity"] = pq.read_table(f"{ROOT}/section3_popularity.parquet", columns=["pop_cart_90d"])["pop_cart_90d"].to_numpy()

per_search = {}
for split, code in SPLITS.items():
    m = (ex.split == code).to_numpy()
    per_search[split] = {n: ndcg_lists(SID[m], LAB[m], s[m]) for n, s in S.items()}
print(pd.DataFrame({sp: {n: v.mean() for n, v in d.items()} for sp, d in per_search.items()}).round(4).to_string(), "\n")

T = per_search["test"]
PAIRS = [("lstm + attention", "popularity"), ("lightgbm", "kgin + features"),
         ("lightgbm", "lstm + attention"), ("kgin + features", "lstm + attention")]
paired = pd.DataFrame({f"{a} - {b}": paired_test(T[a], T[b]) for a, b in PAIRS}).T
paired.to_csv(f"{ROOT}/lightgbm/paired_tests_cross_model.csv")
print("paired gaps on test:\n" + paired.round(4).to_string(), "\n")

m = (ex.split == SPLITS["valid"]).to_numpy()
pos = m & (LAB > 0)
is_cold = pq.read_table(f"{ROOT}/section3_popularity.parquet", columns=["is_cold"])["is_cold"].to_numpy() == 1
cold_share = pd.Series(is_cold[pos]).groupby(SID[pos]).mean()
cold = pd.Series(np.select([cold_share == 0, cold_share == 1], ["none cold", "all cold"], "some cold"), index=cold_share.index)
ctx = pd.read_parquet(f"{ROOT}/section4_context.parquet", columns=["sid", "clicks_7d"]).set_index("sid")
activity = pd.cut(ctx.clicks_7d, [-1, 0, 20, 100, np.inf], labels=["0", "1-20", "21-100", "over 100"]).astype(str)
qm = pd.read_parquet(f"{ROOT}/section2_affinity.parquet", columns=["sid", "qmatch_cate4_share_7d"])
repeat = qm.qmatch_cate4_share_7d.notna().groupby(qm.sid).any().map({True: "repeat", False: "new"})

GROUPS = {"interacted products": (cold, ["none cold", "some cold", "all cold"]),
          "clicks in the last 7 days": (activity, ["0", "1-20", "21-100", "over 100"]),
          "search": (repeat, ["new", "repeat"])}
rows = []
for title, (group, order) in GROUPS.items():
    g_row = group.reindex(SID).to_numpy()
    for g in order:
        s = m & (g_row == g)
        row = {"grouping": title, "group": g, "searches": pd.Series(SID[s]).nunique()}
        row.update({n: ndcg_lists(SID[s], LAB[s], v[s]).mean() for n, v in S.items()})
        rows.append(row)
segments = pd.DataFrame(rows)
segments.to_csv(f"{ROOT}/lightgbm/segments_cross_model.csv", index=False)
print("validation NDCG@10 by segment:\n" + segments.round(4).to_string(index=False))
