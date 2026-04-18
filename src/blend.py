import pandas as pd
from collections import defaultdict

best_sub = pd.read_csv('submission (4) (1).csv')
old_sub = pd.read_csv('submission888.csv')

best_dict = best_sub.groupby('user_id').apply(lambda x: list(zip(x['edition_id'], x['rank']))).to_dict()
old_dict = old_sub.groupby('user_id').apply(lambda x: list(zip(x['edition_id'], x['rank']))).to_dict()

users = sorted(list(set(best_dict.keys()) | set(old_dict.keys())))

rows_anchor = []

w_best = 0.85
w_old = 0.15
k = 60

for uid in users:
    b_items = best_dict.get(uid, [])
    o_items = old_dict.get(uid, [])

    scores_rrf = defaultdict(float)

    for eid, r in b_items:
        scores_rrf[eid] += w_best / (k + r)
    for eid, r in o_items:
        scores_rrf[eid] += w_old / (k + r)

    rrf_sorted = sorted(scores_rrf.items(), key=lambda x: -x[1])
    rrf_recs = [x[0] for x in rrf_sorted][:20]

    anchor_recs = []
    used = set()

    b_eids = [x[0] for x in b_items]

    for eid in b_eids[:7]:
        anchor_recs.append(eid)
        used.add(eid)

    for eid in rrf_recs:
        if len(anchor_recs) >= 20:
            break
        if eid not in used:
            anchor_recs.append(eid)
            used.add(eid)

    for rank, eid in enumerate(anchor_recs, 1):
        rows_anchor.append((uid, int(eid), rank))

df_anchor = pd.DataFrame(rows_anchor, columns=['user_id', 'edition_id', 'rank'])
df_anchor.to_csv('blend_2_anchor_top7.csv', index=False)