import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from collections import defaultdict
from scipy import sparse
import warnings
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = 'data/'

users = pd.read_csv(DATA_DIR + 'users.csv')
editions_raw = pd.read_csv(DATA_DIR + 'editions.csv')
authors = pd.read_csv(DATA_DIR + 'authors.csv')
genres = pd.read_csv(DATA_DIR + 'genres.csv')
book_genres = pd.read_csv(DATA_DIR + 'book_genres.csv')
interactions = pd.read_csv(DATA_DIR + 'interactions.csv', parse_dates=['event_ts'])
targets = pd.read_csv(DATA_DIR + 'targets.csv')
candidates = pd.read_csv(DATA_DIR + 'candidates.csv')

ed2author = editions_raw.set_index('edition_id')['author_id'].to_dict()
ed2lang = editions_raw.set_index('edition_id')['language_id'].to_dict()
ed2year = editions_raw.set_index('edition_id')['publication_year'].to_dict()
ed2book = editions_raw.set_index('edition_id')['book_id'].to_dict()
ed2pub = editions_raw.set_index('edition_id')['publisher_id'].to_dict()
book2genres = book_genres.groupby('book_id')['genre_id'].apply(list).to_dict()
ed2genres = {eid: book2genres.get(bid, []) for eid, bid in ed2book.items()}

interactions = interactions.sort_values('event_ts')
cutoff_ts = interactions['event_ts'].max() - pd.Timedelta(days=30)
train_inter = interactions[interactions['event_ts'] < cutoff_ts].copy()
val_inter = interactions[interactions['event_ts'] >= cutoff_ts].copy()

all_u = users['user_id'].unique()
all_i = editions_raw['edition_id'].unique()
u2i = {uid: i for i, uid in enumerate(all_u)}
i2i = {iid: i for i, iid in enumerate(all_i)}
nu, ni = len(all_u), len(all_i)
all_a = authors['author_id'].unique()
a2i = {aid: i for i, aid in enumerate(all_a)}
all_g = genres['genre_id'].unique()
g2i = {gid: i for i, gid in enumerate(all_g)}
all_genre_ids = sorted(genres['genre_id'].unique())
genre2pos = {g: i for i, g in enumerate(all_genre_ids)}
book_genre_idx = {}
for bid, gl in book2genres.items():
    idx = [g2i[g] for g in gl if g in g2i]
    if idx: book_genre_idx[bid] = idx
cand_sets = candidates.groupby('user_id')['edition_id'].apply(set).to_dict()

def build_matrix(inter_df, w_read=3.0, w_wish=1.0, use_rating=True):
    df = inter_df.copy()
    df['w'] = df['event_type'].map({1: w_wish, 2: w_read}).fillna(1.0)
    if use_rating:
        hr = df['rating'].notna()
        df.loc[hr, 'w'] *= (0.5 + df.loc[hr, 'rating'] / 10.0)
    rows = df['user_id'].map(u2i).fillna(-1).astype(int).values
    cols = df['edition_id'].map(i2i).fillna(-1).astype(int).values
    w = df['w'].values
    m = (rows >= 0) & (cols >= 0)
    mat = sparse.csr_matrix((w[m], (rows[m], cols[m])), shape=(nu, ni))
    mat.sum_duplicates()
    return mat

def als_cg(R, factors=128, iterations=10, reg=0.01, alpha=2.0):
    Rc = R.tocsr()
    Rt = R.T.tocsr()
    n_u, n_i = R.shape
    X = np.random.normal(0, 0.01, (n_u, factors)).astype(np.float32)
    Y = np.random.normal(0, 0.01, (n_i, factors)).astype(np.float32)
    best_X, best_Y, best_loss = None, None, float('inf')
    for it in range(iterations):
        YtY = Y.T @ Y
        for u in range(n_u):
            s, e = Rc.indptr[u], Rc.indptr[u+1]
            if s == e: continue
            idx = Rc.indices[s:e]; vals = Rc.data[s:e]
            Yu = Y[idx]; cu = alpha * vals
            A = YtY + (Yu.T * cu) @ Yu
            A[np.diag_indices_from(A)] += reg
            b = Yu.T @ (cu + 1.0)
            try: X[u] = np.linalg.solve(A, b)
            except: X[u] = np.linalg.lstsq(A, b, rcond=None)[0]
        XtX = X.T @ X
        for i in range(n_i):
            s, e = Rt.indptr[i], Rt.indptr[i+1]
            if s == e: continue
            idx = Rt.indices[s:e]; vals = Rt.data[s:e]
            Xi = X[idx]; ci = alpha * vals
            A = XtX + (Xi.T * ci) @ Xi
            A[np.diag_indices_from(A)] += reg
            b = Xi.T @ (ci + 1.0)
            try: Y[i] = np.linalg.solve(A, b)
            except: Y[i] = np.linalg.lstsq(A, b, rcond=None)[0]
        su = np.random.choice(n_u, min(500, n_u), replace=False)
        err, cnt = 0.0, 0
        for uu in su:
            s2, e2 = Rc.indptr[uu], Rc.indptr[uu+1]
            if s2 == e2: continue
            preds = Y[Rc.indices[s2:e2]] @ X[uu]
            c = 1.0 + alpha * Rc.data[s2:e2]
            err += np.sum(c*(1.0-preds)**2); cnt += e2-s2
        loss = err/max(cnt,1)
        if loss < best_loss:
            best_loss = loss
            best_X = X.copy(); best_Y = Y.copy()
    return best_X.astype(np.float64), best_Y.astype(np.float64)

def train_bpr_batch(R, factors=64, epochs=100, lr=0.05, reg=0.0001, batch_size=8192):
    Rc = R.tocsr()
    n_u, n_i = R.shape
    X = np.random.normal(0, 0.01, (n_u, factors)).astype(np.float32)
    Y = np.random.normal(0, 0.01, (n_i, factors)).astype(np.float32)
    pos_u, pos_i = [], []
    for u in range(n_u):
        s, e = Rc.indptr[u], Rc.indptr[u+1]
        if s < e:
            items = Rc.indices[s:e]
            pos_u.extend([u]*len(items)); pos_i.extend(items)
    pos_u = np.array(pos_u, dtype=np.int32)
    pos_i = np.array(pos_i, dtype=np.int32)
    n_pairs = len(pos_u)
    user_pos = {}
    for u in range(n_u):
        s, e = Rc.indptr[u], Rc.indptr[u+1]
        if s < e: user_pos[u] = set(Rc.indices[s:e])
    for ep in range(epochs):
        perm = np.random.permutation(n_pairs)
        total_loss = 0.0
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)
            bi = perm[start:end]; bs = len(bi)
            bu = pos_u[bi]; bpos = pos_i[bi]
            bneg = np.random.randint(0, n_i, size=bs)
            for _ in range(2):
                mask = np.array([bneg[k] in user_pos.get(bu[k], set()) for k in range(bs)])
                if not mask.any(): break
                bneg[mask] = np.random.randint(0, n_i, size=mask.sum())
            xu = X[bu]; yi = Y[bpos]; yj = Y[bneg]
            diff = np.clip(np.sum(xu * (yi - yj), axis=1), -50, 50)
            sig = 1.0 / (1.0 + np.exp(diff))
            total_loss += np.sum(-np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-10))
            sig2 = sig[:, None]
            X[bu] += lr * (sig2 * (yi - yj) - reg * xu)
            Y[bpos] += lr * (sig2 * xu - reg * yi)
            Y[bneg] += lr * (-sig2 * xu - reg * yj)
        if (ep+1) % 25 == 0 or ep == 0:
            su = np.random.choice(n_pairs, min(10000, n_pairs), replace=False)
            sp = np.sum(X[pos_u[su]] * Y[pos_i[su]], axis=1)
            sn = np.sum(X[pos_u[su]] * Y[np.random.randint(0,n_i,len(su))], axis=1)
    return X.astype(np.float64), Y.astype(np.float64)

def train_ease(R, reg=500.0, max_items=15000):
    item_counts = np.array(R.sum(axis=0)).flatten()
    if np.count_nonzero(item_counts) > max_items:
        active_items = np.sort(np.argsort(-item_counts)[:max_items])
    else:
        active_items = np.where(item_counts > 0)[0]
    n_active = len(active_items)
    item_remap = {int(old): new for new, old in enumerate(active_items)}
    R_sub = R[:, active_items].tocsr()
    G = (R_sub.T @ R_sub).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += reg
    P = np.linalg.inv(G)
    B = P / (-np.diag(P)[None, :]); np.fill_diagonal(B, 0)
    return R_sub, B, active_items, item_remap

def build_item2vec(inter_df, window=5, dim=64, label=''):
    seqs = inter_df.sort_values(['user_id','event_ts']).groupby('user_id')['edition_id'].apply(list)
    freq = set(inter_df['edition_id'].value_counts()[lambda x: x >= 2].index)
    cooc = defaultdict(float)
    for seq in seqs:
        f = [x for x in seq if x in freq]
        for i, a in enumerate(f):
            for j in range(max(0, i-window), min(len(f), i+window+1)):
                if i != j: cooc[(a, f[j])] += 1.0 / abs(i - j)
    freq_list = sorted(freq); fmap = {iid: i for i, iid in enumerate(freq_list)}; nf = len(freq_list)
    rs, cs, vs = [], [], []
    for (a, b), v in cooc.items():
        if a in fmap and b in fmap: rs.append(fmap[a]); cs.append(fmap[b]); vs.append(v)
    C = sparse.csr_matrix((vs, (rs, cs)), shape=(nf, nf))
    row_s = np.array(C.sum(axis=1)).flatten(); col_s = np.array(C.sum(axis=0)).flatten(); total = C.sum()
    Cc = C.tocoo()
    pmi = np.maximum(np.log(Cc.data * total / (row_s[Cc.row] * col_s[Cc.col] + 1e-10) + 1e-10), 0)
    PPMI = sparse.csr_matrix((pmi, (Cc.row, Cc.col)), shape=(nf, nf))
    vecs = TruncatedSVD(n_components=dim, n_iter=10, random_state=42).fit_transform(PPMI)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-8)
    i2v = {iid: vecs[idx] for iid, idx in fmap.items()}
    return i2v

def build_all_embeddings(inter_df, label=''):
    E = {}; mat = build_matrix(inter_df)

    ml = mat.copy(); ml.data = np.log1p(ml.data * 2.0)
    s1 = TruncatedSVD(n_components=64, n_iter=12, random_state=42)
    E['u_svd'] = s1.fit_transform(ml); E['i_svd'] = s1.components_.T

    mb = mat.copy(); mb.data = np.ones_like(mb.data)
    s2 = TruncatedSVD(n_components=32, n_iter=10, random_state=123)
    E['u_svd2'] = s2.fit_transform(mb); E['i_svd2'] = s2.components_.T

    E['u_als'], E['i_als'] = als_cg(mat, factors=128, iterations=10, reg=0.01, alpha=2.0)

    E['u_als2'], E['i_als2'] = als_cg(mat, factors=64, iterations=10, reg=0.05, alpha=5.0)

    E['u_bpr'], E['i_bpr'] = train_bpr_batch(mat, factors=64, epochs=100, lr=0.05, reg=0.0001, batch_size=8192)

    E['ease_R'], E['ease_B'], E['ease_items'], E['ease_remap'] = train_ease(mat, reg=500.0, max_items=15000)

    E['i2v'] = build_item2vec(inter_df, window=5, dim=64, label=label)

    dfa = inter_df.copy(); dfa['aid'] = dfa['edition_id'].map(ed2author)
    dfa = dfa[dfa['aid'].notna()]; dfa['w'] = dfa['event_type'].map({1:1.0, 2:3.0})
    ra = dfa['user_id'].map(u2i).fillna(-1).astype(int).values
    ca = dfa['aid'].map(a2i).fillna(-1).astype(int).values
    ma = (ra>=0)&(ca>=0)
    mata = sparse.csr_matrix((dfa['w'].values[ma], (ra[ma],ca[ma])), shape=(nu,len(all_a)))
    mata.sum_duplicates(); mata.data = np.log1p(mata.data)
    sa = TruncatedSVD(n_components=32, n_iter=10, random_state=42)
    E['u_auth'] = sa.fit_transform(mata); E['a_auth'] = sa.components_.T

    rg,cg,vg = [],[],[]
    mg = inter_df.merge(editions_raw[['edition_id','book_id']], on='edition_id')
    for row in mg.itertuples():
        ui = u2i.get(row.user_id)
        if ui is None: continue
        w = 3.0 if row.event_type==2 else 1.0
        for gi in book_genre_idx.get(row.book_id, []):
            rg.append(ui); cg.append(gi); vg.append(w)
    matg = sparse.csr_matrix((vg,(rg,cg)), shape=(nu,len(all_g)))
    matg.data = np.log1p(matg.data)
    sg = TruncatedSVD(n_components=24, n_iter=10, random_state=42)
    E['u_genre'] = sg.fit_transform(matg); E['g_genre'] = sg.components_.T

    dfp = inter_df.copy(); dfp['pub'] = dfp['edition_id'].map(ed2pub)
    dfp = dfp[dfp['pub'].notna()]
    all_pubs = sorted(dfp['pub'].unique())
    p2i_local = {p:i for i,p in enumerate(all_pubs)}
    dfp['w'] = dfp['event_type'].map({1:1.0, 2:3.0})
    rp = dfp['user_id'].map(u2i).fillna(-1).astype(int).values
    cp = dfp['pub'].map(p2i_local).fillna(-1).astype(int).values
    mp = (rp>=0)&(cp>=0)
    matp = sparse.csr_matrix((dfp['w'].values[mp], (rp[mp],cp[mp])), shape=(nu,len(all_pubs)))
    matp.sum_duplicates(); matp.data = np.log1p(matp.data)
    sp = TruncatedSVD(n_components=16, n_iter=10, random_state=42)
    E['u_pub'] = sp.fit_transform(matp); E['p_pub'] = sp.components_.T
    E['p2i'] = p2i_local
    return E

print('Building embeddings...')
emb = build_all_embeddings(train_inter, label='train')

from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
texts = (editions_raw['title'].fillna('') + ". " +
         editions_raw['description'].fillna('').astype(str).str.slice(0, 200)).tolist()
content_vecs = bert_model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
norms = np.linalg.norm(content_vecs, axis=1, keepdims=True)
content_vecs = content_vecs / (norms + 1e-9)
from sklearn.decomposition import PCA
content_vecs = PCA(n_components=64, random_state=42).fit_transform(content_vecs)
norms = np.linalg.norm(content_vecs, axis=1, keepdims=True)
content_vecs = content_vecs / (norms + 1e-9)
eid2cidx = {eid: i for i, eid in enumerate(editions_raw['edition_id'])}
del bert_model, texts; gc.collect()

def build_caches(inter_df, emb_dict, label=''):
    C = {}; df = inter_df.copy()
    max_ts = df['event_ts'].max()
    df['days_ago'] = (max_ts - df['event_ts']).dt.total_seconds() / 86400

    C['ucontent'] = {}
    for uid, grp in tqdm(df.groupby('user_id'), desc=f'Profiles {label}', leave=False):
        try:
            valid = np.array([eid2cidx.get(e) is not None for e in grp['edition_id']])
            if not valid.any(): continue
            indices = np.array([eid2cidx[e] for e, v in zip(grp['edition_id'], valid) if v])
            ws = grp['event_type'].map({2:3.0, 1:1.0}).values[valid]
            p = np.average(content_vecs[indices], axis=0, weights=ws)
            n = np.linalg.norm(p)
            if n > 0: p /= n
            C['ucontent'][uid] = p
        except: continue

    C['u_i2v'] = {}
    i2v = emb_dict.get('i2v', {})
    for uid, grp in df.groupby('user_id'):
        vecs, ws = [], []
        for _, r in grp.sort_values('event_ts', ascending=False).head(30).iterrows():
            v = i2v.get(r['edition_id'])
            if v is not None:
                w = 3.0 if r['event_type'] == 2 else 1.0
                vecs.append(v); ws.append(w)
        if vecs:
            p = np.average(vecs, axis=0, weights=np.array(ws))
            n = np.linalg.norm(p)
            if n > 0: p /= n
            C['u_i2v'][uid] = p

    un = emb_dict['u_svd'] / (np.linalg.norm(emb_dict['u_svd'], axis=1, keepdims=True)+1e-8)
    sim = un @ un.T; np.fill_diagonal(sim, 0)
    C['neighbors'] = {}
    for idx in range(nu):
        top = np.argpartition(sim[idx], -80)[-80:]
        C['neighbors'][all_u[idx]] = [(all_u[ni], sim[idx][ni]) for ni in top]
    del sim; gc.collect()

    km = MiniBatchKMeans(n_clusters=50, random_state=42, batch_size=256)
    cl = km.fit_predict(emb_dict['u_svd'])
    C['ucluster'] = {uid:c for uid,c in zip(all_u, cl)}
    cdf = df.copy(); cdf['cl'] = cdf['user_id'].map(C['ucluster'])
    cdf['w'] = cdf['event_type'].map({1:1.0, 2:3.0})
    C['cl_bw'] = cdf.groupby(['cl','edition_id'])['w'].sum().to_dict()
    C['cl_tot'] = cdf.groupby('cl')['w'].sum().to_dict()

    C['utop'] = {}
    for uid, grp in df.groupby('user_id'):
        items = grp.sort_values('event_ts', ascending=False).head(50)
        wl = []
        for _, r in items.iterrows():
            w = 3.0 if r['event_type']==2 else 1.0
            if pd.notna(r['rating']): w *= (0.5 + r['rating']/10.0)
            wl.append((r['edition_id'], w))
        C['utop'][uid] = wl

    C['uhist'] = df.sort_values('event_ts').groupby('user_id')['edition_id'].apply(list).to_dict()
    C['co'] = defaultdict(float)
    for uid, books in tqdm(C['uhist'].items(), desc=f'Covisit {label}', leave=False):
        seen=set(); uniq=[]
        for b in books:
            if b not in seen: uniq.append(b); seen.add(b)
        w = uniq[-30:]; nn = len(w)
        if nn > 1:
            for ai in range(nn):
                for bi in range(nn):
                    if ai != bi: C['co'][(w[ai],w[bi])] += 1.0/(1.0+abs(ai-bi)*0.1)

    seqs = df.sort_values(['user_id','event_ts']).groupby('user_id')['edition_id'].apply(list)
    C['seqs'] = seqs.to_dict()
    C['trans'] = defaultdict(int); C['tout'] = defaultdict(int)
    for seq in seqs:
        for i in range(len(seq)-1):
            a,b = seq[i], seq[i+1]
            if a!=b: C['trans'][(a,b)]+=1; C['tout'][a]+=1

    C['hsets'] = df.groupby('user_id')['edition_id'].apply(set).to_dict()
    C['hw'] = df.groupby(['user_id','edition_id'])['event_type'].max().to_dict()
    C['gpop'] = df['edition_id'].value_counts().to_dict()
    C['rpop'] = df[df['event_type']==2]['edition_id'].value_counts().to_dict()
    for wn,d in [('7d',7),('14d',14),('30d',30)]:
        ct = max_ts - pd.Timedelta(days=d)
        C[f'rp_{wn}'] = df[df['event_ts']>=ct]['edition_id'].value_counts().to_dict()
    df['rw'] = np.exp(-0.03*df['days_ago']) * df['event_type'].map({1:1.0,2:3.0})
    C['recpop'] = df.groupby('edition_id')['rw'].sum().to_dict()
    df['py'] = df['edition_id'].map(ed2year)
    C['uavgy'] = df[df['py']>1900].groupby('user_id')['py'].mean().to_dict()
    C['gavgy'] = df[df['py']>1900]['py'].mean()
    tmp = df.copy(); tmp['aid'] = tmp['edition_id'].map(ed2author)
    C['uacnt'] = tmp.groupby(['user_id','aid']).size().to_dict()
    C['uarat'] = tmp[tmp['rating'].notna()].groupby(['user_id','aid'])['rating'].mean().to_dict()
    C['agrat'] = tmp[tmp['rating'].notna()].groupby('aid')['rating'].mean().to_dict()
    C['abcnt'] = editions_raw.groupby('author_id').size().to_dict()
    C['uarat_global'] = df[df['rating'].notna()].groupby('user_id')['rating'].mean().to_dict()
    C['erat'] = df[df['rating'].notna()].groupby('edition_id')['rating'].mean().to_dict()
    C['grat'] = df[df['rating'].notna()]['rating'].mean()
    C['uasets'] = tmp[tmp['aid'].notna()].groupby('user_id')['aid'].apply(set).to_dict()
    C['ulang'] = df.merge(editions_raw[['edition_id','language_id']], on='edition_id').groupby('user_id')['language_id'].agg(lambda x: x.mode()[0] if len(x)>0 else -1).to_dict()
    C['ugvec'] = {}; C['ugent'] = {}
    ugc = defaultdict(lambda: defaultdict(float))
    mg = df.merge(editions_raw[['edition_id','book_id']], on='edition_id')
    for row in mg.itertuples():
        for g in book2genres.get(row.book_id, []):
            ugc[row.user_id][g] += (3.0 if row.event_type==2 else 1.0)
    for uid, gc_d in ugc.items():
        tot = sum(gc_d.values())
        p = np.array(list(gc_d.values()))/tot if tot>0 else np.array([0])
        C['ugent'][uid] = -np.sum(p*np.log(p+1e-10))
        v = np.zeros(len(all_genre_ids))
        for g,c in gc_d.items():
            if g in genre2pos: v[genre2pos[g]] = c
        n = np.linalg.norm(v)
        if n>0: v/=n
        C['ugvec'][uid] = v
    C['ubsets'] = mg.groupby('user_id')['book_id'].apply(set).to_dict()
    C['bpop'] = mg.groupby('book_id').size().to_dict()
    C['breads'] = mg[mg['event_type']==2].groupby('book_id').size().to_dict()
    si = sorted(C['gpop'].items(), key=lambda x:x[1], reverse=True)
    nr = len(si)
    C['granks'] = {e:(i+1)/nr for i,(e,_) in enumerate(si)}
    C['ua7'] = df[df['days_ago']<=7].groupby('user_id').size().to_dict()
    C['ua14'] = df[df['days_ago']<=14].groupby('user_id').size().to_dict()
    return C

print('Building caches...')
cache = build_caches(train_inter, emb, label='train')

def score_pairs(df, E, C, label=''):
    df = df.copy(); uids = df['user_id'].values; eids = df['edition_id'].values; n = len(df)

    svd1=np.zeros(n); svd2=np.zeros(n); als1=np.zeros(n); als2=np.zeros(n); bpr1=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        ui,ei = u2i.get(u), i2i.get(e)
        if ui is not None and ei is not None:
            svd1[i]=E['u_svd'][ui]@E['i_svd'][ei]; svd2[i]=E['u_svd2'][ui]@E['i_svd2'][ei]
            als1[i]=E['u_als'][ui]@E['i_als'][ei]; als2[i]=E['u_als2'][ui]@E['i_als2'][ei]
            bpr1[i]=E['u_bpr'][ui]@E['i_bpr'][ei]
    df['svd_item']=svd1; df['svd_item2']=svd2; df['als_score']=als1; df['als_score2']=als2; df['bpr_score']=bpr1

    ease_R=E['ease_R']; ease_B=E['ease_B']; ease_remap=E['ease_remap']
    ease_sc = np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        ui=u2i.get(u); ei_g=i2i.get(e)
        if ui is None or ei_g is None: continue
        ei_e = ease_remap.get(ei_g)
        if ei_e is None: continue
        s,end=ease_R.indptr[ui],ease_R.indptr[ui+1]
        if s<end: ease_sc[i]=ease_R.data[s:end]@ease_B[ease_R.indices[s:end],ei_e]
    df['ease_score']=ease_sc

    i2v=E.get('i2v',{}); i2v_sc=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        up=C.get('u_i2v',{}).get(u); ev=i2v.get(e)
        if up is not None and ev is not None: i2v_sc[i]=np.dot(up,ev)
    df['i2v_score']=i2v_sc

    pub_sc=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        ui=u2i.get(u); pub=ed2pub.get(e)
        if ui is not None and pub is not None:
            pi=E['p2i'].get(pub)
            if pi is not None: pub_sc[i]=E['u_pub'][ui]@E['p_pub'][pi]
    df['svd_pub']=pub_sc

    cs=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        up=C['ucontent'].get(u)
        if up is None: continue
        ci=eid2cidx.get(e)
        if ci is None: continue
        ev=content_vecs[ci]; nn=np.linalg.norm(ev)
        if nn>0: cs[i]=np.dot(up,ev/nn)
    df['content_sim']=cs

    ics_a=np.zeros(n); ics_m=np.zeros(n); ica_a=np.zeros(n); ica_m=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        ei=i2i.get(e)
        if ei is None: continue
        hist=C['utop'].get(u,[])
        if not hist: continue
        evs=E['i_svd'][ei]; ens=np.linalg.norm(evs); eva=E['i_als'][ei]; ena=np.linalg.norm(eva)
        tws,wss,bss=0,0,-1; twa,wsa,bsa=0,0,-1
        for he,w in hist[:30]:
            hi=i2i.get(he)
            if hi is None: continue
            if ens>1e-8:
                hv=E['i_svd'][hi]; hn=np.linalg.norm(hv)
                if hn>1e-8: s=np.dot(evs,hv)/(ens*hn); wss+=s*w; tws+=w; bss=max(bss,s)
            if ena>1e-8:
                hv=E['i_als'][hi]; hn=np.linalg.norm(hv)
                if hn>1e-8: s=np.dot(eva,hv)/(ena*hn); wsa+=s*w; twa+=w; bsa=max(bsa,s)
        ics_a[i]=wss/tws if tws>0 else 0; ics_m[i]=max(bss,0)
        ica_a[i]=wsa/twa if twa>0 else 0; ica_m[i]=max(bsa,0)
    df['icf_svd_avg']=ics_a; df['icf_svd_max']=ics_m; df['icf_als_avg']=ica_a; df['icf_als_max']=ica_m

    sa=np.zeros(n); sg=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        ui=u2i.get(u)
        if ui is None: continue
        try:
            ai=a2i.get(ed2author.get(e))
            if ai is not None: sa[i]=E['u_auth'][ui]@E['a_auth'][ai]
        except: pass
        try:
            bid=ed2book.get(e); gi=book_genre_idx.get(bid,[])
            if gi: sg[i]=E['u_genre'][ui]@np.mean([E['g_genre'][j] for j in gi],axis=0)
        except: pass
    df['svd_author']=sa; df['svd_genre']=sg

    knn=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        nbs=C['neighbors'].get(u,[])
        if not nbs: continue
        sc,st=0.0,0.0
        for nid,sim in nbs:
            if nid in C['hsets'] and e in C['hsets'][nid]:
                w=3.0 if C['hw'].get((nid,e),1)==2 else 1.0; sc+=sim*w
            st+=abs(sim)
        knn[i]=sc/st if st>0 else 0
    df['knn_score']=knn

    cl=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        c=C['ucluster'].get(u)
        if c is not None: cl[i]=C['cl_bw'].get((c,e),0)/C['cl_tot'].get(c,1)
    df['cluster_aff']=cl

    cov=np.zeros(n); mkv=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        h=C['uhist'].get(u,[])
        if h:
            seen=set()
            for p in reversed(h[-20:]):
                if p==e or p in seen: continue
                seen.add(p); cov[i]+=C['co'].get((p,e),0)
        h2=C['seqs'].get(u,[])
        if h2:
            decay=1.0
            for p in reversed(h2[-5:]):
                tc=C['trans'].get((p,e),0)
                if tc>0: mkv[i]+=(tc/C['tout'].get(p,1))*decay
                decay*=0.5
    df['covisit']=cov; df['markov']=mkv

    ac=np.zeros(n); aur=np.zeros(n); agr=np.zeros(n); am=np.zeros(n); lm=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        aid=ed2author.get(e)
        if aid is not None:
            ac[i]=C['uacnt'].get((u,aid),0)/C['abcnt'].get(aid,1)
            r=C['uarat'].get((u,aid))
            if r is not None: aur[i]=r
            agr[i]=C['agrat'].get(aid,0)
            if aid in C['uasets'].get(u,set()): am[i]=1
        if ed2lang.get(e)==C['ulang'].get(u,-1): lm[i]=1
    df['auth_compl']=ac; df['auth_u_rat']=aur; df['auth_g_rat']=agr; df['author_match']=am; df['lang_match']=lm

    gc_arr=np.zeros(n); ng=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        uv=C['ugvec'].get(u); gs=ed2genres.get(e,[])
        ng[i]=len(gs)
        if uv is not None and gs:
            ev=np.zeros(len(all_genre_ids))
            for g in gs:
                if g in genre2pos: ev[genre2pos[g]]=1.0
            en=np.linalg.norm(ev)
            if en>0: gc_arr[i]=np.dot(uv,ev/en)
    df['genre_cosine']=gc_arr; df['n_genres']=ng

    sb=np.zeros(n); yd=np.zeros(n); rd=np.zeros(n)
    for i,(u,e) in enumerate(zip(uids,eids)):
        bid=ed2book.get(e)
        if bid is not None and bid in C['ubsets'].get(u,set()): sb[i]=1
        uy=C['uavgy'].get(u,C['gavgy']); by=ed2year.get(e,C['gavgy'])
        if pd.isna(by) or by==0: by=C['gavgy']
        yd[i]=abs(uy-by)
        rd[i]=C['erat'].get(e,C['grat'])-C['uarat_global'].get(u,C['grat'])
    df['same_book']=sb; df['year_diff']=yd; df['rating_dev']=rd

    df['global_pop']=[C['gpop'].get(e,0) for e in eids]
    df['read_pop']=[C['rpop'].get(e,0) for e in eids]
    df['recency_pop']=[C['recpop'].get(e,0) for e in eids]
    df['pop_7d']=[C['rp_7d'].get(e,0) for e in eids]
    df['pop_14d']=[C['rp_14d'].get(e,0) for e in eids]
    df['trend']=df['pop_14d']/(df['global_pop']+1)
    df['read_ratio']=df['read_pop']/(df['global_pop']+1)
    df['global_rank']=[C['granks'].get(e,1.0) for e in eids]
    bp=np.zeros(n); br=np.zeros(n)
    for i,e in enumerate(eids):
        bid=ed2book.get(e)
        if bid: bp[i]=C['bpop'].get(bid,0); br[i]=C['breads'].get(bid,0)
    df['book_pop']=bp; df['book_reads']=br; df['book_read_r']=br/(bp+1)

    df['svd_x_pop']=df['svd_item']*np.log1p(df['global_pop'])
    df['svd_x_knn']=df['svd_item']*df['knn_score']
    df['als_x_pop']=df['als_score']*np.log1p(df['global_pop'])
    df['als_x_knn']=df['als_score']*df['knn_score']
    df['svd_x_content']=df['svd_item']*df['content_sim']
    df['icf_x_svd']=df['icf_svd_avg']*df['svd_item']
    df['svd_x_genre']=df['svd_item']*df['genre_cosine']
    df['covisit_x_svd']=df['covisit']*df['svd_item']
    df['bpr_x_pop']=df['bpr_score']*np.log1p(df['global_pop'])
    df['ease_x_pop']=df['ease_score']*np.log1p(df['global_pop'])
    df['ease_x_svd']=df['ease_score']*df['svd_item']
    df['i2v_x_svd']=df['i2v_score']*df['svd_item']
    df['bpr_x_svd']=df['bpr_score']*df['svd_item']
    df['pub_x_svd']=df['svd_pub']*df['svd_item']
    return df

def add_cand_rank_features(df):
    rank_cols = ['svd_item','als_score','bpr_score','knn_score','covisit','global_pop','content_sim','icf_svd_avg','genre_cosine','svd_author','ease_score','i2v_score']
    rank_cols = [c for c in rank_cols if c in df.columns]
    for col in rank_cols:
        df[f'{col}_prank'] = df.groupby('user_id')[col].rank(pct=True, method='min')
    return df

def build_user_feats(inter_df, C):
    uf = inter_df.groupby('user_id').agg(
        u_total=('event_type','count'),
        u_reads=('event_type',lambda x:(x==2).sum()),
        u_read_ratio=('event_type',lambda x:(x==2).mean()),
        u_avg_rating=('rating','mean'),
        u_std_rating=('rating','std'),
        u_unique=('edition_id','nunique'),
        u_last_ts=('event_ts','max'),
        u_first_ts=('event_ts','min'),
    ).reset_index()
    uf = uf.merge(users[['user_id','age','gender']], on='user_id', how='left')
    uf['u_age']=uf['age'].fillna(uf['age'].median()); uf['u_gender']=uf['gender'].fillna(0)
    uf['u_entropy']=uf['user_id'].map(C['ugent']).fillna(0)
    uf['u_act_7d']=uf['user_id'].map(C['ua7']).fillna(0)
    uf['u_act_14d']=uf['user_id'].map(C['ua14']).fillna(0)
    return uf.drop(columns=['age','gender'])

def build_edition_feats(inter_df):
    ef = inter_df.groupby('edition_id').agg(
        e_total=('user_id','count'),
        e_reads=('event_type',lambda x:(x==2).sum()),
        e_wishes=('event_type',lambda x:(x==1).sum()),
        e_avg_rating=('rating','mean'),
        e_n_ratings=('rating',lambda x:x.notna().sum()),
        e_n_users=('user_id','nunique'),
    ).reset_index()
    eds = editions_raw[['edition_id','age_restriction','language_id','publication_year']].copy()
    eds['e_years']=2026-eds['publication_year']
    ap = inter_df.merge(editions_raw[['edition_id','author_id']], on='edition_id').groupby('author_id').size().rename('author_pop')
    ef2 = eds.merge(ef, on='edition_id', how='left')
    ef2 = ef2.merge(editions_raw[['edition_id','author_id']], on='edition_id').merge(ap, on='author_id', how='left')
    return ef2.drop(columns=['author_id','publication_year'])

print('Building training data (all candidates)...')
val_pos = val_inter.groupby(['user_id','edition_id'])['event_type'].max().reset_index()
val_pos['target'] = val_pos['event_type'].map({2:3, 1:1})
val_pos = val_pos[['user_id','edition_id','target']]
pos_set = set(zip(val_pos['user_id'], val_pos['edition_id']))
neg_rows = []
for uid in val_pos['user_id'].unique():
    cs = cand_sets.get(uid, set())
    for eid in cs:
        if (uid, eid) not in pos_set:
            neg_rows.append({'user_id': uid, 'edition_id': eid, 'target': 0})
neg_df = pd.DataFrame(neg_rows)
train_df = pd.concat([val_pos, neg_df], ignore_index=True)

uf = build_user_feats(train_inter, cache)
ef = build_edition_feats(train_inter)
train_df = train_df.merge(uf, on='user_id', how='left')
train_df = train_df.merge(ef, on='edition_id', how='left')
train_df = score_pairs(train_df, emb, cache, label='train')
train_df = add_cand_rank_features(train_df)

exclude = {'user_id','edition_id','target','group_id'}
def safe_ts(x):
    if pd.isna(x): return 0.0
    if hasattr(x,'timestamp'): return float(x.timestamp())
    return float(x) if isinstance(x,(int,float)) else 0.0
for tc in ['u_last_ts','u_first_ts']:
    if tc in train_df.columns: train_df[tc]=train_df[tc].apply(safe_ts)
train_df['language_id']=train_df['language_id'].astype(str).fillna('0')

feature_cols = []
for c in train_df.columns:
    if c in exclude: continue
    if train_df[c].dtype=='object' and c!='language_id': continue
    if c!='language_id': train_df[c]=pd.to_numeric(train_df[c],errors='coerce').fillna(0)
    feature_cols.append(c)

train_df['group_id']=train_df['user_id'].factorize()[0]
train_df = train_df.sort_values('user_id')
cat_feats = [feature_cols.index(c) for c in feature_cols if train_df[c].dtype=='object']

print('Training 5-model ensemble...')
pool = Pool(data=train_df[feature_cols], label=train_df['target'],
            group_id=train_df['group_id'], cat_features=cat_feats)

configs = [
    {'iters':6000, 'depth':8, 'lr':0.02, 'l2':5, 'seed':42, 'loss':'YetiRank', 'bag_temp':0},
    {'iters':5000, 'depth':10, 'lr':0.015, 'l2':9, 'seed':777, 'loss':'YetiRank', 'bag_temp':1},
    {'iters':4500, 'depth':6, 'lr':0.03, 'l2':3, 'seed':2024, 'loss':'YetiRankPairwise', 'bag_temp':0.5},
    {'iters':5000, 'depth':7, 'lr':0.025, 'l2':7, 'seed':314, 'loss':'YetiRank', 'bag_temp':0.3},
    {'iters':4000, 'depth':8, 'lr':0.02, 'l2':4, 'seed':999, 'loss':'YetiRankPairwise', 'bag_temp':0.8},
]
models = []
for idx,cfg in enumerate(configs):
    m = CatBoostRanker(
        iterations=cfg['iters'], learning_rate=cfg['lr'], depth=cfg['depth'],
        l2_leaf_reg=cfg['l2'], verbose=1000, random_seed=cfg['seed'],
        loss_function=cfg['loss'], eval_metric='NDCG:top=20', task_type='GPU',
        bagging_temperature=cfg.get('bag_temp',0),
    )
    m.fit(pool); models.append(m); gc.collect()

def diversity_rerank(items_scores, k=20, n_fixed=7, n_pool=50, lam=0.7):
    sorted_items = sorted(items_scores, key=lambda x: -x[1])[:n_pool]
    if len(sorted_items) <= n_fixed:
        return [eid for eid, _ in sorted_items[:k]]
    scores = np.array([s for _, s in sorted_items])
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        norm_scores = (scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones(len(scores))
    selected = []
    selected_genres = set()
    used = set()
    for i in range(min(n_fixed, len(sorted_items))):
        eid = sorted_items[i][0]
        selected.append(eid)
        used.add(i)
        selected_genres |= set(ed2genres.get(eid, []))
    for pos in range(len(selected), k):
        best_idx = -1
        best_combined = -float('inf')
        for j in range(len(sorted_items)):
            if j in used: continue
            eid = sorted_items[j][0]
            rel = norm_scores[j]
            item_genres = set(ed2genres.get(eid, []))
            new_genres = item_genres - selected_genres
            if len(item_genres) > 0:
                novelty = len(new_genres) / len(item_genres)
            else:
                novelty = 0.0
            combined = lam * rel + (1 - lam) * novelty
            if combined > best_combined:
                best_combined = combined
                best_idx = j
        if best_idx >= 0:
            eid = sorted_items[best_idx][0]
            selected.append(eid)
            used.add(best_idx)
            selected_genres |= set(ed2genres.get(eid, []))
        else:
            break
    for j in range(len(sorted_items)):
        if len(selected) >= k: break
        if j not in used:
            selected.append(sorted_items[j][0])
    return selected[:k]

def local_validate(scored_df, val_df, k=20, use_diversity_rerank=True):
    gt = val_df.groupby(['user_id','edition_id'])['event_type'].max().reset_index()
    gt['rel'] = gt['event_type'].map({2:3, 1:1})
    gt_dict = {(r.user_id,r.edition_id):r.rel for r in gt.itertuples()}
    if use_diversity_rerank:
        recs = {}
        for uid, grp in scored_df.groupby('user_id'):
            items_scores = list(zip(grp['edition_id'].values, grp['score'].values))
            recs[uid] = diversity_rerank(items_scores, k=k, n_fixed=7, n_pool=50, lam=0.7)
    else:
        recs = scored_df.groupby('user_id').apply(lambda g: g.nlargest(k,'score')['edition_id'].tolist()).to_dict()
    scored_items = scored_df.groupby('user_id')['edition_id'].apply(set).to_dict()
    total_genres = set(g for gl in ed2genres.values() for g in gl)
    ndcgs, divs = [], []
    for uid, rl in recs.items():
        dcg = sum(gt_dict.get((uid,e),0)/np.log2(i+2) for i,e in enumerate(rl[:k]))
        pool = scored_items.get(uid, set())
        rels = sorted([gt_dict.get((uid,e),0) for e in pool if gt_dict.get((uid,e),0)>0], reverse=True)
        idcg = sum(r/np.log2(i+2) for i,r in enumerate(rels[:k]))
        ndcgs.append(dcg/idcg if idcg>0 else 0)
        rg=set(); gsets=[]
        for e in rl[:k]:
            gs=set(ed2genres.get(e,[])); rg|=gs; gsets.append(gs)
        cov=len(rg)/len(total_genres) if total_genres else 0
        if len(gsets)>1:
            ild_s,ild_c = 0.0,0
            for a in range(len(gsets)):
                for b in range(a+1,len(gsets)):
                    un=gsets[a]|gsets[b]; it=gsets[a]&gsets[b]
                    ild_s+=1.0-len(it)/max(len(un),1); ild_c+=1
            ild=ild_s/ild_c
        else: ild=0
        divs.append(0.5*cov+0.5*ild)
    an,ad = np.mean(ndcgs),np.mean(divs)
    sc = 0.7*an + 0.3*ad
    print(f'  NDCG@20:      {an:.4f}')
    print(f'  Diversity@20: {ad:.4f}')
    print(f'  SCORE:        {sc:.4f}')
    return sc

print('Local validation...')
val_uids = set(val_pos['user_id'].unique()) & set(candidates['user_id'].unique())
val_cdf = candidates[candidates['user_id'].isin(val_uids)].copy()
val_cdf = val_cdf.merge(uf, on='user_id', how='left')
val_cdf = val_cdf.merge(ef, on='edition_id', how='left')
val_cdf = score_pairs(val_cdf, emb, cache, label='val')
val_cdf = add_cand_rank_features(val_cdf)
for tc in ['u_last_ts','u_first_ts']:
    if tc in val_cdf.columns: val_cdf[tc]=val_cdf[tc].apply(safe_ts)
val_cdf['language_id']=val_cdf['language_id'].astype(str).fillna('0')
for c in feature_cols:
    if c not in val_cdf.columns: val_cdf[c]=0
    if c=='language_id': val_cdf[c]=val_cdf[c].astype(str).fillna('0')
    else: val_cdf[c]=pd.to_numeric(val_cdf[c],errors='coerce').fillna(0)
val_cdf['score']=0
for m in models:
    p=m.predict(val_cdf[feature_cols]); p=(p-p.mean())/(p.std()+1e-8); val_cdf['score']+=p
val_cdf['score']/=len(models)

print('\n  --- Without diversity reranking ---')
local_score_pure = local_validate(val_cdf, val_inter, k=20, use_diversity_rerank=False)
print('\n  --- With diversity reranking (n_fixed=7, lam=0.7) ---')
local_score_div = local_validate(val_cdf, val_inter, k=20, use_diversity_rerank=True)

best_lam = 0.7
best_score = local_score_div
for test_lam in [0.5, 0.6, 0.8, 0.9]:
    recs_test = {}
    for uid, grp in val_cdf.groupby('user_id'):
        items_scores = list(zip(grp['edition_id'].values, grp['score'].values))
        recs_test[uid] = diversity_rerank(items_scores, k=20, n_fixed=7, n_pool=50, lam=test_lam)
    gt = val_inter.groupby(['user_id','edition_id'])['event_type'].max().reset_index()
    gt['rel'] = gt['event_type'].map({2:3, 1:1})
    gt_dict = {(r.user_id,r.edition_id):r.rel for r in gt.itertuples()}
    scored_items = val_cdf.groupby('user_id')['edition_id'].apply(set).to_dict()
    total_genres = set(g for gl in ed2genres.values() for g in gl)
    ndcgs, divs = [], []
    for uid, rl in recs_test.items():
        dcg = sum(gt_dict.get((uid,e),0)/np.log2(i+2) for i,e in enumerate(rl[:20]))
        pool = scored_items.get(uid, set())
        rels = sorted([gt_dict.get((uid,e),0) for e in pool if gt_dict.get((uid,e),0)>0], reverse=True)
        idcg = sum(r/np.log2(i+2) for i,r in enumerate(rels[:20]))
        ndcgs.append(dcg/idcg if idcg>0 else 0)
        rg=set(); gsets=[]
        for e in rl[:20]:
            gs=set(ed2genres.get(e,[])); rg|=gs; gsets.append(gs)
        cov=len(rg)/len(total_genres) if total_genres else 0
        if len(gsets)>1:
            ild_s,ild_c=0.0,0
            for a in range(len(gsets)):
                for b in range(a+1,len(gsets)):
                    un=gsets[a]|gsets[b]; it=gsets[a]&gsets[b]
                    ild_s+=1.0-len(it)/max(len(un),1); ild_c+=1
            ild=ild_s/ild_c
        else: ild=0
        divs.append(0.5*cov+0.5*ild)
    sc = 0.7*np.mean(ndcgs) + 0.3*np.mean(divs)
    print(f'    lam={test_lam}: NDCG={np.mean(ndcgs):.4f} Div={np.mean(divs):.4f} Score={sc:.4f}')
    if sc > best_score:
        best_score = sc; best_lam = test_lam

print(f'\n  Best lambda: {best_lam} -> Score: {best_score:.4f}')

print('Full rebuild and predict...')
full = interactions.copy()
emb_full = build_all_embeddings(full, label='full')
cache_full = build_caches(full, emb_full, label='full')

test_df = candidates[candidates['user_id'].isin(targets['user_id'])].copy()
uf_full = build_user_feats(full, cache_full); ef_full = build_edition_feats(full)
test_df = test_df.merge(uf_full, on='user_id', how='left')
test_df = test_df.merge(ef_full, on='edition_id', how='left')
test_df = score_pairs(test_df, emb_full, cache_full, label='test')
test_df = add_cand_rank_features(test_df)
for tc in ['u_last_ts','u_first_ts']:
    if tc in test_df.columns: test_df[tc]=test_df[tc].apply(safe_ts)
test_df['language_id']=test_df['language_id'].astype(str).fillna('0')
for c in feature_cols:
    if c not in test_df.columns: test_df[c]=0
    if c=='language_id': test_df[c]=test_df[c].astype(str).fillna('0')
    else: test_df[c]=pd.to_numeric(test_df[c],errors='coerce').fillna(0)

test_df['score']=0
for i,m in enumerate(models):
    p=m.predict(test_df[feature_cols]); p=(p-p.mean())/(p.std()+1e-8); test_df['score']+=p
test_df['score']/=len(models)

print('Diversity reranking...')
rows = []
for uid in targets['user_id'].unique():
    grp = test_df[test_df['user_id']==uid]
    if len(grp)==0: continue
    items_scores = list(zip(grp['edition_id'].values, grp['score'].values))
    reranked = diversity_rerank(items_scores, k=20, n_fixed=7, n_pool=50, lam=best_lam)
    for rank, eid in enumerate(reranked, 1):
        rows.append([uid, int(eid), rank])

sub = pd.DataFrame(rows, columns=['user_id','edition_id','rank'])
popular = full['edition_id'].value_counts().head(20).index.tolist()
final = []
for uid in targets['user_id']:
    recs = sub[sub['user_id']==uid]
    if len(recs)==0:
        for r,b in enumerate(popular,1): final.append({'user_id':uid,'edition_id':b,'rank':r})
    elif len(recs)<20:
        exist=set(recs['edition_id']); add=[b for b in popular if b not in exist][:20-len(recs)]
        final.extend(recs.to_dict('records')); cr=int(recs['rank'].max())
        for j,b in enumerate(add,1): final.append({'user_id':uid,'edition_id':b,'rank':cr+j})
    else: final.extend(recs.head(20).to_dict('records'))

fdf = pd.DataFrame(final).sort_values(['user_id','rank'])
fdf['edition_id']=fdf['edition_id'].astype(int); fdf['rank']=fdf['rank'].astype(int)
fdf.to_csv('submission (4) (1).csv', index=False)

print(f'Local val (pure):     {local_score_pure:.4f}')
print(f'Local val (diverse):  {best_score:.4f}')
print(f'Best lambda:          {best_lam}')
print(f'Rows: {len(fdf)}, Users: {fdf["user_id"].nunique()}')