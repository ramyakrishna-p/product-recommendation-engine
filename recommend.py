import numpy as np, pandas as pd, json, pathlib, itertools, collections
np.random.seed(21)
root=pathlib.Path(__file__).parent; (root/"data").mkdir(exist_ok=True); (root/"results").mkdir(exist_ok=True)
products=["Laptop","Headphones","Tablet","Jacket","Shoes","Lamp","Chair","Serum","Perfume","Bike","Kettle","T-Shirt"]
rows=[]
for u in range(1,401):
    basket=np.random.choice(products, size=int(np.random.randint(2,6)), replace=False, p=[.12,.14,.08,.09,.11,.07,.06,.07,.05,.06,.07,.08])
    for pr in basket: rows.append([u,pr])
df=pd.DataFrame(rows,columns=["user_id","product"]); df.to_csv(root/"data/transactions.csv",index=False)
# FP-Growth style: frequent pairs via counting (mlxtend-free implementation)
pair_counts=collections.Counter()
baskets=df.groupby("user_id").product.apply(list)
for b in baskets:
    for a,c in itertools.combinations(sorted(set(b)),2): pair_counts[(a,c)]+=1
support={f"{a}|{c}":round(cnt/len(baskets),4) for (a,c),cnt in pair_counts.most_common(10)}
# Collaborative filtering: cosine on user-item matrix
mat=df.assign(v=1).pivot_table(index="user_id",columns="product",values="v",fill_value=0)
from numpy.linalg import norm
class Recommender:
    def __init__(self, mat): self.mat=mat; self.sim=mat.values @ mat.values.T; 
    def recommend(self, user_id, k=3):
        if user_id not in self.mat.index: return []
        idx=list(self.mat.index).index(user_id); scores={}
        user_items=set(self.mat.columns[self.mat.iloc[idx]>0])
        # item co-occurrence scoring
        for (a,c),cnt in pair_counts.items():
            if a in user_items and c not in user_items: scores[c]=scores.get(c,0)+cnt
            if c in user_items and a not in user_items: scores[a]=scores.get(a,0)+cnt
        return [p for p,_ in sorted(scores.items(), key=lambda x:-x[1])[:k]]
rec=Recommender(mat)
example={str(u): rec.recommend(u,3) for u in [1,2,3,10,20]}
# Precision@k split: hold out 1 item per user with >=3 items
hits=0; total=0
for u,b in baskets.items():
    if len(b)>=3:
        held=b[-1]; train_user_items=set(b[:-1]); total+=1
        # predict from train items using pair counts
        scores={}
        for (a,c),cnt in pair_counts.items():
            if a in train_user_items: scores[c]=scores.get(c,0)+cnt
            if c in train_user_items: scores[a]=scores.get(a,0)+cnt
        top=[p for p,_ in sorted(scores.items(), key=lambda x:-x[1])[:3]]
        if held in top: hits+=1
prec=round(hits/total,4) if total else 0.0
summary={"users":int(df.user_id.nunique()),"transactions":int(len(df)),"products":len(products),"top_pair_support":support,"example_recommendations":example,"precision_at_3_holdout":prec,"holdout_users_evaluated":int(total),"hits":int(hits)}
(root/"results/summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
