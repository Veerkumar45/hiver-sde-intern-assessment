import argparse, pandas as pd, numpy as np, re
from pathlib import Path
from src.pipeline import heuristic_intent, clean_text

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/historical_pairs.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data/golden/golden_eval.csv")
    a=ap.parse_args()
    df=pd.read_csv(a.pairs).drop_duplicates("tweet_id").copy()
    df["bootstrap_intent"]=df["customer_text"].map(heuristic_intent)
    df["length"]=df["customer_text"].map(lambda x: len(clean_text(x)))
    # Prefer diverse lengths and intents. The labels remain blank and must be human supplied.
    rng=np.random.default_rng(42)
    groups=[]
    for label,g in df.groupby("bootstrap_intent"):
        take=max(1, round(a.n*len(g)/len(df)))
        groups.append(g.sample(min(take,len(g)), random_state=42))
    out=pd.concat(groups).drop_duplicates("tweet_id")
    if len(out)<a.n:
        rest=df[~df.tweet_id.isin(out.tweet_id)]
        out=pd.concat([out,rest.sample(min(a.n-len(out),len(rest)),random_state=42)])
    out=out.sample(min(a.n,len(out)),random_state=42).reset_index(drop=True)
    result=pd.DataFrame({
        "id": np.arange(1,len(out)+1),
        "tweet_id": out["tweet_id"].astype(str),
        "text": out["customer_text"].astype(str),
        "intent": "",
        "reply_quality_human_0_2": "",
        "escalation_human": "",
        "notes": "",
    })
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.out,index=False)
    print(f"Created {len(result)} annotation rows at {a.out}")
    print("IMPORTANT: intent labels are intentionally blank. A human must label them before evaluation.")

if __name__=="__main__":
    main()
