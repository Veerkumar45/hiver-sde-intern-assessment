"""Evaluate intent predictions against the golden set and compare simple baselines."""
from __future__ import annotations

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

INTENTS = [
    "update_performance_or_crash",
    "battery_drain",
    "app_or_media_behavior",
    "account_or_verification",
    "device_issue_other",
]

KEYWORDS = {
    "battery_drain": ["battery", "drain", "charging", "charge", "battery life", "dies"],
    "update_performance_or_crash": ["update", "ios", "macos", "crash", "crashes", "crashed", "freeze", "freezes", "slow", "lag", "performance", "restart", "reboot", "bug"],
    "account_or_verification": ["account", "login", "log in", "sign in", "password", "verify", "verification", "locked", "icloud", "id"],
    "app_or_media_behavior": ["app", "apps", "itunes", "music", "podcast", "video", "photo", "camera", "safari", "browser", "notification", "message", "messages", "airplay", "download", "play", "screen", "keyboard"],
}

def classify_keyword(text):
    t = str(text).lower()
    scores = {k: sum(1 for w in ws if w in t) for k, ws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "device_issue_other"

def report(name, y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENTS, zero_division=0
    )
    print(f"\n{name}")
    print(f"n={len(y_true)} accuracy={accuracy_score(y_true,y_pred):.4f} macro_f1={f1_score(y_true,y_pred,labels=INTENTS,average='macro',zero_division=0):.4f}")
    for intent, pp, rr, ff in zip(INTENTS,p,r,f):
        print(f"  {intent}: precision={pp:.4f} recall={rr:.4f} f1={ff:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions)
    gold = pd.read_csv(args.golden)

    if "tweet_id" not in pred.columns:
        raise ValueError("predictions.csv needs tweet_id")
    if "intent" not in pred.columns:
        raise ValueError("predictions.csv needs intent")
    if "tweet_id" not in gold.columns or "intent" not in gold.columns:
        raise ValueError("golden file needs tweet_id and intent")

    pred["tweet_id"] = pred["tweet_id"].astype(str)
    gold["tweet_id"] = gold["tweet_id"].astype(str)

    # Prefer golden-marked predictions when available; otherwise join by ID.
    if "is_golden" in pred.columns:
        gp = pred[pred["is_golden"].astype(str).str.lower().isin(["true","1"])].copy()
    else:
        gp = pred[pred["tweet_id"].isin(set(gold["tweet_id"]))].copy()

    merged = gold[["tweet_id","text","intent"]].merge(
        gp[["tweet_id","intent"]].rename(columns={"intent":"pred_intent"}),
        on="tweet_id", how="left"
    )

    missing = merged["pred_intent"].isna().sum()
    if missing:
        raise ValueError(f"Missing {missing} golden predictions. Re-run pipeline with --golden.")

    y = merged["intent"].astype(str)
    yp = merged["pred_intent"].astype(str)

    majority = y.value_counts().idxmax()
    report("Majority baseline", y, [majority]*len(y))
    report("Keyword baseline", y, [classify_keyword(x) for x in merged["text"]])
    report("Retrieval/Hybrid agent", y, yp)

    out = pd.DataFrame({
        "tweet_id": merged["tweet_id"],
        "gold_intent": y,
        "pred_intent": yp,
    })
    out.to_csv("outputs/evaluation_comparison.csv", index=False)
    print(f"\nMatched golden examples: {len(merged)} / {len(gold)}")
    print("Saved outputs/evaluation_comparison.csv")

if __name__ == "__main__":
    main()
