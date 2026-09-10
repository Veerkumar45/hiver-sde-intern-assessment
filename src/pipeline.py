"""
Hiver SDE Intern Assessment - final evaluation-safe pipeline.

Key design:
- AppleSupport historical data is used as the retrieval corpus.
- Golden examples are evaluation-only: their text is predicted directly, even when
  their original tweet_id is outside the selected raw-data sample.
- Golden texts are excluded from the retrieval corpus by normalized exact match.
- TF-IDF is fitted once, then all queries are scored in batches.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INTENTS = [
    "update_performance_or_crash",
    "battery_drain",
    "app_or_media_behavior",
    "account_or_verification",
    "device_issue_other",
]

KEYWORDS = {
    "battery_drain": ["battery", "drain", "charging", "charge", "battery life", "dies"],
    "update_performance_or_crash": [
        "update", "ios", "macos", "crash", "crashes", "crashed", "freeze",
        "freezes", "slow", "lag", "performance", "restart", "reboot", "bug",
    ],
    "account_or_verification": [
        "account", "login", "log in", "sign in", "password", "verify",
        "verification", "locked", "icloud", "id",
    ],
    "app_or_media_behavior": [
        "app", "apps", "itunes", "music", "podcast", "video", "photo",
        "camera", "safari", "browser", "notification", "message", "messages",
        "airplay", "download", "play", "screen", "keyboard",
    ],
}

ESCALATION_TERMS = [
    "refund", "charged twice", "unauthorized", "hack", "hacked", "stolen",
    "lawsuit", "legal", "data loss", "lost data", "security breach",
]


def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classify_keyword(text: str) -> str:
    t = norm(text)
    scores = {k: sum(1 for w in ws if w in t) for k, ws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "device_issue_other"


def escalation(text: str, confidence: float, retrieval_score: float) -> tuple[bool, str]:
    t = norm(text)
    hits = [x for x in ESCALATION_TERMS if x in t]
    if hits:
        return True, "Escalated because the message contains a high-risk/support-sensitive term: " + ", ".join(hits[:3])
    if confidence < 0.55:
        return True, "Escalated because intent confidence is low."
    if retrieval_score < 0.18:
        return True, "Escalated because no sufficiently similar historical example was retrieved."
    return False, "Auto-handle: intent confidence and historical similarity passed the current thresholds."


def make_reply(intent: str, text: str, retrieval_text: str = "") -> str:
    if intent == "battery_drain":
        return "Sorry you're dealing with battery drain. Please check Battery settings for apps using the most power, install any pending software updates, and let us know if the issue continues."
    if intent == "update_performance_or_crash":
        return "Sorry about the performance/crash issue. Please make sure your device is updated, restart it, and check whether the problem continues after the restart."
    if intent == "account_or_verification":
        return "Sorry you're having trouble with your account. Please confirm that you're using the correct account credentials and follow the official verification or password-reset flow. If you remain locked out, we can help escalate the case."
    if intent == "app_or_media_behavior":
        return "Sorry you're having trouble with the app or media feature. Please restart the affected app/device and check for available app and software updates. If it continues, tell us the app and the exact behavior you're seeing."
    return "Sorry you're experiencing this issue. Please restart the device, check for software updates, and share the device model and the exact steps that reproduce the problem so we can narrow it down."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--brand", default="AppleSupport")
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data, nrows=args.limit * 2)
    cols = {c.lower(): c for c in raw.columns}
    text_col = cols.get("text") or cols.get("tweet_text")
    id_col = cols.get("tweet_id") or cols.get("id")
    if not text_col or not id_col:
        raise ValueError(f"Could not find text/id columns. Found: {list(raw.columns)}")

    # Keep AppleSupport rows. Some versions of TWCS store the brand in in_reply_to_user_id
    # rather than a direct brand column, so retain rows whose text mentions the support handle
    # when no explicit brand column is available.
    brand_col = next((c for c in raw.columns if c.lower() in {"brand", "author", "username", "user_name"}), None)
    if brand_col:
        mask = raw[brand_col].astype(str).str.lower().eq(args.brand.lower())
        hist = raw.loc[mask, [id_col, text_col]].copy()
    else:
        hist = raw[raw[text_col].astype(str).str.contains(re.escape("@" + args.brand), case=False, na=False)][[id_col, text_col]].copy()

    hist.columns = ["tweet_id", "text"]
    hist["text"] = hist["text"].astype(str)
    hist = hist.drop_duplicates(subset=["tweet_id"]).reset_index(drop=True)

    golden = None
    if args.golden:
        golden = pd.read_csv(args.golden)
        if "text" not in golden.columns or "intent" not in golden.columns:
            raise ValueError("Golden file must contain text and intent columns.")
        golden["text"] = golden["text"].astype(str)
        golden_ids = set(golden["tweet_id"].astype(str)) if "tweet_id" in golden.columns else set()
        golden_norms = set(golden["text"].map(norm))
        hist = hist[~hist["text"].map(norm).isin(golden_norms)].copy()
    else:
        golden_ids = set()

    # Fit retrieval once on historical corpus.
    corpus = hist["text"].tolist()
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=1, max_features=120000
    )
    X = vectorizer.fit_transform(corpus if corpus else [""])

    def predict_frame(df: pd.DataFrame, is_golden=False):
        texts = df["text"].astype(str).tolist()
        Q = vectorizer.transform(texts)
        sims = cosine_similarity(Q, X, dense_output=False)
        rows = []
        for i, text in enumerate(texts):
            row = sims.getrow(i)
            if row.nnz:
                j = row.indices[row.data.argmax()]
                rscore = float(row.data.max())
                rtext = corpus[j]
                retrieval_intent = classify_keyword(rtext)
            else:
                rscore = 0.0
                rtext = ""
                retrieval_intent = "device_issue_other"

            keyword_intent = classify_keyword(text)
            # Hybrid: keyword evidence is primary; retrieval breaks ties when similarity is strong.
            if retrieval_intent != "device_issue_other" and rscore >= 0.30:
                if keyword_intent == "device_issue_other":
                    intent = retrieval_intent
                    conf = min(0.95, 0.60 + 0.35 * rscore)
                else:
                    intent = keyword_intent
                    conf = min(0.97, 0.58 + 0.30 * rscore)
            else:
                intent = keyword_intent
                conf = 0.78 if keyword_intent != "device_issue_other" else max(0.50, 0.58 + 0.12*rscore)

            esc, reason = escalation(text, conf, rscore)
            rows.append({
                "tweet_id": str(df.iloc[i]["tweet_id"]),
                "text": text,
                "intent": intent,
                "confidence": round(conf, 6),
                "retrieval_score": round(rscore, 6),
                "escalate": esc,
                "escalation_reason": reason,
                "reply": make_reply(intent, text, rtext),
                "is_golden": bool(is_golden),
            })
        return pd.DataFrame(rows)

    production = hist.head(args.limit).copy()
    # If golden exists, create a separate prediction workload from golden text.
    # Do not require golden IDs to exist in the raw AppleSupport sample.
    pred_prod = predict_frame(production, False)
    if golden is not None:
        if "tweet_id" not in golden.columns:
            golden.insert(0, "tweet_id", [f"golden_{i+1}" for i in range(len(golden))])
        pred_gold = predict_frame(golden[["tweet_id", "text"]].copy(), True)
        predictions = pd.concat([pred_prod, pred_gold], ignore_index=True)
    else:
        predictions = pred_prod

    predictions.to_csv(outdir / "predictions.csv", index=False)
    hist.to_csv(outdir / "historical_pairs.csv", index=False)

    summary = {
        "brand": args.brand,
        "historical_rows": int(len(hist)),
        "production_predictions": int(len(pred_prod)),
        "golden_predictions": int(len(golden)) if golden is not None else 0,
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
