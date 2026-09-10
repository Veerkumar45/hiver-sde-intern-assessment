import pandas as pd
from src.pipeline import clean_text, heuristic_intent, escalation, make_retriever, retrieve

def test_clean():
    assert "URL" in clean_text("HELLO https://example.com")

def test_battery_intent():
    assert heuristic_intent("my battery drains very fast") == "battery_drain"

def test_escalation():
    esc, reason = escalation("my account was hacked", 0.9, 0.9)
    assert esc

def test_retrieval_excludes_current_tweet():
    df = pd.DataFrame([
        {"tweet_id":"1","customer_text":"battery drains fast","response_text":"DM us"},
        {"tweet_id":"2","customer_text":"battery issue after update","response_text":"Restart"},
    ])
    ret = retrieve("battery drains fast", df, make_retriever(df), k=2, exclude_tweet_id="1")
    assert ret
    assert ret[0][1] != "battery drains fast"
