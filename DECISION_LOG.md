# Decision Log

1. **AppleSupport as the brand** — the included sample contains enough AppleSupport interactions to demonstrate the end-to-end pipeline while keeping the taxonomy narrow.
2. **Small intent taxonomy** — the assignment asks for a small set of intents; over-fragmentation would make the evaluation less reliable.
3. **Word + character TF-IDF** — robust to short, noisy, misspelled Twitter text and fast enough for a <15-minute reproduction.
4. **Logistic regression** — interpretable, fast, and produces probabilities usable for escalation.
5. **Historical retrieval before generation** — the assignment specifically asks for replies grounded in historical resolution behavior.
6. **Brand-specific reconstruction** — the source dataset has no direct brand field, so linked outbound support authors define the brand slice.
7. **Conservative escalation** — low confidence should fail safe rather than produce an unsupported answer.
8. **No credential collection** — a support agent should not request passwords or unnecessary secrets.
9. **No fabricated golden labels** — the assignment explicitly requires hand-labelled examples; synthetic labels would invalidate the evaluation.
10. **200-row annotation template** — centered in the required 150–250 range and easy to freeze before evaluation.
11. **Two baselines** — majority and transparent keyword rules separate "learning" gains from trivial heuristics.
12. **LLM judge is optional at runtime** — the core harness remains runnable without an API key, while the judge can be enabled for the required qualitative evaluation.
13. **Human-vs-judge agreement is measured separately** — judge scores without calibration are not evidence of trustworthiness.
14. **Historical policy caveat** — historical replies demonstrate behavior, not necessarily current product policy.
