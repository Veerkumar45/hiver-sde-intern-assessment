# Assessment Report — AppleSupport AI Support Agent

## 1. Problem framing

The objective is not to build a generic chatbot. It is to build a narrow, auditable support agent that can:
1. classify an inbound AppleSupport customer message into a small set of intents;
2. draft a response using historical AppleSupport evidence and conservative troubleshooting language;
3. decide whether automation is safe, with an explicit escalation reason.

For this implementation, "good" means:
- intent classification is measured on a frozen 200-example evaluation set;
- the system is compared against simple baselines;
- historical examples are used for retrieval without using the evaluated message itself;
- low-confidence / weak-evidence cases can be escalated;
- evaluation is reproducible and reports limitations rather than hiding them.

### What is intentionally not built

- No autonomous account changes, refunds, purchases, or credential handling.
- No claim that a historical response is a current Apple policy.
- No full-scale fine-tuning.
- No attempt to cover every possible support intent.

The supplied assignment emphasizes evidence over system complexity and asks for a 150–250 example golden set, automated metrics, baselines, failure analysis, a misleading-headline-number section, and a decision log.

## 2. Architecture

`TWCS CSV -> AppleSupport filtering -> historical retrieval corpus -> hybrid intent classification -> conservative reply -> escalation gate -> evaluation`

### Data and retrieval

The final run used the supplied TWCS CSV and produced **2,805 usable AppleSupport historical rows** in the selected workload. The command used a 50,000-row input limit, but only 2,805 rows satisfied the AppleSupport filtering condition; the report therefore uses 2,805 rather than claiming 50,000 AppleSupport examples.

The 200 golden messages are evaluated directly from the frozen golden file. They do not have to appear in the selected raw-data sample. Golden texts are excluded from the retrieval corpus by normalized exact text match, reducing direct leakage.

### Classifier

The final implementation is a lightweight hybrid:
- transparent keyword evidence identifies strong intent signals;
- TF-IDF retrieval finds similar historical AppleSupport messages;
- retrieval evidence can resolve otherwise weak/unknown keyword cases;
- confidence and retrieval similarity feed the escalation gate.

The five frozen intents are:
1. `update_performance_or_crash`
2. `battery_drain`
3. `app_or_media_behavior`
4. `account_or_verification`
5. `device_issue_other`

### Reply generation

Replies are deliberately conservative and operationally safe. They provide common troubleshooting steps, ask for useful diagnostic context where appropriate, and avoid inventing refunds, policy decisions, or account actions.

### Escalation

The system escalates when high-risk language is detected, intent confidence is low, or historical retrieval evidence is weak. Otherwise it permits auto-handling.

## 3. Baselines

### Baseline A — majority intent

Always predicts the most frequent intent in the 200-example golden set.

### Baseline B — keyword rules

Uses a transparent keyword map for battery, account/verification, app/media, and update/crash issues. Anything unmatched falls into `device_issue_other`.

### Final agent — retrieval/hybrid

Combines the keyword classifier with TF-IDF historical retrieval and the escalation gate.

All three approaches are evaluated against the same frozen 200 examples.

## 4. Evaluation

The final evaluation contains **200/200 matched golden examples**.

### Intent results

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 36.50% | 0.1070 |
| Keyword baseline | 44.50% | 0.4902 |
| **Retrieval/hybrid agent** | **51.50%** | **0.5465** |

The final agent improves over the keyword baseline by **7.0 percentage points in accuracy** and **0.0563 Macro-F1**.

### Per-intent F1 — final agent

| Intent | Precision | Recall | F1 |
|---|---:|---:|---:|
| update_performance_or_crash | 0.3279 | 0.7692 | 0.4598 |
| battery_drain | 0.9167 | 0.7333 | 0.8148 |
| app_or_media_behavior | 0.5600 | 0.3836 | 0.4553 |
| account_or_verification | 0.4375 | 0.4375 | 0.4375 |
| device_issue_other | 0.6066 | 0.5286 | 0.5649 |

The strongest class is `battery_drain`. The weakest recalls are `app_or_media_behavior` and `account_or_verification`, showing that short or overlapping support messages remain difficult.

### Reply-quality evaluation limitation

The repository contains a reply-review artifact with a 0–2 quality rubric. The current non-API judge is **rule-based, not an LLM**, and the uploaded review labels were model-assisted rather than independently collected human labels. Therefore, the submission does **not** present the resulting agreement number as validated human-vs-LLM agreement.

This is an intentional limitation: it is better to disclose the evaluation boundary than to claim human or LLM-judge evidence that was not independently produced.

## 5. Failure analysis

The following are the five failure modes I would prioritize based on the observed intent results and system design.

### 1. Mixed-intent messages

A tweet can mention an update, battery drain, and crashing in the same message. A single-label taxonomy forces the classifier to choose one dominant intent.

**Hypothesis:** keyword and retrieval evidence can disagree when multiple issue terms are present.

**Mitigation:** move to multi-label candidate generation followed by a primary-intent selector, and explicitly expose secondary intents internally.

### 2. Sparse context

Messages such as "it's not working" or "please help" contain too little information to reliably distinguish device, app, account, or update problems.

**Hypothesis:** the classifier is being asked to infer information that is absent from the message.

**Mitigation:** route low-information messages to clarification-first responses rather than forcing a confident intent.

### 3. Historical drift

A retrieved historical workaround may have been valid when the tweet was posted but may not represent current product behavior.

**Hypothesis:** lexical similarity does not guarantee temporal validity.

**Mitigation:** add timestamps, prefer recent evidence, and attach a freshness penalty to old examples.

### 4. App/media versus device ambiguity

The `app_or_media_behavior` class has only **0.3836 recall**, while `device_issue_other` has substantially better precision. Messages about screens, apps, media playback, or device behavior often share vocabulary.

**Hypothesis:** overlapping terms such as "screen", "play", "message", or "app" are not sufficient to identify the actual failure boundary.

**Mitigation:** use conversation context, character/word embeddings, and a clarification question when the top two intents are close.

### 5. Account/verification ambiguity

`account_or_verification` has **0.4375 F1**, indicating that account terms alone do not reliably separate authentication problems from general device or service issues.

**Hypothesis:** words such as "ID", "account", and "login" occur in otherwise unrelated support requests.

**Mitigation:** add intent-specific phrase features, contextual retrieval, and a conservative escalation rule for sensitive account requests.

## 6. What is misleading about my headline number?

The headline **51.5% accuracy** is useful but incomplete.

It can be misleading because:
- the taxonomy contains subjective boundaries between overlapping intents;
- accuracy hides per-class behavior;
- the golden set contains only 200 examples;
- `app_or_media_behavior` and `account_or_verification` remain difficult despite the overall improvement;
- replies can sound plausible without proving that historical evidence is still valid;
- historical Twitter support is not equivalent to current product policy;
- near-duplicate messages can inflate retrieval similarity if leakage controls are not enforced.

Therefore, the headline should always be accompanied by Macro-F1, per-intent precision/recall, baseline comparisons, and explicit evaluation limitations.

The most defensible claim is:

> **On a frozen 200-example evaluation set, the retrieval/hybrid classifier achieved 51.5% accuracy and 0.5465 Macro-F1, outperforming a keyword baseline at 44.5% accuracy and 0.4902 Macro-F1.**

## 7. Non-obvious engineering decisions

1. **Keep the taxonomy deliberately small.** Five intents make errors inspectable instead of creating a long, noisy label list.
2. **Use a frozen golden set.** Evaluation should not change after seeing model predictions.
3. **Predict golden examples directly.** Golden IDs do not need to exist in the selected raw-data sample.
4. **Exclude golden text from retrieval.** This reduces direct text leakage.
5. **Exclude the evaluated message itself from retrieval.** Otherwise nearest-neighbor similarity can become artificially perfect.
6. **Fit TF-IDF once.** Re-fitting inside every prediction loop is unnecessarily expensive and makes the experiment difficult to reproduce.
7. **Use keyword evidence as a transparent fallback.** A simple baseline remains valuable when a learned/retrieval component fails.
8. **Do not fabricate current policy from historical tweets.** Historical support is evidence, not authority.
9. **Use an explicit escalation reason.** A boolean alone is not auditable.
10. **Escalate on weak evidence.** The system should prefer a human review over unsupported automation.
11. **Separate production predictions from golden predictions.** This prevents evaluation examples from silently becoming part of the production workload.
12. **Report the actual available AppleSupport sample.** The input limit was 50,000, but only 2,805 rows were usable after filtering.
13. **Keep reply generation conservative.** The agent asks for diagnostic information instead of inventing product-specific guarantees.
14. **Treat evaluation limitations as results.** A non-LLM local judge is explicitly labeled as such instead of being presented as an LLM judge.
15. **Optimize for reproducibility over model complexity.** The final system can be run locally from the supplied dataset without requiring a hosted model API.

## 8. What I would do with one more week

- Independently hand-label and adjudicate 250 examples.
- Add temporal and conversation-level leakage controls.
- Build a better taxonomy from clustering plus human review rather than keyword bootstrapping alone.
- Calibrate intent probabilities and tune escalation thresholds on a validation split.
- Compare TF-IDF retrieval against dense embeddings.
- Add a structured evidence object containing retrieved examples, scores, and timestamps.
- Add a real, independently run LLM judge and blind human comparison.
- Measure escalation precision/recall against independently reviewed labels.
- Add a small FastAPI or Streamlit demonstration.
- Add regression tests for the five failure modes.
- Add response-groundedness checks that verify every troubleshooting recommendation against retrieved evidence.
