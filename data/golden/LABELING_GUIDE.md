# Golden Set Labeling Guide

Label the `intent` column independently from the model prediction.

Use exactly one of:

- `update_performance_or_crash`
- `battery_drain`
- `app_or_media_behavior`
- `account_or_verification`
- `device_issue_other`

Rules:
1. Label the customer's actual primary problem, not the historical agent reply.
2. If multiple issues are present, choose the issue driving the support request.
3. Use `device_issue_other` when none of the first four categories is a defensible fit.
4. Do not use model predictions as ground truth.
5. `reply_quality_human_0_2`: 0 = poor, 1 = acceptable, 2 = strong.
6. `escalation_human`: use `yes` when a human should handle it, otherwise `no`.
7. Explain ambiguous decisions in `notes`.
