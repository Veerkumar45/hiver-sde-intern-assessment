# Updated run state

The project was updated to make the evaluation methodology safer:
- retrieval excludes the current evaluated tweet;
- golden examples are excluded from training/retrieval;
- a 200-row golden annotation file can be generated;
- majority and keyword baselines are reported;
- accuracy, macro F1 and per-intent metrics are reported;
- optional LLM judging is supported.

Run the README commands from the project root.
