# SLQP 200-response scoring protocol

Score every assigned record using the provided `problem`, `ground_truth`, and full `response`.

Use the project's original seven rubrics. Every rubric is one of
`1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0`.

1. `understanding`: Problem Understanding and Constraint Use. Weight 0.15.
2. `rigor`: Mathematical Rigor. Weight 0.15.
3. `correctness`: Answer Correctness and Verifiability. Weight 0.20.
4. `exploration`: Exploration and Exploitation. Weight 0.20.
5. `reasonableness`: Solution Reasonableness. Weight 0.125.
6. `fluency`: Expression Fluency. Weight 0.10.
7. `conciseness`: Expression Conciseness. Weight 0.075.

Anchors:

- 4.0: excellent, essentially no substantive issue.
- 3.0: generally good, only minor flaws.
- 2.0: clear issues, but meaningful/relevant mathematical content remains.
- 1.0: no effective mathematical content, unrelated, impossible to evaluate, or wholly broken.

Principles:

- Judge the mathematical trajectory, not only the final answer.
- A wrong final answer can retain partial credit for correct and relevant progress.
- A correct final answer with invalid/unsupported reasoning must lose rigor,
  exploration, and reasonableness credit.
- Do not reward long, repetitive, random, malformed, or non-progressing text.
- `finish_reason=length` and truncation are not quality defects by themselves and
  must never trigger an automatic penalty. Judge only the mathematical content
  actually present before the cutoff. A truncated trajectory with strong,
  relevant, rigorous progress can still receive high understanding, rigor,
  exploration, reasonableness, fluency, and conciseness scores. Only observed
  failures such as repetition, invalid reasoning, aimless wandering, or lack of
  useful progress should lower those rubrics. The absence of a final answer may
  affect correctness/verifiability, but must not collapse the other dimensions.
- Compare the stated final answer against `ground_truth`, including requested format.

For each input record, emit exactly one JSON object on one line with this schema:

```json
{
  "id": "sample_...",
  "rubric_scores": {
    "understanding": 1.0,
    "rigor": 1.0,
    "correctness": 1.0,
    "exploration": 1.0,
    "reasonableness": 1.0,
    "fluency": 1.0,
    "conciseness": 1.0
  },
  "weighted_score_1_to_4": 1.0,
  "quality_score": 25.0,
  "answer_status": "correct|partially_correct|wrong|no_final_answer|unverifiable",
  "trajectory_status": "complete|truncated_but_useful|truncated|repetitive|malformed|off_topic",
  "brief_reason": "one concise factual sentence"
}
```

Compute:

`weighted_score_1_to_4 = 0.15*understanding + 0.15*rigor + 0.20*correctness + 0.20*exploration + 0.125*reasonableness + 0.10*fluency + 0.075*conciseness`

`quality_score = 25 * weighted_score_1_to_4`

Keep six decimals or fewer. Do not skip records. Do not add prose outside JSONL.
