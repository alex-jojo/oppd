# SLQP calibration result for Qwen3-0.6B-Base

## Data and scoring

- 200 dapo14k questions and Qwen3-0.6B-Base responses.
- Every valid generated response token has a final-layer hidden state in the
  original collection. The server training package does not include those
  large hidden arrays.
- Ground-truth answers were restored by exact `dataset_index` join.
- GPT-5.5 scored all 200 responses with the project's original seven rubrics.
- `finish_reason=length` and truncation were not treated as defects by
  themselves. Only the mathematical content actually present was scored.

## Score distribution

- Mean quality score: 29.8625 / 100.
- Median: 25.0.
- Range: 25.0 to 99.0625.
- Correct / partially correct / wrong / no final answer: 4 / 4 / 38 / 154.

The low distribution is caused by visible base-model degeneration: repeated
symbols, malformed continuations, unrelated text, and invalid mathematics.
It is not an automatic truncation penalty.

## Geometry

The frozen SLQP representation remains the previously selected differentiable
student-only four-feature representation:

1. mean prompt-relative hidden distance;
2. std prompt-relative hidden distance;
3. mean local hidden step;
4. std local hidden step.

The four standardized features are projected on PCA-1. Rubric quality only
orients the sign. KMeans-2 on the projection supplies the frozen midpoint.

Diagnostics on 200 samples:

- quality Pearson: 0.5391;
- quality Spearman: 0.6325;
- quality Pearson after linearly controlling log response length: 0.5030;
- 1-D two-cluster silhouette: 0.6663;
- exact bottom-50 versus top-50 AUC: 0.9044;
- exact bottom-50 versus top-50 boundary accuracy: 0.78;
- cluster mean quality gap: 8.5821 points;
- 500-run bootstrap direction cosine mean: 0.99984;
- bootstrap direction cosine 5th percentile: 0.99954.

The negative projection/length correlation is real, because many low-quality
responses repeat until the token limit. It is not the whole signal: the
quality relation remains 0.503 after controlling log length. Case inspection
shows the low pole is blank/repeated-symbol behavior, while the high pole is
structured mathematical attempt. The axis is therefore best described as
trajectory health / mathematical engagement, not final-answer correctness.

This distinction is intentional. SLQP tests whether a student-only latent
trajectory-health objective can move a base model out of degenerate rollout
regions. Vanilla OPD remains the teacher-supervised correctness baseline.
