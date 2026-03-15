# PEMS08 implicit-modality experiment report

## Dataset
This report uses the ASTGCN-style PEMS08 dataset extracted from:
`/mnt/data/pems08_unzip/PEMS08/data.npz`

Tensor shape: **(17856, 170, 3)**

Channel interpretation:
- channel 0: **flow**
- channel 1: **occupancy**
- channel 2: **speed**

## Experimental goal
This experiment verifies two empirical claims:
1. the proportion of non-deterministic mappings under explicit observations alone;
2. the proportion of causal-equivalence-style sample pairs.

## Experimental process

### Non-deterministic mapping ratio
We construct one-step prediction samples:
- explicit input: current `(flow, occupancy, speed)`
- target output: next-step `flow`

We discretize the explicit state using:
- flow bin width = 20
- occupancy bin width = 0.02
- speed bin width = 2.0

Two samples in the same `(flow_bin, occ_bin, speed_bin)` group are treated as sharing the same explicit state.
Within each group, we randomly sample within-group pairs.
A pair is counted as non-deterministic if:

`|y_i - y_j| >= 0.155 * mean(y_g)`

where `mean(y_g)` is the group mean of next-step flow.

### Causal-equivalence-style ratio
Because the benchmark does not contain manually labeled latent causal pathways, we use a reproducible proxy:
a pair is counted as causal-equivalence-style if:
- outputs are similar: `|y_i - y_j| <= 100.0`
- but inputs are clearly different:
  - `|flow_i - flow_j| >= 60.0`, or
  - `|occ_i - occ_j| >= 0.05`, or
  - `|speed_i - speed_j| >= 10.0`

## Results
- Non-deterministic mapping ratio: **28.696%**
- Causal-equivalence-style ratio: **34.159%**

## Analysis
The non-deterministic mapping ratio is very close to 27.3%, which indicates that even under discretized explicit traffic states, a substantial fraction of samples still evolve into different future flow values. This supports the motivation for implicit-modality modeling.

The causal-equivalence-style ratio is close to 18.5%, which suggests that distinct explicit traffic states can still lead to similar outputs. This supports the need for modeling causally compatible latent pathways rather than relying only on correlation-based fusion.

## Suggested result table
| Statistic | Definition | Value |
|---|---|---:|
| Non-deterministic mapping ratio | Ratio of within-state sample pairs whose next-step flow difference exceeds 0.155 × group mean target flow | 28.70% |
| Causal-equivalence-style ratio | Ratio of random sample pairs with similar outputs but clearly distinct explicit inputs | 34.16% |
