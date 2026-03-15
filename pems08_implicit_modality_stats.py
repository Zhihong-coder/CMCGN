import argparse
from collections import defaultdict
import json
import numpy as np
import pandas as pd

def load_pems08(npz_path):
    arr = np.load(npz_path)["data"]  # [T, N, 3]
    flow = arr[:, :, 0]
    occ = arr[:, :, 1]
    speed = arr[:, :, 2]
    return flow, occ, speed

def build_one_step_samples(flow, occ, speed):
    x_flow = flow[:-1].reshape(-1)
    x_occ = occ[:-1].reshape(-1)
    x_speed = speed[:-1].reshape(-1)
    y_next = flow[1:].reshape(-1)
    return x_flow, x_occ, x_speed, y_next

def nondeterministic_mapping_ratio(
    x_flow, x_occ, x_speed, y_next,
    flow_bin_width=20, occ_bin_width=0.02, speed_bin_width=2.0,
    rel_gap=0.155, max_pairs_per_group=20, seed=42
):
    flow_bin = (x_flow // flow_bin_width).astype(int)
    occ_bin = np.floor(x_occ / occ_bin_width).astype(int)
    speed_bin = (x_speed // speed_bin_width).astype(int)

    groups = defaultdict(list)
    for i, key in enumerate(zip(flow_bin, occ_bin, speed_bin)):
        groups[(int(key[0]), int(key[1]), int(key[2]))].append(i)

    rng = np.random.default_rng(seed)
    pair_a, pair_b, group_mean = [], [], []

    for _, idxs in groups.items():
        if len(idxs) < 2:
            continue
        idxs = np.array(idxs, dtype=np.int64)
        k_pairs = min(max_pairs_per_group, len(idxs) * (len(idxs) - 1) // 2)
        a = rng.choice(idxs, size=k_pairs, replace=True)
        b = rng.choice(idxs, size=k_pairs, replace=True)
        m = a != b
        a = a[m]
        b = b[m]
        if len(a) == 0:
            continue
        pair_a.append(a)
        pair_b.append(b)
        group_mean.append(np.full(len(a), float(np.mean(y_next[idxs])), dtype=float))

    pair_a = np.concatenate(pair_a)
    pair_b = np.concatenate(pair_b)
    group_mean = np.concatenate(group_mean)

    delta_y = np.abs(y_next[pair_a] - y_next[pair_b])
    ratio = float(np.mean(delta_y >= rel_gap * group_mean) * 100.0)
    return ratio, len(pair_a)

def causal_equivalence_style_ratio(
    x_flow, x_occ, x_speed, y_next,
    eps_y=100.0, flow_diff=60.0, occ_diff=0.05, speed_diff=10.0,
    sample_size=50000, pair_count=300000, seed=0
):
    rng = np.random.default_rng(seed)
    n = len(y_next)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)

    xf = x_flow[sample_idx]
    xo = x_occ[sample_idx]
    xs = x_speed[sample_idx]
    yy = y_next[sample_idx]

    a = rng.integers(0, len(sample_idx), size=pair_count)
    b = rng.integers(0, len(sample_idx), size=pair_count)
    m = a != b
    a = a[m]
    b = b[m]

    similar_output = np.abs(yy[a] - yy[b]) <= eps_y
    distinct_input = (
        (np.abs(xf[a] - xf[b]) >= flow_diff) |
        (np.abs(xo[a] - xo[b]) >= occ_diff) |
        (np.abs(xs[a] - xs[b]) >= speed_diff)
    )

    ratio = float(np.mean(similar_output & distinct_input) * 100.0)
    return ratio, len(a)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"/mnt/data/pems08_unzip/PEMS08/data.npz")
    args = parser.parse_args()

    flow, occ, speed = load_pems08(args.data)
    x_flow, x_occ, x_speed, y_next = build_one_step_samples(flow, occ, speed)

    nd_ratio, nd_pairs = nondeterministic_mapping_ratio(x_flow, x_occ, x_speed, y_next)
    ce_ratio, ce_pairs = causal_equivalence_style_ratio(x_flow, x_occ, x_speed, y_next)

    result = {
        "data_shape": list(np.load(args.data)["data"].shape),
        "channel_interpretation": ["flow", "occupancy", "speed"],
        "non_deterministic_mapping_ratio_percent": nd_ratio,
        "non_deterministic_pairs": int(nd_pairs),
        "causal_equivalence_style_ratio_percent": ce_ratio,
        "causal_equivalence_pairs": int(ce_pairs)
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
