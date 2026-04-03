#!/usr/bin/env python3
"""
Regression model for predicting micro-batch forward/backward computation time
based on packed sequence characteristics (cu_seqlens distribution).

Key insight: FlashAttention computation cost is proportional to sum(seq_len_i^2)
rather than sum(seq_len_i). A single long sequence of 8192 tokens costs much more
than 8 short sequences of 1024 tokens each, even though total tokens are the same.

Features used:
  - sum_sq_seqlens: sum of squared sequence lengths (most theoretically grounded)
  - max_seq_len: maximum sequence length in the pack
  - num_sequences: number of sequences packed together
"""

import json
import numpy as np
import os
from collections import defaultdict


def extract_features_from_seq_info(seq_info_entry):
    """Extract regression features from a single micro-batch's seq_info."""
    all_lens = [l for s in seq_info_entry['seq_lengths'] for l in s]
    if not all_lens:
        return None
    return {
        'num_sequences': len(all_lens),
        'max_seq_len': max(all_lens),
        'min_seq_len': min(all_lens),
        'mean_seq_len': np.mean(all_lens),
        'std_seq_len': np.std(all_lens) if len(all_lens) > 1 else 0,
        'total_tokens': sum(all_lens),
        'sum_sq_seqlens': sum(l**2 for l in all_lens),
    }


def extract_features_from_cu_seqlens(cu_seqlens_raw):
    """Extract regression features directly from cu_seqlens tensor(s)."""
    seq_lengths = []
    if isinstance(cu_seqlens_raw, (list, tuple)):
        for cu in cu_seqlens_raw:
            if hasattr(cu, 'tolist'):
                cu_list = cu.tolist()
            elif hasattr(cu, '__len__'):
                cu_list = list(cu)
            else:
                continue
            if isinstance(cu_list, (int, float)):
                continue
            lengths = [cu_list[j+1] - cu_list[j] for j in range(len(cu_list)-1)]
            seq_lengths.extend(lengths)
    elif hasattr(cu_seqlens_raw, 'tolist'):
        cu_list = cu_seqlens_raw.tolist()
        if isinstance(cu_list, list) and len(cu_list) > 1:
            lengths = [cu_list[j+1] - cu_list[j] for j in range(len(cu_list)-1)]
            seq_lengths.extend(lengths)

    if not seq_lengths:
        return None

    return {
        'num_sequences': len(seq_lengths),
        'max_seq_len': max(seq_lengths),
        'min_seq_len': min(seq_lengths),
        'mean_seq_len': np.mean(seq_lengths),
        'std_seq_len': np.std(seq_lengths) if len(seq_lengths) > 1 else 0,
        'total_tokens': sum(seq_lengths),
        'sum_sq_seqlens': sum(l**2 for l in seq_lengths),
    }


class MicroBatchTimePredictor:
    """
    Predicts micro-batch forward/backward time based on packed sequence features.

    Uses a linear model: time = a * sum_sq_seqlens + b * max_seq_len + c

    The model can be:
    1. Fitted from profiling data (JSON files from profile_fwd_bwd=True)
    2. Used online during training to predict iteration time
    """

    def __init__(self):
        self.fwd_coeffs = None  # (a, b, c) for forward time
        self.bwd_coeffs = None  # (a, b, c) for backward time
        self.is_fitted = False
        self.data_points = []  # accumulated (features, fwd_time, bwd_time) tuples

    def add_data_point(self, features, fwd_time=None, bwd_time=None):
        """Add a single observation for online fitting."""
        self.data_points.append((features, fwd_time, bwd_time))

    def fit_from_profiling_json(self, json_paths, skip_first_n_iters=2, outlier_threshold=2.0):
        """
        Fit the model from profiling JSON files.

        Args:
            json_paths: list of paths to PP_rank_*.json files
            skip_first_n_iters: skip first N iterations (warmup effects)
            outlier_threshold: remove data points beyond this many std devs
        """
        all_X = []
        all_y_fwd = []
        all_y_bwd = []

        for path in json_paths:
            with open(path) as f:
                data = json.load(f)

            for iter_idx in range(skip_first_n_iters, data['num_iterations']):
                it = data['iterations'][iter_idx]
                fwd = it['fwd_times']
                bwd = it.get('bwd_times', [])
                seq = it['seq_info']

                for mb_idx in range(min(len(fwd), len(seq))):
                    features = extract_features_from_seq_info(seq[mb_idx])
                    if features is None:
                        continue
                    all_X.append([features['sum_sq_seqlens'], features['max_seq_len']])
                    all_y_fwd.append(fwd[mb_idx])

                # For backward, the mapping to micro-batches depends on the schedule
                # In ZBH1: bwd has fewer entries than fwd (no bwd for warmup micro-batches)
                for mb_idx in range(min(len(bwd), len(seq))):
                    features = extract_features_from_seq_info(seq[mb_idx])
                    if features is None:
                        continue
                    all_y_bwd.append(bwd[mb_idx])

        X = np.array(all_X)
        y_fwd = np.array(all_y_fwd)

        # Remove outliers
        fwd_mean, fwd_std = y_fwd.mean(), y_fwd.std()
        mask = np.abs(y_fwd - fwd_mean) < outlier_threshold * fwd_std
        X_clean = X[mask]
        y_fwd_clean = y_fwd[mask]

        # Fit forward model: fwd_time = a * sum_sq + b * max_len + c
        X_with_bias = np.column_stack([X_clean, np.ones(len(X_clean))])
        self.fwd_coeffs, _, _, _ = np.linalg.lstsq(X_with_bias, y_fwd_clean, rcond=None)

        # Fit backward model if data available
        if len(all_y_bwd) > 10:
            X_bwd = np.array(all_X[:len(all_y_bwd)])
            y_bwd = np.array(all_y_bwd)
            bwd_mean, bwd_std = y_bwd.mean(), y_bwd.std()
            mask_bwd = np.abs(y_bwd - bwd_mean) < outlier_threshold * bwd_std
            X_bwd_clean = X_bwd[mask_bwd]
            y_bwd_clean = y_bwd[mask_bwd]
            X_bwd_bias = np.column_stack([X_bwd_clean, np.ones(len(X_bwd_clean))])
            self.bwd_coeffs, _, _, _ = np.linalg.lstsq(X_bwd_bias, y_bwd_clean, rcond=None)

        self.is_fitted = True

        # Report accuracy
        y_pred = X_with_bias @ self.fwd_coeffs
        r2 = 1 - np.sum((y_fwd_clean - y_pred)**2) / np.sum((y_fwd_clean - y_fwd_clean.mean())**2)
        mae = np.mean(np.abs(y_fwd_clean - y_pred))
        mape = np.mean(np.abs((y_fwd_clean - y_pred) / y_fwd_clean)) * 100

        print(f"Model fitted on {len(y_fwd_clean)} data points (fwd)")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae:.6f}s")
        print(f"  MAPE = {mape:.2f}%")
        print(f"  Coefficients: sum_sq={self.fwd_coeffs[0]:.12e}, max_len={self.fwd_coeffs[1]:.8e}, intercept={self.fwd_coeffs[2]:.6f}")

        return {'r2': r2, 'mae': mae, 'mape': mape}

    def predict_fwd_time(self, features):
        """Predict forward time for a micro-batch given its sequence features."""
        if not self.is_fitted:
            return None
        x = np.array([features['sum_sq_seqlens'], features['max_seq_len'], 1.0])
        return float(x @ self.fwd_coeffs)

    def predict_bwd_time(self, features):
        """Predict backward time for a micro-batch given its sequence features."""
        if not self.is_fitted or self.bwd_coeffs is None:
            return None
        x = np.array([features['sum_sq_seqlens'], features['max_seq_len'], 1.0])
        return float(x @ self.bwd_coeffs)

    def predict_iteration_time(self, micro_batch_features_list):
        """
        Predict total forward+backward time for an iteration given all micro-batch features.

        Args:
            micro_batch_features_list: list of feature dicts, one per micro-batch

        Returns:
            dict with predicted fwd/bwd times per micro-batch and totals
        """
        fwd_times = []
        bwd_times = []
        for features in micro_batch_features_list:
            fwd_t = self.predict_fwd_time(features)
            bwd_t = self.predict_bwd_time(features)
            fwd_times.append(fwd_t)
            bwd_times.append(bwd_t)

        return {
            'fwd_times': fwd_times,
            'bwd_times': bwd_times,
            'total_fwd': sum(t for t in fwd_times if t is not None),
            'total_bwd': sum(t for t in bwd_times if t is not None),
            'max_fwd': max(t for t in fwd_times if t is not None) if fwd_times else None,
            'max_bwd': max(t for t in bwd_times if t is not None) if bwd_times else None,
        }

    def save(self, path):
        """Save model coefficients to JSON."""
        model_data = {
            'fwd_coeffs': self.fwd_coeffs.tolist() if self.fwd_coeffs is not None else None,
            'bwd_coeffs': self.bwd_coeffs.tolist() if self.bwd_coeffs is not None else None,
            'feature_names': ['sum_sq_seqlens', 'max_seq_len', 'bias'],
        }
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model coefficients from JSON."""
        with open(path) as f:
            model_data = json.load(f)
        self.fwd_coeffs = np.array(model_data['fwd_coeffs']) if model_data['fwd_coeffs'] else None
        self.bwd_coeffs = np.array(model_data['bwd_coeffs']) if model_data['bwd_coeffs'] else None
        self.is_fitted = True
        print(f"Model loaded from {path}")


def analyze_iteration_variance(json_paths, skip_first_n_iters=2):
    """
    Analyze how much of iteration time variance is explained by packing vs other factors.
    """
    print("=" * 70)
    print("ITERATION TIME VARIANCE ANALYSIS")
    print("=" * 70)

    for path in json_paths:
        with open(path) as f:
            data = json.load(f)

        rank_name = os.path.basename(path)
        print(f"\n--- {rank_name} ---")

        iter_fwd_totals = []
        iter_max_sum_sq = []
        iter_mean_sum_sq = []

        for iter_idx in range(skip_first_n_iters, data['num_iterations']):
            it = data['iterations'][iter_idx]
            fwd = it['fwd_times']
            seq = it['seq_info']

            iter_fwd_totals.append(sum(fwd))

            sum_sqs = []
            for si in seq:
                all_lens = [l for s in si['seq_lengths'] for l in s]
                sum_sqs.append(sum(l**2 for l in all_lens))

            iter_max_sum_sq.append(max(sum_sqs) if sum_sqs else 0)
            iter_mean_sum_sq.append(np.mean(sum_sqs) if sum_sqs else 0)

        fwd_arr = np.array(iter_fwd_totals)
        max_sq_arr = np.array(iter_max_sum_sq)
        mean_sq_arr = np.array(iter_mean_sum_sq)

        print(f"  Iteration fwd_total: mean={fwd_arr.mean():.4f}s, std={fwd_arr.std():.4f}s, cv={fwd_arr.std()/fwd_arr.mean()*100:.1f}%")
        print(f"  Corr(iter_fwd_total, max_sum_sq_across_mbs): {np.corrcoef(max_sq_arr, fwd_arr)[0,1]:.4f}")
        print(f"  Corr(iter_fwd_total, mean_sum_sq_across_mbs): {np.corrcoef(mean_sq_arr, fwd_arr)[0,1]:.4f}")

        # The bottleneck: max micro-batch time per iteration
        iter_max_fwd = []
        for iter_idx in range(skip_first_n_iters, data['num_iterations']):
            it = data['iterations'][iter_idx]
            iter_max_fwd.append(max(it['fwd_times']))

        max_fwd_arr = np.array(iter_max_fwd)
        print(f"  Max fwd per iter: mean={max_fwd_arr.mean():.4f}s, std={max_fwd_arr.std():.4f}s")
        print(f"  Corr(max_fwd_per_iter, max_sum_sq): {np.corrcoef(max_sq_arr, max_fwd_arr)[0,1]:.4f}")


if __name__ == "__main__":
    import sys

    # Default: fit from latest profiling data
    result_dir = "/mnt/shared-storage-user/ailab-sys/matenghui/InternEvo/results/fwd_bwd_time"

    if len(sys.argv) > 1:
        json_paths = sys.argv[1:]
    else:
        # Find latest results
        import glob
        json_paths = sorted(glob.glob(os.path.join(result_dir, "**", "PP_rank_*.json"), recursive=True))
        if not json_paths:
            print("No profiling data found. Run training with profile_fwd_bwd=True first.")
            sys.exit(1)
        # Get latest timestamp directory
        latest_dir = os.path.dirname(max(json_paths, key=os.path.getmtime))
        json_paths = sorted(glob.glob(os.path.join(latest_dir, "PP_rank_*.json")))

    print(f"Using profiling data from: {json_paths}")

    # Fit model
    predictor = MicroBatchTimePredictor()
    predictor.fit_from_profiling_json(json_paths)

    # Save model
    model_path = os.path.join(os.path.dirname(json_paths[0]), "regression_model.json")
    predictor.save(model_path)

    # Analyze variance
    analyze_iteration_variance(json_paths)

    # Demo predictions
    print("\n\n=== DEMO PREDICTIONS ===")
    test_cases = [
        {"desc": "1 long seq (8192)", "sum_sq_seqlens": 8192**2, "max_seq_len": 8192},
        {"desc": "2 seqs (4096 each)", "sum_sq_seqlens": 2 * 4096**2, "max_seq_len": 4096},
        {"desc": "8 seqs (1024 each)", "sum_sq_seqlens": 8 * 1024**2, "max_seq_len": 1024},
        {"desc": "4 seqs (2048 each)", "sum_sq_seqlens": 4 * 2048**2, "max_seq_len": 2048},
    ]
    for tc in test_cases:
        pred = predictor.predict_fwd_time(tc)
        print(f"  {tc['desc']:30s}: predicted fwd = {pred:.4f}s")
