#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ONNX outputs for a model.")
    parser.add_argument("--model-path", required=True, type=Path, help="Path to model.onnx")
    parser.add_argument("--meta-path", required=True, type=Path, help="Path to model_meta.json")
    parser.add_argument("--features-path", default=None, type=Path, help="Optional features jsonl(.gz)")
    parser.add_argument("--max-rows", default=256, type=int, help="Max rows to evaluate")
    return parser.parse_args()


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(path: Path, feature_order: list[str], max_rows: int) -> np.ndarray:
    import pandas as pd

    df = pd.read_json(path, lines=True, compression="gzip" if path.suffix == ".gz" else None)
    if df.empty:
        raise RuntimeError(f"No rows found in features file {path}")
    missing = [col for col in feature_order if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in {path}: {missing}")
    x = df[feature_order].head(max_rows).copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x = x.fillna(0.0)
    return x.to_numpy(dtype=np.float32)


def resolve_prob_output(meta: dict, output_names: list[str]) -> str:
    prob_name = meta.get("probOutputName")
    if prob_name and prob_name in output_names:
        return prob_name
    if "probabilities" in output_names:
        return "probabilities"
    return output_names[0]


def main() -> None:
    args = parse_args()
    meta = load_meta(args.meta_path)
    feature_order = meta.get("featureOrder") or []
    if not feature_order:
        raise RuntimeError("featureOrder missing in meta")
    up_class_index = int(meta.get("upClassIndex", 1))
    up_rate = meta.get("upRate")
    if args.features_path:
        x = load_features(args.features_path, feature_order, args.max_rows)
    else:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(min(args.max_rows, 256), len(feature_order))).astype(np.float32)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.model_path), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in sess.get_outputs()]
    prob_name = resolve_prob_output(meta, output_names)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run([prob_name], {input_name: x})
    if not outputs:
        raise RuntimeError("No outputs returned from ONNX session")
    probs = np.asarray(outputs[0])
    if probs.ndim < 2 or probs.shape[1] <= up_class_index:
        raise RuntimeError(f"Unexpected probabilities shape {probs.shape} for upClassIndex {up_class_index}")
    p_up = probs[:, up_class_index]
    p_min = float(np.min(p_up))
    p_max = float(np.max(p_up))
    p_mean = float(np.mean(p_up))
    print(f"PUP_STATS min={p_min:.6f} max={p_max:.6f} mean={p_mean:.6f} probOutput={prob_name}")
    if up_rate is not None:
        try:
            up_rate_val = float(up_rate)
            diff = abs(p_mean - up_rate_val)
            if diff > 0.2:
                print(f"WARN_PUP_MEAN_DRIFT mean={p_mean:.6f} upRate={up_rate_val:.6f} diff={diff:.6f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
