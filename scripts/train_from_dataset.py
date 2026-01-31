#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

FEATURE_ORDER = [
    "ret_1",
    "logRet_1",
    "ret_3",
    "ret_12",
    "realizedVol_6",
    "realizedVol_24",
    "rangePct",
    "bodyPct",
    "upperWickPct",
    "lowerWickPct",
    "closePos",
    "volRatio_12",
    "tradeRatio_12",
    "buySellRatio",
    "deltaVolNorm",
    "rsi14",
    "atr14",
    "ema20DistPct",
    "ema200DistPct",
]

FEATURES_VERSION = "ftr_5m_v1"
LABEL_TYPE = "next_close_direction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model from dataset JSONL.gz files.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Root data directory")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output models directory")
    parser.add_argument("--model-version", default=None, help="Optional model version")
    return parser.parse_args()


def find_jsonl_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/*.jsonl.gz"))


def read_jsonl_gz(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, compression="gzip")


def extract_dates(paths: list[Path]) -> set[str]:
    dates = set()
    pattern = re.compile(r"-(\d{8})\.jsonl\.gz$")
    for path in paths:
        match = pattern.search(path.name)
        if match:
            dates.add(match.group(1))
    return dates


def load_features_labels(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    features_root = data_dir / "features"
    labels_root = data_dir / "labels"
    feature_files = find_jsonl_files(features_root)
    label_files = find_jsonl_files(labels_root)
    if not feature_files:
        raise RuntimeError(f"No feature files found under {features_root}")
    if not label_files:
        raise RuntimeError(f"No label files found under {labels_root}")
    feature_frames = [read_jsonl_gz(path) for path in feature_files]
    label_frames = [read_jsonl_gz(path) for path in label_files]
    features = pd.concat(feature_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)
    train_days = len(extract_dates(feature_files))
    return features, labels, train_days


def build_training_frame(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features_filtered = features[(features["windowReady"] == True) & (features["featuresVersion"] == FEATURES_VERSION)]
    labels_filtered = labels[labels["labelType"] == LABEL_TYPE]
    merged = features_filtered.merge(
        labels_filtered,
        on=["symbol", "closeTimeMs"],
        how="inner",
        suffixes=("_feat", "_label"),
    )
    if merged.empty:
        raise RuntimeError("No rows available after joining features and labels")
    missing_features = [col for col in FEATURE_ORDER if col not in merged.columns]
    if missing_features:
        raise RuntimeError(f"Missing expected feature columns: {missing_features}")
    x = merged[FEATURE_ORDER].copy()
    x = x.fillna(0.0)
    x = x.astype(float)
    y = merged["labelUp"].astype(int)
    return x, y


def train_model(x: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=200, n_jobs=None)
    model.fit(x, y)
    return model


def export_onnx(model: LogisticRegression, feature_count: int, output_path: Path) -> None:
    initial_type = [("input", FloatTensorType([None, feature_count]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())


def write_meta(
    output_path: Path,
    model_version: str,
    train_days: int,
    train_rows: int,
    trained_at_ms: int,
) -> None:
    meta = {
        "modelVersion": model_version,
        "featuresVersion": FEATURES_VERSION,
        "featureOrder": FEATURE_ORDER,
        "imputeStrategy": "zero",
        "trainedAtMs": trained_at_ms,
        "trainDays": train_days,
        "trainRows": train_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def copy_current(model_dir: Path, out_dir: Path) -> None:
    current_dir = out_dir / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / "model.onnx", current_dir / "model.onnx")
    shutil.copy2(model_dir / "model_meta.json", current_dir / "model_meta.json")


def main() -> None:
    args = parse_args()
    features, labels, train_days = load_features_labels(args.data_dir)
    x, y = build_training_frame(features, labels)
    model = train_model(x, y)
    trained_at_ms = int(time.time() * 1000)
    model_version = args.model_version or time.strftime("%Y%m%d%H%M%S", time.gmtime())
    model_dir = args.out_dir / model_version
    export_onnx(model, x.shape[1], model_dir / "model.onnx")
    write_meta(model_dir / "model_meta.json", model_version, train_days, len(x), trained_at_ms)
    copy_current(model_dir, args.out_dir)
    print(f"Trained model version {model_version} with {len(x)} rows.")


if __name__ == "__main__":
    main()
