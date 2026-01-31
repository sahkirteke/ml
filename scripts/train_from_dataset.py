#!/usr/bin/env python3
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
    parser.add_argument("--data-dir", default=Path("data"), type=Path, help="Root data directory")
    parser.add_argument("--out-dir", default=Path("models"), type=Path, help="Output models directory")
    parser.add_argument("--exclude-today", action="store_true", help="Exclude today's partition (Europe/Istanbul)")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols to include")
    parser.add_argument("--min-rows-per-symbol", default=20000, type=int, help="Minimum rows required to train")
    parser.add_argument("--logreg-c", default=0.3, type=float, help="LogisticRegression C value")
    return parser.parse_args()


def find_jsonl_files(
    root: Path,
    *,
    exclude_today: bool,
) -> list[Path]:
    if not root.exists():
        return []
    pattern = re.compile(r"-(\d{8})\.jsonl\.gz$")
    today_ymd = None
    if exclude_today:
        tz = resolve_istanbul_tz()
        today_ymd = datetime.now(tz=tz).strftime("%Y%m%d")
    paths: list[Path] = []
    for path in root.glob("**/*.jsonl.gz"):
        match = pattern.search(path.name)
        if not match:
            continue
        ymd = match.group(1)
        if exclude_today and ymd == today_ymd:
            continue
        paths.append(path)
    return sorted(paths)


def read_jsonl_gz(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, compression="gzip")


def extract_dates(paths: Iterable[Path]) -> set[str]:
    dates = set()
    pattern = re.compile(r"-(\d{8})\.jsonl\.gz$")
    for path in paths:
        match = pattern.search(path.name)
        if match:
            dates.add(match.group(1))
    return dates


def resolve_istanbul_tz():
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        try:
            from backports.zoneinfo import ZoneInfo
        except ImportError as exc:
            raise RuntimeError(
                "ZoneInfo is required for --exclude-today; install backports.zoneinfo for Python < 3.9."
            ) from exc
    return ZoneInfo("Europe/Istanbul")


def load_features_labels(
    data_dir: Path,
    symbol: str,
    *,
    exclude_today: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path], list[Path]]:
    features_root = data_dir / "features" / symbol
    labels_root = data_dir / "labels" / symbol
    feature_files = find_jsonl_files(
        features_root,
        exclude_today=exclude_today,
    )
    label_files = find_jsonl_files(
        labels_root,
        exclude_today=exclude_today,
    )
    if not feature_files or not label_files:
        return pd.DataFrame(), pd.DataFrame(), feature_files, label_files
    feature_frames = [read_jsonl_gz(path) for path in feature_files]
    label_frames = [read_jsonl_gz(path) for path in label_files]
    features = pd.concat(feature_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)
    return features, labels, feature_files, label_files


def build_training_frame(
    features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    features_filtered = features[(features["windowReady"] == True) & (features["featuresVersion"] == FEATURES_VERSION)]
    labels_filtered = labels[labels["labelType"] == LABEL_TYPE]
    merged = features_filtered.merge(
        labels_filtered,
        on=["symbol", "tf", "closeTimeMs"],
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
    return x, y, merged


def train_model(x: pd.DataFrame, y: pd.Series, *, c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=2000,
        n_jobs=None,
        class_weight="balanced",
        C=c_value,
        solver="lbfgs",
    )
    model.fit(x, y)
    return model


def export_onnx(model, feature_count: int, output_path: Path) -> None:
    initial_type = [("input", FloatTensorType([None, feature_count]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())


def write_meta(
    output_path: Path,
    model_version: str,
    symbol: str,
    train_days: int,
    train_rows: int,
    mean_ret_up: float | None,
    mean_ret_down: float | None,
    n_up: int,
    n_down: int,
    up_rate: float,
) -> None:
    meta = {
        "modelVersion": model_version,
        "symbol": symbol,
        "featuresVersion": FEATURES_VERSION,
        "featureOrder": FEATURE_ORDER,
        "imputeStrategy": "zero",
        "rows": train_rows,
        "days": train_days,
        "trainRows": train_rows,
        "trainDays": train_days,
        "meanRetUp": mean_ret_up,
        "meanRetDown": mean_ret_down,
        "nUp": n_up,
        "nDown": n_down,
        "upRate": up_rate,
        "decisionPolicy": {
            "minConfidence": 0.55,
            "minAbsExpectedPct": 0.05,
            "mode": "FILTERED",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_current(model_dir: Path, out_dir: Path, symbol: str) -> None:
    current_dir = out_dir / symbol / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    model_src = model_dir / "model.onnx"
    meta_src = model_dir / "model_meta.json"
    model_tmp = current_dir / "model.onnx.tmp"
    meta_tmp = current_dir / "model_meta.json.tmp"
    model_dst = current_dir / "model.onnx"
    meta_dst = current_dir / "model_meta.json"
    model_tmp.write_bytes(model_src.read_bytes())
    meta_tmp.write_bytes(meta_src.read_bytes())
    model_tmp.replace(model_dst)
    meta_tmp.replace(meta_dst)


def main() -> None:
    args = parse_args()
    if args.symbols:
        symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()]
    else:
        features_root = args.data_dir / "features"
        if not features_root.exists():
            raise RuntimeError(f"No features directory found at {features_root}")
        symbols = sorted([path.name.upper() for path in features_root.iterdir() if path.is_dir()])
    for symbol in symbols:
        features, labels, feature_files, label_files = load_features_labels(
            args.data_dir,
            symbol,
            exclude_today=args.exclude_today,
        )
        if features.empty or labels.empty:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows=0 min={args.min_rows_per_symbol}")
            continue
        x, y, merged = build_training_frame(features, labels)
        if len(x) < args.min_rows_per_symbol:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows={len(x)} min={args.min_rows_per_symbol}")
            continue
        up_mask = merged["labelUp"] == 1
        down_mask = merged["labelUp"] == 0
        mean_ret_up = None
        mean_ret_down = None
        if up_mask.any():
            mean_ret_up = float(merged.loc[up_mask, "futureRet_1"].mean())
        if down_mask.any():
            mean_ret_down = float(merged.loc[down_mask, "futureRet_1"].mean())
        n_up = int(up_mask.sum())
        n_down = int(down_mask.sum())
        total_labels = n_up + n_down
        up_rate = float(n_up / total_labels) if total_labels > 0 else 0.0
        model = train_model(x, y, c_value=args.logreg_c)
        calibrated = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv=3)
        calibrated.fit(x, y)
        model_version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        model_dir = args.out_dir / symbol / model_version
        export_model = calibrated
        try:
            export_onnx(export_model, x.shape[1], model_dir / "model.onnx")
        except Exception as exc:
            print(f"ONNX_EXPORT_FAILED symbol={symbol} calibrated failed ({exc}); falling back to LogisticRegression.")
            export_model = model
            try:
                export_onnx(export_model, x.shape[1], model_dir / "model.onnx")
            except Exception as logreg_exc:
                raise RuntimeError(
                    f"ONNX export failed for symbol {symbol}: calibrated and LogisticRegression failed: {logreg_exc}"
                ) from logreg_exc
        train_days = len(extract_dates(feature_files))
        write_meta(
            model_dir / "model_meta.json",
            model_version,
            symbol,
            train_days,
            len(x),
            mean_ret_up,
            mean_ret_down,
            n_up,
            n_down,
            up_rate,
        )
        write_current(model_dir, args.out_dir, symbol)
        wrote_path = args.out_dir / symbol / "current"
        print(f"TRAIN_SYMBOL symbol={symbol} rows={len(x)} upRate={up_rate:.6f} wrote={wrote_path}")


if __name__ == "__main__":
    main()
