#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    parser.add_argument("--calibrate", action="store_true", help="Enable probability calibration")
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
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
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
    x = x.apply(pd.to_numeric, errors="coerce")
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x = x.astype(np.float32)
    y = merged["labelUp"].astype(int)
    stds = x.std(axis=0, skipna=True)
    keep_cols = [col for col in x.columns if stds[col] > 0]
    if keep_cols and len(keep_cols) != len(x.columns):
        x = x[keep_cols].copy()
    return x, y, merged, list(x.columns)


def build_pipeline(solver: str) -> Pipeline:
    scaler = StandardScaler()
    base_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", scaler),
    ]
    if solver == "saga":
        lr = LogisticRegression(solver="saga", max_iter=4000, tol=1e-4, n_jobs=-1)
    else:
        lr = LogisticRegression(solver="lbfgs", max_iter=4000, tol=1e-4)
    return Pipeline(base_steps + [("classifier", lr)])


def export_onnx(model, feature_count: int, output_path: Path) -> tuple[list[str], list[str]]:
    initial_type = [("input", FloatTensorType([None, feature_count]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={id(model): {"zipmap": False}},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())
    input_names = [node.name for node in onnx_model.graph.input]
    output_names = [node.name for node in onnx_model.graph.output]
    return input_names, output_names


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
    classes: list[int],
    up_class_index: int,
    feature_order: list[str],
    onnx_outputs: list[str],
    prob_output_name: str | None,
) -> None:
    def _json_default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    meta = {
        "modelVersion": model_version,
        "symbol": symbol,
        "featuresVersion": FEATURES_VERSION,
        "featureOrder": feature_order,
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
        "classes": classes,
        "upClass": 1,
        "upClassIndex": up_class_index,
        "onnxOutputs": onnx_outputs,
        "probOutputName": prob_output_name,
        "decisionPolicy": {
            "minConfidence": 0.55,
            "minAbsExpectedPct": 0.05,
            "mode": "FILTERED",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


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
        x, y, merged, feature_order = build_training_frame(features, labels)
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
        base_pipeline = build_pipeline("lbfgs")
        print(
            "MODEL_CONFIG symbol={} solver=lbfgs max_iter=4000 scaler=StandardScaler".format(symbol)
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            base_pipeline.fit(x, y)
        has_convergence_warning = any(
            isinstance(warning.message, ConvergenceWarning) for warning in caught
        )
        if has_convergence_warning:
            print("WARN_CONVERGENCE symbol={} solver=lbfgs fallback_solver=saga".format(symbol))
            base_pipeline = build_pipeline("saga")
            print(
                "MODEL_CONFIG symbol={} solver=saga max_iter=4000 scaler=StandardScaler".format(symbol)
            )
            base_pipeline.fit(x, y)
        calibrated = None
        if args.calibrate:
            tscv = TimeSeriesSplit(n_splits=3)
            calibrated = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=tscv)
            calibrated.fit(x, y)
        model_for_stats = calibrated if calibrated is not None else base_pipeline
        classifier = base_pipeline.named_steps["classifier"]
        classes = list(getattr(classifier, "classes_", []))
        if not classes:
            raise RuntimeError(f"No classes_ found for symbol {symbol}")
        if 1 not in classes:
            raise RuntimeError(f"UP class (1) missing in classes for symbol {symbol}: {classes}")
        up_class_index = classes.index(1)
        p_up = model_for_stats.predict_proba(x)[:, up_class_index]
        p_up_min = float(np.min(p_up))
        p_up_max = float(np.max(p_up))
        p_up_mean = float(np.mean(p_up))
        print(f"PUP_STATS symbol={symbol} min={p_up_min:.6f} max={p_up_max:.6f} mean={p_up_mean:.6f}")
        if p_up_max < 0.05 or p_up_mean < 0.01:
            print(
                "WARN_PUP_DEGENERATE symbol={} max={:.6f} mean={:.6f} "
                "reason=class_index_or_features_or_nan_issue".format(symbol, p_up_max, p_up_mean)
            )
        model_version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        model_dir = args.out_dir / symbol / model_version
        export_model = calibrated if calibrated is not None else base_pipeline
        try:
            input_names, output_names = export_onnx(export_model, x.shape[1], model_dir / "model.onnx")
            print(f"ONNX_EXPORT symbol={symbol} inputs={input_names} outputs={output_names}")
        except Exception as exc:
            print(f"ONNX_EXPORT_FAILED symbol={symbol} calibrated failed ({exc}); fallback: export non-calibrated.")
            export_model = base_pipeline
            input_names, output_names = export_onnx(export_model, x.shape[1], model_dir / "model.onnx")
            print(f"ONNX_EXPORT symbol={symbol} inputs={input_names} outputs={output_names}")
        prob_output_name = "probabilities" if "probabilities" in output_names else None
        if export_model is base_pipeline:
            onnx_checked = False
        else:
            onnx_checked = True
        if onnx_checked:
            try:
                import onnxruntime as ort

                sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
                input_name = sess.get_inputs()[0].name
                x_check = x.iloc[:256].to_numpy(dtype=np.float32)
                outputs = sess.run(None, {input_name: x_check})
                if outputs:
                    onnx_probs = np.array(outputs[0])
                    if onnx_probs.ndim >= 2 and onnx_probs.shape[1] > up_class_index:
                        p_up_onnx = onnx_probs[:, up_class_index]
                        mean_onnx = float(np.mean(p_up_onnx))
                        print(
                            f"ONNX_CHECK symbol={symbol} meanSklearn={p_up_mean:.6f} meanOnnx={mean_onnx:.6f}"
                        )
                        if abs(p_up_mean - mean_onnx) > 1e-2:
                            print(f"WARN_ONNX_MISMATCH symbol={symbol} diff={abs(p_up_mean - mean_onnx):.6f}")
                    else:
                        print(f"WARN_ONNX_MISMATCH symbol={symbol} reason=unexpected_output_shape")
                else:
                    print(f"WARN_ONNX_MISMATCH symbol={symbol} reason=no_outputs")
            except Exception as exc:
                print(f"WARN_ONNX_MISMATCH symbol={symbol} error={exc}")
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
            classes,
            up_class_index,
            feature_order,
            output_names,
            prob_output_name,
        )
        write_current(model_dir, args.out_dir, symbol)
        wrote_path = args.out_dir / symbol / "current"
        print(
            "TRAIN_SYMBOL symbol={} rows={} upRate={:.6f} calibrated={} wrote={}".format(
                symbol,
                len(x),
                up_rate,
                args.calibrate,
                wrote_path,
            )
        )


if __name__ == "__main__":
    main()
