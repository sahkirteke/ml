#!/usr/bin/env python3
from __future__ import annotations
import argparse
import itertools
import json
import math
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
MAX_HORIZON_BARS = 7
TP_PCT = 0.004
SL_PCT = 0.002
GAP_MS = 300000
CONF_THRESHOLD = 0.55
TRAIN_LIMIT = 400000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model from dataset JSONL.gz files.")
    parser.add_argument("--data-dir", default=Path("data"), type=Path, help="Root data directory")
    parser.add_argument("--out-dir", default=Path("models"), type=Path, help="Output models directory")
    parser.add_argument("--exclude-today", action="store_true", help="Exclude today's partition (Europe/Istanbul)")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols to include")
    parser.add_argument("--min-rows-per-symbol", default=20000, type=int, help="Minimum rows required to train")
    parser.add_argument("--train-rows", default=None, type=int, help="Rows to use for training (most recent)")
    parser.add_argument("--test-rows", default=100000, type=int, help="Rows to hold out for evaluation")
    parser.add_argument("--val-rows", default=50000, type=int, help="Rows to hold out for validation")
    parser.add_argument("--auto-tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--max-trials", default=80, type=int, help="Maximum tuning trials")
    parser.add_argument("--target-acc", default=0.65, type=float, help="Target accHi to stop tuning")
    parser.add_argument("--min-coverage", default=0.0002, type=float, help="Minimum coverage to stop tuning")
    parser.add_argument("--conf-threshold", default=CONF_THRESHOLD, type=float, help="Confidence threshold")
    parser.add_argument("--min-orders-per-day", default=7.0, type=float, help="Minimum orders per day")
    parser.add_argument("--min-total-orders", default=500, type=int, help="Minimum total orders in validation")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (skip export)")
    parser.add_argument("--fast-tail", action="store_true", help="Read only the latest rows needed")
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


def load_feature_frames(
    data_dir: Path,
    symbol: str,
    *,
    exclude_today: bool,
    fast_tail: bool,
    need_rows: int | None,
) -> tuple[pd.DataFrame, list[Path]]:
    features_root = data_dir / "features" / symbol
    feature_files = find_jsonl_files(
        features_root,
        exclude_today=exclude_today,
    )
    if not feature_files:
        return pd.DataFrame(), feature_files
    if not fast_tail or need_rows is None:
        feature_frames = [read_jsonl_gz(path) for path in feature_files]
        return pd.concat(feature_frames, ignore_index=True), feature_files
    need_days = int(math.ceil(need_rows / 288))
    selected_files = feature_files[-need_days:]
    print(
        "FAST_TAIL symbol={} needRows={} needDays={} files={}".format(
            symbol, need_rows, need_days, len(selected_files)
        )
    )
    feature_frames = [read_jsonl_gz(path) for path in selected_files]
    return pd.concat(feature_frames, ignore_index=True), selected_files


def load_raw_frames(
    data_dir: Path,
    symbol: str,
    *,
    exclude_today: bool,
    fast_tail: bool,
    need_rows: int | None,
) -> tuple[pd.DataFrame, list[Path]]:
    raw_root = data_dir / "raw" / symbol
    if not raw_root.exists():
        raw_root = data_dir / "raw" / "features" / symbol
    raw_files = find_jsonl_files(
        raw_root,
        exclude_today=exclude_today,
    )
    if not raw_files:
        return pd.DataFrame(), raw_files
    if not fast_tail or need_rows is None:
        raw_frames = [read_jsonl_gz(path) for path in raw_files]
        raw = pd.concat(raw_frames, ignore_index=True)
        return raw, raw_files
    need_days = int(math.ceil(need_rows / 288))
    selected_files = raw_files[-need_days:]
    print(
        "FAST_TAIL_RAW symbol={} needRows={} needDays={} files={}".format(
            symbol, need_rows, need_days, len(selected_files)
        )
    )
    raw_frames = [read_jsonl_gz(path) for path in selected_files]
    raw = pd.concat(raw_frames, ignore_index=True)
    return raw, selected_files


def outcome_long(
    close_time: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    idx: int,
) -> tuple[int, int, str] | None:
    entry = close[idx]
    if np.isnan(entry):
        return None
    tp_price = entry * (1.0 + TP_PCT)
    sl_price = entry * (1.0 - SL_PCT)
    for k in range(1, MAX_HORIZON_BARS + 1):
        if close_time[idx + k] - close_time[idx + k - 1] != GAP_MS:
            return None
        hi = high[idx + k]
        lo = low[idx + k]
        if np.isnan(hi) or np.isnan(lo):
            return None
        hit_tp = hi >= tp_price
        hit_sl = lo <= sl_price
        if hit_tp and hit_sl:
            return 0, k, "SL_HIT_SAME_BAR"
        if hit_sl:
            return 0, k, "SL_HIT"
        if hit_tp:
            return 1, k, "TP_HIT"
    return None


def outcome_short(
    close_time: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    idx: int,
) -> tuple[int, int, str] | None:
    entry = close[idx]
    if np.isnan(entry):
        return None
    tp_price = entry * (1.0 - TP_PCT)
    sl_price = entry * (1.0 + SL_PCT)
    for k in range(1, MAX_HORIZON_BARS + 1):
        if close_time[idx + k] - close_time[idx + k - 1] != GAP_MS:
            return None
        hi = high[idx + k]
        lo = low[idx + k]
        if np.isnan(hi) or np.isnan(lo):
            return None
        hit_tp = lo <= tp_price
        hit_sl = hi >= sl_price
        if hit_tp and hit_sl:
            return 0, k, "SL_HIT_SAME_BAR"
        if hit_sl:
            return 0, k, "SL_HIT"
        if hit_tp:
            return 1, k, "TP_HIT"
    return None


def build_labels_from_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = ["closeTimeMs", "highPrice", "lowPrice", "closePrice"]
    missing = [col for col in required_columns if col not in raw.columns]
    if missing:
        raise RuntimeError(f"Raw data missing columns: {missing}")
    raw_sorted = raw.sort_values("closeTimeMs").reset_index(drop=True)
    close_time = pd.to_numeric(raw_sorted["closeTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(raw_sorted["highPrice"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(raw_sorted["lowPrice"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(raw_sorted["closePrice"], errors="coerce").to_numpy(dtype=np.float64)
    long_records: list[dict[str, object]] = []
    short_records: list[dict[str, object]] = []
    for idx in range(len(raw_sorted) - MAX_HORIZON_BARS):
        if np.isnan(close_time[idx]):
            continue
        long_outcome = outcome_long(close_time, close, high, low, idx)
        short_outcome = outcome_short(close_time, close, high, low, idx)
        if long_outcome is None and short_outcome is None:
            continue
        base_record: dict[str, object] = {
            "closeTimeMs": int(close_time[idx]),
            "tpPct": TP_PCT,
            "slPct": SL_PCT,
            "maxHorizonBars": MAX_HORIZON_BARS,
        }
        if "symbol" in raw_sorted.columns:
            base_record["symbol"] = raw_sorted.at[idx, "symbol"]
        if "tf" in raw_sorted.columns:
            base_record["tf"] = raw_sorted.at[idx, "tf"]
        if long_outcome is not None:
            label_hit, time_to_event, event = long_outcome
            record = dict(base_record)
            record.update(
                {
                    "labelType": "tp0_004_sl0_002_within_7_event_driven_LONG",
                    "labelHit": int(label_hit),
                    "timeToEvent": int(time_to_event),
                    "event": event,
                }
            )
            long_records.append(record)
        if short_outcome is not None:
            label_hit, time_to_event, event = short_outcome
            record = dict(base_record)
            record.update(
                {
                    "labelType": "tp0_004_sl0_002_within_7_event_driven_SHORT",
                    "labelHit": int(label_hit),
                    "timeToEvent": int(time_to_event),
                    "event": event,
                }
            )
            short_records.append(record)
    return pd.DataFrame(long_records), pd.DataFrame(short_records)


def build_training_frames(
    features: pd.DataFrame, raw: pd.DataFrame
) -> tuple[
    tuple[pd.DataFrame, pd.Series, np.ndarray],
    tuple[pd.DataFrame, pd.Series, np.ndarray],
    list[str],
]:
    features_filtered = features[(features["windowReady"] == True) & (features["featuresVersion"] == FEATURES_VERSION)]
    if features_filtered.empty:
        raise RuntimeError("No rows available after filtering features")
    if raw.empty:
        raise RuntimeError("Raw data required for label generation is empty")
    long_labels, short_labels = build_labels_from_raw(raw)
    if long_labels.empty and short_labels.empty:
        raise RuntimeError("No rows available after labeling")
    def ensure_symbol_tf(labels: pd.DataFrame) -> pd.DataFrame:
        if labels.empty:
            return labels
        if "symbol" not in labels.columns:
            if "symbol" in features_filtered.columns and features_filtered["symbol"].nunique() == 1:
                labels = labels.copy()
                labels["symbol"] = features_filtered["symbol"].iloc[0]
            else:
                raise RuntimeError("Label data missing symbol for join")
        if "tf" not in labels.columns:
            if "tf" in features_filtered.columns and features_filtered["tf"].nunique() == 1:
                labels = labels.copy()
                labels["tf"] = features_filtered["tf"].iloc[0]
            else:
                raise RuntimeError("Label data missing tf for join")
        return labels

    long_labels = ensure_symbol_tf(long_labels)
    short_labels = ensure_symbol_tf(short_labels)
    merged_long = features_filtered.merge(
        long_labels,
        on=["symbol", "tf", "closeTimeMs"],
        how="inner",
    ).sort_values("closeTimeMs").reset_index(drop=True)
    merged_short = features_filtered.merge(
        short_labels,
        on=["symbol", "tf", "closeTimeMs"],
        how="inner",
    ).sort_values("closeTimeMs").reset_index(drop=True)
    if merged_long.empty and merged_short.empty:
        raise RuntimeError("No rows available after joining features and labels")
    missing_features = [col for col in FEATURE_ORDER if col not in features_filtered.columns]
    if missing_features:
        raise RuntimeError(f"Missing expected feature columns: {missing_features}")

    def build_xy(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        if merged.empty:
            return pd.DataFrame(), pd.Series(dtype=int), np.array([])
        x = merged[FEATURE_ORDER].copy()
        x = x.apply(pd.to_numeric, errors="coerce")
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x = x.astype(np.float32)
        stds = x.std(axis=0, skipna=True)
        keep_cols = [col for col in x.columns if stds[col] > 0]
        if keep_cols and len(keep_cols) != len(x.columns):
            x = x[keep_cols].copy()
        y = merged["labelHit"].astype(int)
        close_times = pd.to_numeric(merged["closeTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
        return x, y, close_times

    x_long, y_long, close_long = build_xy(merged_long)
    x_short, y_short, close_short = build_xy(merged_short)
    feature_order = list(x_long.columns if not x_long.empty else x_short.columns)
    return (x_long, y_long, close_long), (x_short, y_short, close_short), feature_order


def build_lr_pipeline(
    solver: str,
    *,
    c_value: float = 1.0,
    class_weight: str | None = None,
    max_iter: int = 4000,
    tol: float = 1e-4,
) -> Pipeline:
    scaler = StandardScaler()
    base_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", scaler),
    ]
    if solver == "saga":
        lr = LogisticRegression(
            solver="saga",
            max_iter=max_iter,
            tol=tol,
            n_jobs=-1,
            C=c_value,
            class_weight=class_weight,
        )
    else:
        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            tol=tol,
            C=c_value,
            class_weight=class_weight,
        )
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
    classes: list[int],
    up_class_index: int,
    feature_order: list[str],
    onnx_outputs: list[str],
    prob_output_name: str | None,
    label_type: str,
    best_params: dict[str, object],
    val_acc_hi: float,
    val_coverage: float,
    val_orders: int,
    val_orders_per_day: float,
    val_days: int,
    val_rows: int,
    test_acc_hi: float,
    test_coverage: float,
    test_orders: int,
    test_orders_per_day: float,
    test_days: int,
    test_rows: int,
    export_fallback: bool,
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
        "labelType": label_type,
        "tpPct": TP_PCT,
        "slPct": SL_PCT,
        "maxHorizonBars": MAX_HORIZON_BARS,
        "classes": classes,
        "upClassIndex": up_class_index,
        "bestParams": best_params,
        "valAccHi": val_acc_hi,
        "valCoverage": val_coverage,
        "valOrders": val_orders,
        "valOrdersPerDay": val_orders_per_day,
        "valDaysSpan": val_days,
        "valRows": val_rows,
        "testAccHi": test_acc_hi,
        "testCoverage": test_coverage,
        "testOrders": test_orders,
        "testOrdersPerDay": test_orders_per_day,
        "testDaysSpan": test_days,
        "testRows": test_rows,
        "exportFallback": export_fallback,
        "onnxOutputs": onnx_outputs,
        "probOutputName": prob_output_name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def write_current(model_dir: Path, out_dir: Path, symbol: str, variant: str) -> None:
    current_dir = out_dir / symbol / f"current_{variant}"
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


def check_onnx_outputs(symbol: str, model_path: Path, x_check: np.ndarray) -> None:
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: x_check})
        if not outputs:
            print("WARN_ONNX_MISMATCH symbol={} probShape={}".format(symbol, None))
            return
        prob_shape = None
        for output in outputs:
            arr = np.array(output)
            if arr.ndim == 1 and arr.shape[0] == 2:
                prob_shape = arr.shape
                break
            if arr.ndim == 2 and arr.shape[1] == 2:
                prob_shape = arr.shape
                break
        if prob_shape is None:
            output_shapes = [np.array(output).shape for output in outputs]
            print("WARN_ONNX_MISMATCH symbol={} probShape={}".format(symbol, output_shapes))
    except Exception as exc:
        print(f"WARN_ONNX_MISMATCH symbol={symbol} error={exc}")


def compute_confidence_metrics(
    y_true: pd.Series,
    preds: np.ndarray,
    p_hit: np.ndarray,
    *,
    conf_threshold: float,
    days_span: int,
) -> tuple[float, float, int, float]:
    confidence = np.maximum(p_hit, 1.0 - p_hit)
    trade_mask = confidence >= conf_threshold
    orders = int(np.sum(trade_mask))
    coverage = float(orders / len(confidence)) if len(confidence) else 0.0
    orders_per_day = float(orders / max(days_span, 1))
    if not np.any(trade_mask):
        return float("nan"), coverage, orders, orders_per_day
    acc_hi = float(accuracy_score(y_true[trade_mask], preds[trade_mask]))
    return acc_hi, coverage, orders, orders_per_day


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
        max_train_rows = args.train_rows or TRAIN_LIMIT
        need_rows = max_train_rows + args.val_rows + args.test_rows + MAX_HORIZON_BARS + 1
        features, feature_files = load_feature_frames(
            args.data_dir,
            symbol,
            exclude_today=args.exclude_today,
            fast_tail=args.fast_tail,
            need_rows=need_rows,
        )
        if features.empty:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows=0 min={args.min_rows_per_symbol}")
            continue
        raw, _ = load_raw_frames(
            args.data_dir,
            symbol,
            exclude_today=args.exclude_today,
            fast_tail=args.fast_tail,
            need_rows=need_rows,
        )
        (long_data, short_data, feature_order) = build_training_frames(features, raw)
        side_configs = [
            ("long", long_data, "tp0_004_sl0_002_within_7_event_driven_LONG"),
            ("short", short_data, "tp0_004_sl0_002_within_7_event_driven_SHORT"),
        ]
        for side, (x, y, close_time), label_type in side_configs:
            if len(x) < args.min_rows_per_symbol:
                print(
                    "SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={} side={} rows={} min={}".format(
                        symbol, side, len(x), args.min_rows_per_symbol
                    )
                )
                continue
            if args.test_rows <= 0 or args.val_rows <= 0:
                raise RuntimeError("--test-rows and --val-rows must be > 0")
            if len(x) <= args.test_rows + args.val_rows:
                print(
                    "SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={} side={} rows={} min={}".format(
                        symbol, side, len(x), args.min_rows_per_symbol
                    )
                )
                continue
            x = x.reset_index(drop=True)
            y = y.reset_index(drop=True)
            test_rows = args.test_rows
            val_rows = args.val_rows
            test_start = len(x) - test_rows
            val_start = test_start - val_rows
            x_train_pool = x.iloc[:val_start]
            y_train_pool = y.iloc[:val_start]
            x_val = x.iloc[val_start:test_start]
            y_val = y.iloc[val_start:test_start]
            x_test = x.iloc[test_start:]
            y_test = y.iloc[test_start:]
            val_close = close_time[val_start:test_start]
            test_close = close_time[test_start:]
            if len(x_train_pool) > TRAIN_LIMIT:
                x_train_pool = x_train_pool.iloc[-TRAIN_LIMIT:]
                y_train_pool = y_train_pool.iloc[-TRAIN_LIMIT:]

            def compute_span(values: np.ndarray) -> int:
                if len(values) == 0:
                    return 0
                start_ms = int(np.nanmin(values))
                end_ms = int(np.nanmax(values))
                return int(math.ceil((end_ms - start_ms + 1) / 86_400_000))

            val_days = compute_span(val_close)
            test_days = compute_span(test_close)
            pos_rate_val = float(np.mean(y_val)) if len(y_val) else 0.0
            pos_rate_test = float(np.mean(y_test)) if len(y_test) else 0.0
            baseline_val = max(pos_rate_val, 1.0 - pos_rate_val)
            baseline_test = max(pos_rate_test, 1.0 - pos_rate_test)
            print(
                "DATA_STATS symbol={} side={} trainRows={} valRows={} testRows={}".format(
                    symbol, side, len(x_train_pool), len(x_val), len(x_test)
                )
            )
            print(
                "SPAN symbol={} side={} valDays={} testDays={}".format(
                    symbol, side, val_days, test_days
                )
            )
            print(
                "POS_RATE symbol={} side={} valPos={:.6f} testPos={:.6f}".format(
                    symbol, side, pos_rate_val, pos_rate_test
                )
            )
            print(
                "BASELINE symbol={} side={} valBaseline={:.6f} testBaseline={:.6f}".format(
                    symbol, side, baseline_val, baseline_test
                )
            )
            conf_threshold = args.conf_threshold
            best = None
            lr_params = list(
                itertools.product(
                    [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                    [None, "balanced"],
                    ["lbfgs", "saga"],
                    [4000],
                    [1e-4, 1e-3],
                )
            )
            if not args.auto_tune:
                lr_params = [(1.0, None, "lbfgs", 4000, 1e-4)]
            if args.max_trials:
                lr_params = lr_params[: args.max_trials]
            for trial_index, params_raw in enumerate(lr_params, start=1):
                c_value, class_weight, solver, max_iter, tol = params_raw
                params = {
                    "C": c_value,
                    "class_weight": class_weight,
                    "solver": solver,
                    "max_iter": max_iter,
                    "tol": tol,
                }
                model = build_lr_pipeline(
                    solver,
                    c_value=c_value,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    tol=tol,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(x_train_pool, y_train_pool)
                has_convergence_warning = any(
                    isinstance(warning.message, ConvergenceWarning) for warning in caught
                )
                if has_convergence_warning:
                    print("WARN_CONVERGENCE symbol={} side={} params={}".format(symbol, side, params))
                val_proba = model.predict_proba(x_val)
                classes = list(getattr(model, "classes_", []))
                if not classes and isinstance(model, Pipeline):
                    classes = list(model.named_steps["classifier"].classes_)
                if 1 not in classes:
                    raise RuntimeError(f"Class 1 missing from classes for symbol {symbol}: {classes}")
                pos_index = classes.index(1)
                p_hit = val_proba[:, pos_index]
                val_preds = model.predict(x_val)
                acc_hi, coverage, val_orders, val_orders_per_day = compute_confidence_metrics(
                    y_val,
                    val_preds,
                    p_hit,
                    conf_threshold=conf_threshold,
                    days_span=val_days,
                )
                print(
                    "TUNE symbol={} side={} trial={} params={} accHi={} coverage={:.6f} "
                    "valOrders={} valOrdersPerDay={:.6f}".format(
                        symbol,
                        side,
                        trial_index,
                        params,
                        "nan" if np.isnan(acc_hi) else f"{acc_hi:.6f}",
                        coverage,
                        val_orders,
                        val_orders_per_day,
                    )
                )
                score_acc_hi = -1.0 if np.isnan(acc_hi) else acc_hi
                metric = (score_acc_hi, val_orders_per_day)
                candidate = {
                    "metric": metric,
                    "params": params,
                    "acc_hi": acc_hi,
                    "coverage": coverage,
                    "val_orders": val_orders,
                    "val_orders_per_day": val_orders_per_day,
                }
                if best is None or metric > best["metric"]:
                    best = candidate
                if (
                    score_acc_hi >= args.target_acc
                    and val_orders_per_day >= args.min_orders_per_day
                    and val_orders >= args.min_total_orders
                ):
                    break
            if best is None:
                print(
                    "SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={} side={} rows={} min={}".format(
                        symbol, side, len(x), args.min_rows_per_symbol
                    )
                )
                continue
            best_params = best["params"]
            train_val_x = pd.concat([x_train_pool, x_val], axis=0)
            train_val_y = pd.concat([y_train_pool, y_val], axis=0)
            final_model = build_lr_pipeline(
                best_params["solver"],
                c_value=best_params["C"],
                class_weight=best_params["class_weight"],
                max_iter=best_params["max_iter"],
                tol=best_params["tol"],
            )
            final_model.fit(train_val_x, train_val_y)
            val_proba = final_model.predict_proba(x_val)
            test_proba = final_model.predict_proba(x_test)
            classes = list(getattr(final_model, "classes_", []))
            if not classes and isinstance(final_model, Pipeline):
                classes = list(final_model.named_steps["classifier"].classes_)
            if 1 not in classes:
                raise RuntimeError(f"Class 1 missing from classes for symbol {symbol}: {classes}")
            up_class_index = classes.index(1)
            val_p_hit = val_proba[:, up_class_index]
            test_p_hit = test_proba[:, up_class_index]
            val_preds = final_model.predict(x_val)
            test_preds = final_model.predict(x_test)
            val_acc_hi, val_coverage, val_orders, val_orders_per_day = compute_confidence_metrics(
                y_val,
                val_preds,
                val_p_hit,
                conf_threshold=conf_threshold,
                days_span=val_days,
            )
            test_acc_hi, test_coverage, test_orders, test_orders_per_day = compute_confidence_metrics(
                y_test,
                test_preds,
                test_p_hit,
                conf_threshold=conf_threshold,
                days_span=test_days,
            )
            print(
                "ORDERS symbol={} side={} valOrders={} valOrdersPerDay={:.6f} "
                "testOrders={} testOrdersPerDay={:.6f}".format(
                    symbol,
                    side,
                    val_orders,
                    val_orders_per_day,
                    test_orders,
                    test_orders_per_day,
                )
            )
            print(
                "FINAL symbol={} side={} testAccHi={} testOrders={} testDays={} testOrdersPerDay={:.6f}".format(
                    symbol,
                    side,
                    "nan" if np.isnan(test_acc_hi) else f"{test_acc_hi:.6f}",
                    test_orders,
                    test_days,
                    test_orders_per_day,
                )
            )
            model_version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
            model_dir = args.out_dir / symbol / f"{model_version}_{side}"
            export_fallback = False
            if args.eval_only:
                print(
                    "EVAL_ONLY symbol={} side={} train_rows={} val_rows={} test_rows={}".format(
                        symbol, side, len(train_val_x), val_rows, test_rows
                    )
                )
                continue
            input_names, output_names = export_onnx(final_model, x.shape[1], model_dir / "model.onnx")
            print(
                "ONNX_EXPORT symbol={} side={} inputs={} outputs={}".format(
                    symbol, side, input_names, output_names
                )
            )
            prob_output_name = "probabilities" if "probabilities" in output_names else None
            x_check = x.iloc[:256].to_numpy(dtype=np.float32)
            check_onnx_outputs(symbol, model_dir / "model.onnx", x_check)
            train_days = len(extract_dates(feature_files))
            write_meta(
                model_dir / "model_meta.json",
                model_version,
                symbol,
                train_days,
                len(train_val_x),
                [int(value) for value in classes],
                up_class_index,
                feature_order,
                output_names,
                prob_output_name,
                label_type,
                best_params,
                0.0 if np.isnan(val_acc_hi) else float(val_acc_hi),
                float(val_coverage),
                val_orders,
                val_orders_per_day,
                val_days,
                val_rows,
                0.0 if np.isnan(test_acc_hi) else float(test_acc_hi),
                float(test_coverage),
                test_orders,
                test_orders_per_day,
                test_days,
                test_rows,
                export_fallback,
            )
            write_current(model_dir, args.out_dir, symbol, side)
            wrote_path = args.out_dir / symbol / f"current_{side}"
            print(
                "TRAIN_SYMBOL symbol={} side={} rows={} wrote={}".format(
                    symbol,
                    side,
                    len(train_val_x),
                    wrote_path,
                )
            )


if __name__ == "__main__":
    main()
