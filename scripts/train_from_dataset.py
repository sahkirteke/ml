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
LABEL_TYPE = "tp0_004_sl0_002_within_7_tp_before_sl"
MAX_HORIZON_BARS = 7
TP_PCT = 0.004
SL_PCT = 0.002
GAP_MS = 300000
CONF_THRESHOLD = 0.55
P_TRADE_CANDIDATES = [0.55, 0.60, 0.65]
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


def build_labels_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["closeTimeMs", "highPrice", "lowPrice", "closePrice"]
    missing = [col for col in required_columns if col not in raw.columns]
    if missing:
        raise RuntimeError(f"Raw data missing columns: {missing}")
    raw_sorted = raw.sort_values("closeTimeMs").reset_index(drop=True)
    close_time = pd.to_numeric(raw_sorted["closeTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(raw_sorted["highPrice"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(raw_sorted["lowPrice"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(raw_sorted["closePrice"], errors="coerce").to_numpy(dtype=np.float64)
    records: list[dict[str, object]] = []
    gap_ok = np.diff(close_time) == GAP_MS
    for idx in range(len(raw_sorted) - MAX_HORIZON_BARS):
        if np.isnan(close_time[idx]) or np.isnan(close[idx]):
            continue
        if not gap_ok[idx : idx + MAX_HORIZON_BARS].all():
            continue
        entry = close[idx]
        tp_price = entry * (1.0 + TP_PCT)
        sl_price = entry * (1.0 - SL_PCT)
        label_hit_long = 0
        label_hit_short = 0
        event_long = "NO_TP"
        event_short = "NO_TP"
        time_to_event_long: int | None = None
        time_to_event_short: int | None = None
        invalid = False
        for k in range(1, MAX_HORIZON_BARS + 1):
            hi = high[idx + k]
            lo = low[idx + k]
            if np.isnan(hi) or np.isnan(lo):
                invalid = True
                break
            hit_tp_long = hi >= tp_price
            hit_sl_long = lo <= sl_price
            if hit_tp_long and hit_sl_long:
                label_hit_long = 0
                event_long = "SL_FIRST"
                time_to_event_long = k
                break
            if hit_sl_long:
                label_hit_long = 0
                event_long = "SL_FIRST"
                time_to_event_long = k
                break
            if hit_tp_long:
                label_hit_long = 1
                event_long = "TP_FIRST"
                time_to_event_long = k
                break
        if not invalid:
            tp_price_short = entry * (1.0 - TP_PCT)
            sl_price_short = entry * (1.0 + SL_PCT)
            for k in range(1, MAX_HORIZON_BARS + 1):
                hi = high[idx + k]
                lo = low[idx + k]
                if np.isnan(hi) or np.isnan(lo):
                    invalid = True
                    break
                hit_tp_short = lo <= tp_price_short
                hit_sl_short = hi >= sl_price_short
                if hit_tp_short and hit_sl_short:
                    label_hit_short = 0
                    event_short = "SL_FIRST"
                    time_to_event_short = k
                    break
                if hit_sl_short:
                    label_hit_short = 0
                    event_short = "SL_FIRST"
                    time_to_event_short = k
                    break
                if hit_tp_short:
                    label_hit_short = 1
                    event_short = "TP_FIRST"
                    time_to_event_short = k
                    break
        if invalid:
            continue
        record: dict[str, object] = {
            "closeTimeMs": int(close_time[idx]),
            "labelType": LABEL_TYPE,
            "labelHitLong": int(label_hit_long),
            "labelHitShort": int(label_hit_short),
            "eventLong": event_long,
            "eventShort": event_short,
            "timeToEventLong": time_to_event_long,
            "timeToEventShort": time_to_event_short,
            "tpPct": TP_PCT,
            "slPct": SL_PCT,
            "maxHorizonBars": MAX_HORIZON_BARS,
        }
        if "symbol" in raw_sorted.columns:
            record["symbol"] = raw_sorted.at[idx, "symbol"]
        if "tf" in raw_sorted.columns:
            record["tf"] = raw_sorted.at[idx, "tf"]
        records.append(record)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def build_training_frame(
    features: pd.DataFrame, raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, list[str]]:
    features_filtered = features[(features["windowReady"] == True) & (features["featuresVersion"] == FEATURES_VERSION)]
    if features_filtered.empty:
        raise RuntimeError("No rows available after filtering features")
    if raw.empty:
        raise RuntimeError("Raw data required for label generation is empty")
    label_frame = build_labels_from_raw(raw)
    if label_frame.empty:
        raise RuntimeError("No rows available after labeling")
    if "symbol" not in label_frame.columns:
        if "symbol" in features_filtered.columns and features_filtered["symbol"].nunique() == 1:
            label_frame["symbol"] = features_filtered["symbol"].iloc[0]
        else:
            raise RuntimeError("Label data missing symbol for join")
    if "tf" not in label_frame.columns:
        if "tf" in features_filtered.columns and features_filtered["tf"].nunique() == 1:
            label_frame["tf"] = features_filtered["tf"].iloc[0]
        else:
            raise RuntimeError("Label data missing tf for join")
    merged = features_filtered.merge(
        label_frame,
        on=["symbol", "tf", "closeTimeMs"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No rows available after joining features and labels")
    merged = merged.sort_values("closeTimeMs").reset_index(drop=True)
    missing_features = [col for col in FEATURE_ORDER if col not in merged.columns]
    if missing_features:
        raise RuntimeError(f"Missing expected feature columns: {missing_features}")
    x = merged[FEATURE_ORDER].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x = x.astype(np.float32)
    y_long = merged["labelHitLong"].astype(int)
    y_short = merged["labelHitShort"].astype(int)
    stds = x.std(axis=0, skipna=True)
    keep_cols = [col for col in x.columns if stds[col] > 0]
    if keep_cols and len(keep_cols) != len(x.columns):
        x = x[keep_cols].copy()
    return x, y_long, y_short, merged, list(x.columns)


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
    best_params: dict[str, object],
    p_trade: float,
    val_acc_trade: float,
    val_orders: int,
    val_orders_per_day: float,
    val_days: int,
    val_rows: int,
    test_acc_trade: float,
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
        "labelType": LABEL_TYPE,
        "tpPct": TP_PCT,
        "slPct": SL_PCT,
        "maxHorizonBars": MAX_HORIZON_BARS,
        "classes": classes,
        "upClassIndex": up_class_index,
        "bestParams": best_params,
        "pTradeChosen": p_trade,
        "valAccTrade": val_acc_trade,
        "valOrders": val_orders,
        "valOrdersPerDay": val_orders_per_day,
        "valDaysSpan": val_days,
        "valRows": val_rows,
        "testAccTrade": test_acc_trade,
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
            print("WARN_ONNX_MISMATCH symbol={} labelShape={} probShape={}".format(symbol, None, None))
            return
        prob_output = None
        label_output = None
        output_shapes = [np.array(output).shape for output in outputs]
        for output in outputs:
            arr = np.array(output)
            if arr.ndim == 1 and arr.shape[0] == 2:
                prob_output = arr
            elif arr.ndim == 2 and arr.shape[1] == 2:
                prob_output = arr
            elif arr.ndim == 1:
                label_output = arr
        if len(outputs) == 2 and prob_output is not None and label_output is not None:
            return
        label_shape = output_shapes[0] if output_shapes else None
        prob_shape = output_shapes[1] if len(output_shapes) > 1 else None
        print(
            "WARN_ONNX_MISMATCH symbol={} labelShape={} probShape={}".format(
                symbol, label_shape, prob_shape
            )
        )
    except Exception as exc:
        print(f"WARN_ONNX_MISMATCH symbol={symbol} error={exc}")


def compute_trade_metrics(
    y_long: pd.Series,
    y_short: pd.Series,
    p_long: np.ndarray,
    p_short: np.ndarray,
    *,
    p_trade: float,
    days_span: int,
) -> tuple[float, int, float]:
    decision_long = p_long >= p_trade
    decision_short = p_short >= p_trade
    trade_mask = decision_long | decision_short
    orders = int(np.sum(trade_mask))
    orders_per_day = float(orders / max(days_span, 1))
    if not np.any(trade_mask):
        return float("nan"), orders, orders_per_day
    use_long = decision_long & (~decision_short | (p_long >= p_short))
    use_short = decision_short & (~decision_long | (p_short > p_long))
    correct_long = (y_long[use_long] == 1).to_numpy()
    correct_short = (y_short[use_short] == 1).to_numpy()
    if correct_long.size and correct_short.size:
        correct = np.concatenate([correct_long, correct_short])
    elif correct_long.size:
        correct = correct_long
    else:
        correct = correct_short
    acc_trade = float(np.mean(correct)) if correct.size else float("nan")
    return acc_trade, orders, orders_per_day


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
        x, y_long, y_short, merged, feature_order = build_training_frame(features, raw)
        if len(x) < args.min_rows_per_symbol:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows={len(x)} min={args.min_rows_per_symbol}")
            continue
        if args.test_rows <= 0 or args.val_rows <= 0:
            raise RuntimeError("--test-rows and --val-rows must be > 0")
        if len(x) <= args.test_rows + args.val_rows:
            print(
                f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows={len(x)} "
                f"min={args.min_rows_per_symbol}"
            )
            continue
        x = x.reset_index(drop=True)
        y_long = y_long.reset_index(drop=True)
        y_short = y_short.reset_index(drop=True)
        merged = merged.reset_index(drop=True)
        test_rows = args.test_rows
        val_rows = args.val_rows
        test_start = len(x) - test_rows
        val_start = test_start - val_rows
        x_train_pool = x.iloc[:val_start]
        y_long_train = y_long.iloc[:val_start]
        y_short_train = y_short.iloc[:val_start]
        x_val = x.iloc[val_start:test_start]
        y_long_val = y_long.iloc[val_start:test_start]
        y_short_val = y_short.iloc[val_start:test_start]
        x_test = x.iloc[test_start:]
        y_long_test = y_long.iloc[test_start:]
        y_short_test = y_short.iloc[test_start:]
        close_time = pd.to_numeric(merged["closeTimeMs"], errors="coerce").to_numpy(dtype=np.float64)
        val_close = close_time[val_start:test_start]
        test_close = close_time[test_start:]
        if len(x_train_pool) > TRAIN_LIMIT:
            x_train_pool = x_train_pool.iloc[-TRAIN_LIMIT:]
            y_long_train = y_long_train.iloc[-TRAIN_LIMIT:]
            y_short_train = y_short_train.iloc[-TRAIN_LIMIT:]
        def compute_span(values: np.ndarray) -> int:
            if len(values) == 0:
                return 0
            start_ms = int(np.nanmin(values))
            end_ms = int(np.nanmax(values))
            days_span = int(math.ceil((end_ms - start_ms + 1) / 86_400_000))
            return days_span

        val_days = compute_span(val_close)
        test_days = compute_span(test_close)
        pos_rate_val_long = float(np.mean(y_long_val)) if len(y_long_val) else 0.0
        pos_rate_val_short = float(np.mean(y_short_val)) if len(y_short_val) else 0.0
        pos_rate_test_long = float(np.mean(y_long_test)) if len(y_long_test) else 0.0
        pos_rate_test_short = float(np.mean(y_short_test)) if len(y_short_test) else 0.0
        baseline_val_long = max(pos_rate_val_long, 1.0 - pos_rate_val_long)
        baseline_val_short = max(pos_rate_val_short, 1.0 - pos_rate_val_short)
        baseline_test_long = max(pos_rate_test_long, 1.0 - pos_rate_test_long)
        baseline_test_short = max(pos_rate_test_short, 1.0 - pos_rate_test_short)
        print(
            "DATA_STATS symbol={} trainRows={} valRows={} testRows={}".format(
                symbol, len(x_train_pool), len(x_val), len(x_test)
            )
        )
        print(
            "SPAN symbol={} valDays={} testDays={}".format(
                symbol,
                val_days,
                test_days,
            )
        )
        print(
            "POS_RATE symbol={} valPosLong={:.6f} valPosShort={:.6f} testPosLong={:.6f} testPosShort={:.6f}".format(
                symbol,
                pos_rate_val_long,
                pos_rate_val_short,
                pos_rate_test_long,
                pos_rate_test_short,
            )
        )
        print(
            "BASELINE symbol={} valBaseLong={:.6f} valBaseShort={:.6f} testBaseLong={:.6f} testBaseShort={:.6f}".format(
                symbol,
                baseline_val_long,
                baseline_val_short,
                baseline_test_long,
                baseline_test_short,
            )
        )
        best = None
        trial_candidates = []
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
        for params_raw in lr_params:
            for p_trade in P_TRADE_CANDIDATES:
                trial_candidates.append((params_raw, p_trade))
        if args.max_trials:
            trial_candidates = trial_candidates[: args.max_trials]
        for trial_index, (params_raw, p_trade) in enumerate(trial_candidates, start=1):
            c_value, class_weight, solver, max_iter, tol = params_raw
            params = {
                "C": c_value,
                "class_weight": class_weight,
                "solver": solver,
                "max_iter": max_iter,
                "tol": tol,
            }
            model_long = build_lr_pipeline(
                solver,
                c_value=c_value,
                class_weight=class_weight,
                max_iter=max_iter,
                tol=tol,
            )
            model_short = build_lr_pipeline(
                solver,
                c_value=c_value,
                class_weight=class_weight,
                max_iter=max_iter,
                tol=tol,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model_long.fit(x_train_pool, y_long_train)
                model_short.fit(x_train_pool, y_short_train)
            has_convergence_warning = any(
                isinstance(warning.message, ConvergenceWarning) for warning in caught
            )
            if has_convergence_warning:
                print("WARN_CONVERGENCE symbol={} params={}".format(symbol, params))
            val_proba_long = model_long.predict_proba(x_val)
            val_proba_short = model_short.predict_proba(x_val)
            classes_long = list(getattr(model_long, "classes_", []))
            classes_short = list(getattr(model_short, "classes_", []))
            if not classes_long and isinstance(model_long, Pipeline):
                classes_long = list(model_long.named_steps["classifier"].classes_)
            if not classes_short and isinstance(model_short, Pipeline):
                classes_short = list(model_short.named_steps["classifier"].classes_)
            if 1 not in classes_long or 1 not in classes_short:
                raise RuntimeError(
                    f"Class 1 missing from classes for symbol {symbol}: long={classes_long}, short={classes_short}"
                )
            pos_index_long = classes_long.index(1)
            pos_index_short = classes_short.index(1)
            p_long = val_proba_long[:, pos_index_long]
            p_short = val_proba_short[:, pos_index_short]
            acc_trade, val_orders, val_orders_per_day = compute_trade_metrics(
                y_long_val,
                y_short_val,
                p_long,
                p_short,
                p_trade=p_trade,
                days_span=val_days,
            )
            print(
                "TUNE symbol={} trial={} params={} pTrade={:.2f} accTrade={} valOrders={} valOrdersPerDay={:.6f}".format(
                    symbol,
                    trial_index,
                    params,
                    p_trade,
                    "nan" if np.isnan(acc_trade) else f"{acc_trade:.6f}",
                    val_orders,
                    val_orders_per_day,
                )
            )
            score_acc_trade = -1.0 if np.isnan(acc_trade) else acc_trade
            metric = (score_acc_trade, val_orders_per_day)
            candidate = {
                "metric": metric,
                "params": params,
                "p_trade": p_trade,
                "acc_trade": acc_trade,
                "val_orders": val_orders,
                "val_orders_per_day": val_orders_per_day,
            }
            if best is None or metric > best["metric"]:
                best = candidate
            if (
                score_acc_trade >= args.target_acc
                and val_orders_per_day >= args.min_orders_per_day
                and val_orders >= args.min_total_orders
            ):
                break
        if best is None:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows={len(x)} min={args.min_rows_per_symbol}")
            continue
        best_params = best["params"]
        best_p_trade = best["p_trade"]
        train_val_x = pd.concat([x_train_pool, x_val], axis=0)
        train_val_y_long = pd.concat([y_long_train, y_long_val], axis=0)
        train_val_y_short = pd.concat([y_short_train, y_short_val], axis=0)
        final_model_long = build_lr_pipeline(
            best_params["solver"],
            c_value=best_params["C"],
            class_weight=best_params["class_weight"],
            max_iter=best_params["max_iter"],
            tol=best_params["tol"],
        )
        final_model_short = build_lr_pipeline(
            best_params["solver"],
            c_value=best_params["C"],
            class_weight=best_params["class_weight"],
            max_iter=best_params["max_iter"],
            tol=best_params["tol"],
        )
        final_model_long.fit(train_val_x, train_val_y_long)
        final_model_short.fit(train_val_x, train_val_y_short)
        val_proba_long = final_model_long.predict_proba(x_val)
        val_proba_short = final_model_short.predict_proba(x_val)
        test_proba_long = final_model_long.predict_proba(x_test)
        test_proba_short = final_model_short.predict_proba(x_test)
        classes_long = list(getattr(final_model_long, "classes_", []))
        classes_short = list(getattr(final_model_short, "classes_", []))
        if not classes_long and isinstance(final_model_long, Pipeline):
            classes_long = list(final_model_long.named_steps["classifier"].classes_)
        if not classes_short and isinstance(final_model_short, Pipeline):
            classes_short = list(final_model_short.named_steps["classifier"].classes_)
        if 1 not in classes_long or 1 not in classes_short:
            raise RuntimeError(
                f"Class 1 missing from classes for symbol {symbol}: long={classes_long}, short={classes_short}"
            )
        up_class_index_long = classes_long.index(1)
        up_class_index_short = classes_short.index(1)
        val_p_long = val_proba_long[:, up_class_index_long]
        val_p_short = val_proba_short[:, up_class_index_short]
        test_p_long = test_proba_long[:, up_class_index_long]
        test_p_short = test_proba_short[:, up_class_index_short]
        val_acc_trade, val_orders, val_orders_per_day = compute_trade_metrics(
            y_long_val,
            y_short_val,
            val_p_long,
            val_p_short,
            p_trade=best_p_trade,
            days_span=val_days,
        )
        test_acc_trade, test_orders, test_orders_per_day = compute_trade_metrics(
            y_long_test,
            y_short_test,
            test_p_long,
            test_p_short,
            p_trade=best_p_trade,
            days_span=test_days,
        )
        print(
            "ORDERS symbol={} pTrade={:.2f} valOrders={} valOrdersPerDay={:.6f} "
            "testOrders={} testOrdersPerDay={:.6f}".format(
                symbol,
                best_p_trade,
                val_orders,
                val_orders_per_day,
                test_orders,
                test_orders_per_day,
            )
        )
        print(
            "FINAL symbol={} bestParams={} testAccTrade={} testOrders={} testDays={} testOrdersPerDay={:.6f}".format(
                symbol,
                best_params,
                "nan" if np.isnan(test_acc_trade) else f"{test_acc_trade:.6f}",
                test_orders,
                test_days,
                test_orders_per_day,
            )
        )
        model_version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        model_dir_long = args.out_dir / symbol / f"{model_version}_long"
        model_dir_short = args.out_dir / symbol / f"{model_version}_short"
        export_fallback = False
        if args.eval_only:
            print(
                "EVAL_ONLY symbol={} train_rows={} val_rows={} test_rows={}".format(
                    symbol, len(train_val_x), val_rows, test_rows
                )
            )
            continue
        x_check = x.iloc[:256].to_numpy(dtype=np.float32)
        train_days = len(extract_dates(feature_files))
        for variant, model, up_class_index, classes, model_dir in (
            ("long", final_model_long, up_class_index_long, classes_long, model_dir_long),
            ("short", final_model_short, up_class_index_short, classes_short, model_dir_short),
        ):
            input_names, output_names = export_onnx(model, x.shape[1], model_dir / "model.onnx")
            print(
                "ONNX_EXPORT symbol={} side={} inputs={} outputs={}".format(
                    symbol, variant, input_names, output_names
                )
            )
            prob_output_name = "probabilities" if "probabilities" in output_names else None
            check_onnx_outputs(symbol, model_dir / "model.onnx", x_check)
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
                best_params,
                best_p_trade,
                0.0 if np.isnan(val_acc_trade) else float(val_acc_trade),
                val_orders,
                val_orders_per_day,
                val_days,
                val_rows,
                0.0 if np.isnan(test_acc_trade) else float(test_acc_trade),
                test_orders,
                test_orders_per_day,
                test_days,
                test_rows,
                export_fallback,
            )
            write_current(model_dir, args.out_dir, symbol, variant)
        wrote_long = args.out_dir / symbol / "current_long"
        wrote_short = args.out_dir / symbol / "current_short"
        print(
            "TRAIN_SYMBOL symbol={} rows={} wroteLong={} wroteShort={}".format(
                symbol,
                len(train_val_x),
                wrote_long,
                wrote_short,
            )
        )


if __name__ == "__main__":
    main()
