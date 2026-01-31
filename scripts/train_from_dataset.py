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
from sklearn.metrics import accuracy_score, confusion_matrix
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
LABEL_TYPE = "tp0_004_sl0_002_within_7_tp_before_sl"
MAX_HORIZON_BARS = 7
TP_PCT = 0.004
SL_PCT = 0.002
GAP_MS = 300000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model from dataset JSONL.gz files.")
    parser.add_argument("--data-dir", default=Path("data"), type=Path, help="Root data directory")
    parser.add_argument("--out-dir", default=Path("models"), type=Path, help="Output models directory")
    parser.add_argument("--exclude-today", action="store_true", help="Exclude today's partition (Europe/Istanbul)")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols to include")
    parser.add_argument("--min-rows-per-symbol", default=20000, type=int, help="Minimum rows required to train")
    parser.add_argument("--train-rows", default=None, type=int, help="Rows to use for training (most recent)")
    parser.add_argument("--train-grid", default=None, help="Comma-separated train row sizes to evaluate")
    parser.add_argument("--test-rows", default=2000, type=int, help="Rows to hold out for evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (skip export)")
    parser.add_argument("--fast-tail", action="store_true", help="Read only the latest rows needed")
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
    collected: list[pd.DataFrame] = []
    total_rows = 0
    for path in reversed(feature_files):
        frame = read_jsonl_gz(path)
        collected.append(frame)
        total_rows += len(frame)
        if total_rows >= need_rows:
            break
    if not collected:
        return pd.DataFrame(), feature_files
    features = pd.concat(reversed(collected), ignore_index=True)
    return features, feature_files


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
    collected: list[pd.DataFrame] = []
    total_rows = 0
    for path in reversed(raw_files):
        frame = read_jsonl_gz(path)
        collected.append(frame)
        total_rows += len(frame)
        if total_rows >= need_rows:
            break
    if not collected:
        return pd.DataFrame(), raw_files
    raw = pd.concat(reversed(collected), ignore_index=True)
    return raw, raw_files


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
        label_hit = 0
        event = "NO_TP"
        time_to_event: int | None = None
        invalid = False
        for k in range(1, MAX_HORIZON_BARS + 1):
            hi = high[idx + k]
            lo = low[idx + k]
            if np.isnan(hi) or np.isnan(lo):
                invalid = True
                break
            hit_tp = hi >= tp_price
            hit_sl = lo <= sl_price
            if hit_tp and hit_sl:
                label_hit = 0
                event = "SL_FIRST"
                time_to_event = k
                break
            if hit_sl:
                label_hit = 0
                event = "SL_FIRST"
                time_to_event = k
                break
            if hit_tp:
                label_hit = 1
                event = "TP_FIRST"
                time_to_event = k
                break
        if invalid:
            continue
        record: dict[str, object] = {
            "closeTimeMs": int(close_time[idx]),
            "labelType": LABEL_TYPE,
            "labelHit": int(label_hit),
            "event": event,
            "timeToEvent": time_to_event,
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
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
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
    missing_features = [col for col in FEATURE_ORDER if col not in merged.columns]
    if missing_features:
        raise RuntimeError(f"Missing expected feature columns: {missing_features}")
    x = merged[FEATURE_ORDER].copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x = x.astype(np.float32)
    y = merged["labelHit"].astype(int)
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
    classes: list[int],
    up_class_index: int,
    feature_order: list[str],
    onnx_outputs: list[str],
    prob_output_name: str | None,
    last_eval_acc_hi: float,
    last_eval_coverage: float,
    last_eval_acc_all: float,
    best_train_rows: int,
    test_rows: int,
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
        "lastEvalAccHi": last_eval_acc_hi,
        "lastEvalCoverage": last_eval_coverage,
        "lastEvalAccAll": last_eval_acc_all,
        "bestTrainRows": best_train_rows,
        "testRows": test_rows,
        "onnxOutputs": onnx_outputs,
        "probOutputName": prob_output_name,
        "decisionPolicy": {
            "minConfidence": 0.55,
            "minAbsExpectedPct": 0.002,
            "minAbsEdge": 0.05,
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


def parse_train_grid(value: str | None) -> list[int] | None:
    if not value:
        return None
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    return parsed or None


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float, float, dict[str, int], float]:
    proba = model.predict_proba(x_test)
    classes = list(model.named_steps["classifier"].classes_)
    if 1 not in classes:
        raise RuntimeError(f"Class 1 missing from classes: {classes}")
    pos_index = classes.index(1)
    preds = model.predict(x_test)
    acc_all = float(accuracy_score(y_test, preds))
    p_hit = proba[:, pos_index]
    confidence = np.maximum(p_hit, 1.0 - p_hit)
    coverage_mask = confidence >= 0.55
    coverage = float(np.mean(coverage_mask)) if len(confidence) else 0.0
    if np.any(coverage_mask):
        acc_hi = float(accuracy_score(y_test[coverage_mask], preds[coverage_mask]))
    else:
        acc_hi = 0.0
    matrix = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    confusion = {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
    win_rate = float(np.mean(y_test)) if len(y_test) else 0.0
    return acc_all, coverage, acc_hi, confusion, win_rate


def main() -> None:
    args = parse_args()
    if args.symbols:
        symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()]
    else:
        features_root = args.data_dir / "features"
        if not features_root.exists():
            raise RuntimeError(f"No features directory found at {features_root}")
        symbols = sorted([path.name.upper() for path in features_root.iterdir() if path.is_dir()])
    train_grid = parse_train_grid(args.train_grid)
    for symbol in symbols:
        if train_grid:
            max_train_rows = max(train_grid)
        else:
            max_train_rows = args.train_rows
        if args.test_rows:
            need_rows = (max_train_rows or 0) + args.test_rows + MAX_HORIZON_BARS + 1
        else:
            need_rows = None
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
        x, y, merged, feature_order = build_training_frame(features, raw)
        if len(x) < args.min_rows_per_symbol:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows={len(x)} min={args.min_rows_per_symbol}")
            continue
        if args.test_rows <= 0:
            raise RuntimeError("--test-rows must be > 0")
        if len(x) <= args.test_rows:
            print(
                f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows={len(x)} "
                f"min={args.min_rows_per_symbol}"
            )
            continue
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        test_rows = args.test_rows
        test_start = len(x) - test_rows
        x_train_pool = x.iloc[:test_start]
        y_train_pool = y.iloc[:test_start]
        x_test = x.iloc[test_start:]
        y_test = y.iloc[test_start:]
        train_grid_values = train_grid or ([args.train_rows] if args.train_rows else [len(x_train_pool)])
        best = None
        for train_rows in train_grid_values:
            if train_rows is None:
                train_rows = len(x_train_pool)
            if train_rows > len(x_train_pool):
                print(
                    f"SKIP_TRAIN_GRID symbol={symbol} train={train_rows} available={len(x_train_pool)}"
                )
                continue
            x_train = x_train_pool.iloc[len(x_train_pool) - train_rows :]
            y_train = y_train_pool.iloc[len(y_train_pool) - train_rows :]
            base_pipeline = build_pipeline("lbfgs")
            print(
                "MODEL_CONFIG symbol={} solver=lbfgs max_iter=4000 scaler=StandardScaler train_rows={}".format(
                    symbol, train_rows
                )
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                base_pipeline.fit(x_train, y_train)
            has_convergence_warning = any(
                isinstance(warning.message, ConvergenceWarning) for warning in caught
            )
            if has_convergence_warning:
                print(
                    "WARN_CONVERGENCE symbol={} solver=lbfgs fallback_solver=saga".format(symbol)
                )
                base_pipeline = build_pipeline("saga")
                print(
                    "MODEL_CONFIG symbol={} solver=saga max_iter=4000 scaler=StandardScaler train_rows={}".format(
                        symbol, train_rows
                    )
                )
                base_pipeline.fit(x_train, y_train)
            calibrated = None
            if args.calibrate:
                tscv = TimeSeriesSplit(n_splits=3)
                calibrated = CalibratedClassifierCV(
                    estimator=base_pipeline, method="sigmoid", cv=tscv
                )
                calibrated.fit(x_train, y_train)
            model_for_eval = calibrated if calibrated is not None else base_pipeline
            acc_all, coverage, acc_hi, confusion, win_rate = evaluate_model(
                model_for_eval, x_test, y_test
            )
            print(
                "EVAL symbol={} train={} test={} acc={:.6f} coverage={:.6f} accHi={:.6f} winRate={:.6f}".format(
                    symbol, train_rows, test_rows, acc_all, coverage, acc_hi, win_rate
                )
            )
            print(
                "CONFUSION symbol={} train={} tp={} tn={} fp={} fn={}".format(
                    symbol,
                    train_rows,
                    confusion["tp"],
                    confusion["tn"],
                    confusion["fp"],
                    confusion["fn"],
                )
            )
            metric = (acc_hi, acc_all, coverage)
            if best is None or metric > best["metric"]:
                best = {
                    "metric": metric,
                    "train_rows": train_rows,
                    "base_model": base_pipeline,
                    "calibrated_model": calibrated,
                    "acc_all": acc_all,
                    "coverage": coverage,
                    "acc_hi": acc_hi,
                    "win_rate": win_rate,
                }
        if best is None:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows={len(x)} min={args.min_rows_per_symbol}")
            continue
        model_version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        model_dir = args.out_dir / symbol / model_version
        classifier = best["base_model"].named_steps["classifier"]
        classes = list(getattr(classifier, "classes_", []))
        if not classes:
            raise RuntimeError(f"No classes_ found for symbol {symbol}")
        if 1 not in classes:
            raise RuntimeError(f"Class 1 missing from classes for symbol {symbol}: {classes}")
        up_class_index = classes.index(1)
        best_train_rows = int(best["train_rows"])
        export_model = best["calibrated_model"] if best["calibrated_model"] is not None else best["base_model"]
        if args.eval_only:
            print(
                "EVAL_ONLY symbol={} train_rows={} test_rows={}".format(
                    symbol, best_train_rows, test_rows
                )
            )
            continue
        try:
            input_names, output_names = export_onnx(export_model, x.shape[1], model_dir / "model.onnx")
            print(f"ONNX_EXPORT symbol={symbol} inputs={input_names} outputs={output_names}")
        except Exception as exc:
            print(f"ONNX_EXPORT_FAILED symbol={symbol} calibrated failed ({exc}); fallback: export non-calibrated.")
            export_model = best["base_model"]
            input_names, output_names = export_onnx(export_model, x.shape[1], model_dir / "model.onnx")
            print(f"ONNX_EXPORT symbol={symbol} inputs={input_names} outputs={output_names}")
        prob_output_name = "probabilities" if "probabilities" in output_names else None
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
                    if onnx_probs.ndim >= 2 and onnx_probs.shape[1] == len(classes):
                        mean_conf = float(np.mean(np.max(onnx_probs, axis=1)))
                        print(f"ONNX_CHECK symbol={symbol} meanMaxProb={mean_conf:.6f}")
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
            best_train_rows,
            [int(value) for value in classes],
            up_class_index,
            feature_order,
            output_names,
            prob_output_name,
            best["acc_hi"],
            best["coverage"],
            best["acc_all"],
            best_train_rows,
            test_rows,
        )
        write_current(model_dir, args.out_dir, symbol)
        wrote_path = args.out_dir / symbol / "current"
        print(
            "TRAIN_SYMBOL symbol={} rows={} calibrated={} wrote={}".format(
                symbol,
                best_train_rows,
                args.calibrate,
                wrote_path,
            )
        )


if __name__ == "__main__":
    main()
