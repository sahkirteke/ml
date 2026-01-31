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
    parser.add_argument("--test-rows", default=100000, type=int, help="Rows to use for evaluation test block")
    parser.add_argument(
        "--train-grid",
        default="100000,200000,300000,400000",
        help="Comma-separated train row sizes to evaluate",
    )
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation without ONNX export")
    parser.add_argument("--fast-tail", action="store_true", help="Read only the newest daily files if possible")
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


def tail_files_by_days(paths: list[Path], days: int) -> list[Path]:
    if days <= 0:
        return paths
    pattern = re.compile(r"-(\d{8})\.jsonl\.gz$")
    dated: list[tuple[str, Path]] = []
    for path in paths:
        match = pattern.search(path.name)
        if match:
            dated.append((match.group(1), path))
    if not dated:
        return paths
    dated.sort(key=lambda item: item[0])
    keep = dated[-days:]
    return [path for _, path in keep]


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
    fast_tail: bool = False,
    fast_tail_days: int | None = None,
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
    if fast_tail and fast_tail_days:
        feature_files = tail_files_by_days(feature_files, fast_tail_days)
        label_files = tail_files_by_days(label_files, fast_tail_days)
        print(
            "FAST_TAIL symbol={} days={} features_files={} labels_files={}".format(
                symbol, fast_tail_days, len(feature_files), len(label_files)
            )
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


def parse_train_grid(value: str) -> list[int]:
    items: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return sorted(set(items))


def fit_model(
    x: pd.DataFrame, y: pd.Series, calibrate: bool, symbol: str
) -> tuple[Pipeline, Pipeline | None, list[int]]:
    base_pipeline = build_pipeline("lbfgs")
    print(
        "MODEL_CONFIG symbol={} solver=lbfgs max_iter=4000 scaler=StandardScaler rows={}".format(
            symbol, len(x)
        )
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
            "MODEL_CONFIG symbol={} solver=saga max_iter=4000 scaler=StandardScaler rows={}".format(
                symbol, len(x)
            )
        )
        base_pipeline.fit(x, y)
    calibrated = None
    if calibrate:
        tscv = TimeSeriesSplit(n_splits=3)
        calibrated = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=tscv)
        calibrated.fit(x, y)
    classifier = base_pipeline.named_steps["classifier"]
    classes = list(getattr(classifier, "classes_", []))
    return base_pipeline, calibrated, classes


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
    last_eval_best_train_rows: int | None = None,
    last_eval_acc_hi: float | None = None,
    last_eval_coverage: float | None = None,
    last_eval_acc_all: float | None = None,
    last_eval_test_rows: int | None = None,
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
        "lastEvalBestTrainRows": last_eval_best_train_rows,
        "lastEvalAccHi": last_eval_acc_hi,
        "lastEvalCoverage": last_eval_coverage,
        "lastEvalAccAll": last_eval_acc_all,
        "lastEvalTestRows": last_eval_test_rows,
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


def main() -> None:
    args = parse_args()
    train_sizes = parse_train_grid(args.train_grid)
    if not train_sizes:
        raise RuntimeError("train-grid must include at least one train size")
    max_train = max(train_sizes)
    test_rows = args.test_rows
    if test_rows <= 0:
        raise RuntimeError("test-rows must be > 0")
    fast_tail_days = None
    if args.fast_tail:
        need = max_train + test_rows + 10000
        fast_tail_days = int(np.ceil(need / 288.0)) + 5
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
            fast_tail=args.fast_tail,
            fast_tail_days=fast_tail_days,
        )
        if features.empty or labels.empty:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows=0 min={args.min_rows_per_symbol}")
            continue
        x, y, merged, feature_order = build_training_frame(features, labels)
        frame = merged.sort_values("closeTimeMs")
        frame = frame[frame["windowReady"] == True]
        x_all = x.loc[frame.index]
        y_all = y.loc[frame.index]
        if len(frame) < args.min_rows_per_symbol:
            print(f"SKIP_SYMBOL_NOT_ENOUGH_DATA symbol={symbol} rows={len(frame)} min={args.min_rows_per_symbol}")
            continue
        need_rows = max_train + test_rows
        if len(frame) < need_rows:
            print(
                "SKIP_EVAL_NOT_ENOUGH_ROWS symbol={} rows={} need={}".format(
                    symbol, len(frame), need_rows
                )
            )
            continue
        frame_tail = frame.tail(need_rows)
        x_tail = x_all.loc[frame_tail.index]
        y_tail = y_all.loc[frame_tail.index]
        test_start = len(frame_tail) - test_rows
        test_end = len(frame_tail)
        results: list[dict[str, float | int]] = []
        for train_size in train_sizes:
            train_start = len(frame_tail) - test_rows - train_size
            if train_start < 0:
                print(
                    "SKIP_EVAL_NOT_ENOUGH_ROWS symbol={} rows={} need={}".format(
                        symbol, len(frame_tail), train_size + test_rows
                    )
                )
                continue
            x_train = x_tail.iloc[train_start:test_start]
            y_train = y_tail.iloc[train_start:test_start]
            x_test = x_tail.iloc[test_start:test_end]
            y_test = y_tail.iloc[test_start:test_end]
            base_pipeline, calibrated, classes = fit_model(x_train, y_train, args.calibrate, symbol)
            model_for_eval = calibrated if calibrated is not None else base_pipeline
            if not classes:
                raise RuntimeError(f"No classes_ found for symbol {symbol}")
            if 1 not in classes:
                raise RuntimeError(f"UP class (1) missing in classes for symbol {symbol}: {classes}")
            up_class_index = classes.index(1)
            y_pred = model_for_eval.predict(x_test)
            proba = model_for_eval.predict_proba(x_test)
            pUp = proba[:, up_class_index]
            confidence = np.maximum(pUp, 1 - pUp)
            mask = confidence >= 0.55
            coverage = float(np.mean(mask)) if len(mask) > 0 else 0.0
            acc_all = float(np.mean(y_pred == y_test))
            tp = int(np.sum((y_pred == 1) & (y_test == 1)))
            tn = int(np.sum((y_pred == 0) & (y_test == 0)))
            fp = int(np.sum((y_pred == 1) & (y_test == 0)))
            fn = int(np.sum((y_pred == 0) & (y_test == 1)))
            up_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            down_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            up_rate_test = float(np.mean(y_test))
            if mask.any():
                acc_highconf = float(np.mean(y_pred[mask] == y_test[mask]))
            else:
                acc_highconf = float("nan")
                print(f"WARN_EVAL_EMPTY_MASK symbol={symbol} train={train_size}")
            print(
                "EVAL symbol={} train={} test={} acc={:.3f} upRec={:.3f} downRec={:.3f} coverage={:.3f} "
                "accHi={:.3f} tp={} tn={} fp={} fn={} upRateTest={:.3f}".format(
                    symbol,
                    train_size,
                    test_rows,
                    acc_all,
                    up_rec,
                    down_rec,
                    coverage,
                    acc_highconf if not np.isnan(acc_highconf) else float("nan"),
                    tp,
                    tn,
                    fp,
                    fn,
                    up_rate_test,
                )
            )
            results.append(
                {
                    "train_size": train_size,
                    "acc_all": acc_all,
                    "coverage": coverage,
                    "acc_hi": acc_highconf,
                }
            )
        if not results:
            continue
        valid_hi = [item for item in results if not np.isnan(item["acc_hi"])]
        if valid_hi:
            def _score(item: dict[str, float | int]) -> tuple[float, float, float]:
                acc_hi = float(item["acc_hi"])
                coverage = float(item["coverage"])
                if coverage < 0.02:
                    acc_hi -= 1.0
                return acc_hi, coverage, float(item["acc_all"])
            best = max(valid_hi, key=_score)
        else:
            best = max(results, key=lambda item: float(item["acc_all"]))
        best_train_size = int(best["train_size"])
        best_acc_hi = float(best["acc_hi"]) if not np.isnan(best["acc_hi"]) else float("nan")
        best_coverage = float(best["coverage"])
        best_acc_all = float(best["acc_all"])
        print(
            "BEST_EVAL symbol={} bestTrain={} bestAccHi={:.3f} bestCoverage={:.3f} bestAcc={:.3f}".format(
                symbol,
                best_train_size,
                best_acc_hi if not np.isnan(best_acc_hi) else float("nan"),
                best_coverage,
                best_acc_all,
            )
        )
        if args.eval_only:
            continue
        train_start = len(frame_tail) - test_rows - best_train_size
        x_train = x_tail.iloc[train_start:test_start]
        y_train = y_tail.iloc[train_start:test_start]
        train_frame = frame_tail.iloc[train_start:test_start]
        up_mask = train_frame["labelUp"] == 1
        down_mask = train_frame["labelUp"] == 0
        mean_ret_up = None
        mean_ret_down = None
        if up_mask.any():
            mean_ret_up = float(train_frame.loc[up_mask, "futureRet_1"].mean())
        if down_mask.any():
            mean_ret_down = float(train_frame.loc[down_mask, "futureRet_1"].mean())
        n_up = int(up_mask.sum())
        n_down = int(down_mask.sum())
        total_labels = n_up + n_down
        up_rate = float(n_up / total_labels) if total_labels > 0 else 0.0
        base_pipeline, calibrated, classes = fit_model(x_train, y_train, args.calibrate, symbol)
        model_for_stats = calibrated if calibrated is not None else base_pipeline
        if not classes:
            raise RuntimeError(f"No classes_ found for symbol {symbol}")
        if 1 not in classes:
            raise RuntimeError(f"UP class (1) missing in classes for symbol {symbol}: {classes}")
        up_class_index = classes.index(1)
        pUp = model_for_stats.predict_proba(x_train)[:, up_class_index]
        pUp_min = float(np.min(pUp))
        pUp_max = float(np.max(pUp))
        pUp_mean = float(np.mean(pUp))
        print(f"PUP_STATS symbol={symbol} min={pUp_min:.6f} max={pUp_max:.6f} mean={pUp_mean:.6f}")
        if pUp_max < 0.05 or pUp_mean < 0.01:
            print(
                "WARN_PUP_DEGENERATE symbol={} max={:.6f} mean={:.6f} "
                "reason=class_index_or_features_or_nan_issue".format(symbol, pUp_max, pUp_mean)
            )
        model_version = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        model_dir = args.out_dir / symbol / model_version
        export_model = calibrated if calibrated is not None else base_pipeline
        try:
            input_names, output_names = export_onnx(
                export_model, x_train.shape[1], model_dir / "model.onnx"
            )
            print(f"ONNX_EXPORT symbol={symbol} inputs={input_names} outputs={output_names}")
        except Exception as exc:
            print(f"ONNX_EXPORT_FAILED symbol={symbol} calibrated failed ({exc}); fallback: export non-calibrated.")
            export_model = base_pipeline
            input_names, output_names = export_onnx(
                export_model, x_train.shape[1], model_dir / "model.onnx"
            )
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
                x_check = x_train.iloc[:256].to_numpy(dtype=np.float32)
                outputs = sess.run(None, {input_name: x_check})
                if outputs:
                    onnx_probs = np.array(outputs[0])
                    if onnx_probs.ndim >= 2 and onnx_probs.shape[1] > up_class_index:
                        pUp_onnx = onnx_probs[:, up_class_index]
                        mean_onnx = float(np.mean(pUp_onnx))
                        print(
                            f"ONNX_CHECK symbol={symbol} meanSklearn={pUp_mean:.6f} meanOnnx={mean_onnx:.6f}"
                        )
                        if abs(pUp_mean - mean_onnx) > 1e-2:
                            print(f"WARN_ONNX_MISMATCH symbol={symbol} diff={abs(pUp_mean - mean_onnx):.6f}")
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
            len(x_train),
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
            best_train_size,
            best_acc_hi,
            best_coverage,
            best_acc_all,
            test_rows,
        )
        write_current(model_dir, args.out_dir, symbol)
        wrote_path = args.out_dir / symbol / "current"
        print(
            "TRAIN_SYMBOL symbol={} rows={} upRate={:.6f} calibrated={} wrote={}".format(
                symbol,
                len(x_train),
                up_rate,
                args.calibrate,
                wrote_path,
            )
        )


if __name__ == "__main__":
    main()
