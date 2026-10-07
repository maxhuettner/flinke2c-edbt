from __future__ import annotations

import math
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast, Callable
from statistics import NormalDist
from collections.abc import Mapping, Sequence

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from pandas.errors import EmptyDataError

from constants import exp_name_mapping
from graph_utils import process_operator_diagram, process_flink_dag
from graphviz import Digraph


class NesRepetitionData(TypedDict):
    throughput: int


class NesExperimentData(TypedDict):
    avg_throughput: float
    repetitions: dict[str, NesRepetitionData]
    exp_key: NotRequired[str]


class NebulaRepetitionData(TypedDict):
    throughput: float
    source_tps: NotRequired[float]
    latency_seconds: float
    latency_ms: float
    total_events: int
    source_start: pd.Timestamp
    sink_end: pd.Timestamp
    sink_end_from_sink: NotRequired[bool]
    event_throughput: list[float]
    event_throughput_time: NotRequired[list[pd.Timestamp]]


class NebulaExperimentData(TypedDict):
    avg_throughput: float
    avg_latency_seconds: float
    repetitions: dict[str, NebulaRepetitionData]
    exp_key: NotRequired[str]


class NmRepetitionData(TypedDict):
    runtime: int
    throughput: int
    operator_graph: Digraph | None


class NmExperimentData(TypedDict):
    avg_throughput: float
    avg_runtime: float
    repetitions: dict[str, NmRepetitionData]
    dag: NotRequired[Digraph | None]
    exp_key: NotRequired[str]


class StreamingRepetitionData(TypedDict):
    runtime: int
    source_throughput: Series
    avg_source_throughput: float
    sink_throughput: Series
    avg_sink_throughput: float
    operator_graph: Digraph | None


class ExperimentData(TypedDict):
    avg_avg_source_throughput: float
    avg_avg_sink_throughput: float
    avg_runtime: float
    repetitions: dict[str, StreamingRepetitionData]
    dag: NotRequired[Digraph | None]
    exp_key: NotRequired[str]


ExperimentSummary = ExperimentData | NmExperimentData | NesExperimentData | NebulaExperimentData
ExperimentWithDag = ExperimentData | NmExperimentData
OperatorRepetition = NmRepetitionData | StreamingRepetitionData


class PlotRow(TypedDict):
    exp_name: str
    value: float


def process_source_throughput(repetition_path: Path | str) -> Series:
    repetition_dir = Path(repetition_path)
    source_file = next(repetition_dir.glob("benchmark_*.parquet"))
    source = pd.read_parquet(source_file, columns=["time", "rate"])
    source["time"] = pd.to_datetime(source["time"])

    return source["rate"]


def process_sink_throughput(repetition_path: Path | str) -> Series:
    repetition_dir = Path(repetition_path)
    sink_file = next(repetition_dir.glob("sink_*.parquet"))
    sink = pd.read_parquet(sink_file, columns=["ts", "message"])
    sink["ts_seconds"] = pd.to_datetime(sink["ts"]).dt.floor('s')

    sink_tput = sink.groupby("ts_seconds").count().reset_index()
    return sink_tput["message"]


def process_tput_and_runtime(repetition_path: Path | str) -> tuple[int, int]:
    repetition_dir = Path(repetition_path)
    with open(repetition_dir / "nexmark.txt", "r", encoding='utf8') as f:
        for line in f.read().splitlines():
            if line.strip().startswith("|Total"):
                parts = [col.strip() for col in line.strip('|').split('|')]
                events_raw = parts[1].replace(",", "")
                time_s = float(parts[3])

                if not events_raw or events_raw.lower() == "nan":
                    raise ValueError("Events Num is missing in nexmark.txt Total row")

                events_num = float(events_raw)
                if not math.isfinite(events_num) or events_num <= 0:
                    raise ValueError(f"Invalid Events Num value: {parts[1]}")

                if not math.isfinite(time_s) or time_s <= 0:
                    raise ValueError(f"Invalid Time(s) value: {parts[3]}")

                throughput = int(events_num / time_s)
                time_ms = round(time_s * 1000)

                return throughput, time_ms

    raise ValueError("No Total line found in nexmark.txt")


def process_runtime(repetition_path: Path | str) -> int:
    repetition_dir = Path(repetition_path)
    with open(repetition_dir / "runtime.txt", "r", encoding='utf8') as f:
        return int(f.read())

def process_tput_nes(repetition_path: Path) -> int:
    with open(repetition_path / "tps.txt", "r", encoding='utf8') as f:
        return int(f.read())


def process_repetitions_nes(exp_path: Path) -> NesExperimentData:
    repetitions: dict[str, NesRepetitionData] = {}
    for repetition_path in exp_path.iterdir():
        if repetition_path.is_dir():
            repetitions[repetition_path.name] = {'throughput': process_tput_nes(repetition_path)}

    avg_throughput = pd.Series([rep["throughput"] for rep in repetitions.values()]).mean()

    return {
        "avg_throughput": avg_throughput,
        "repetitions": repetitions,
    }


def _find_io_dir(exp_path: Path, dirname: str) -> Path:
    direct_path = exp_path / dirname
    if direct_path.is_dir():
        return direct_path

    logs_path = exp_path / "logs" / dirname
    if logs_path.is_dir():
        return logs_path

    candidates = sorted(
        [p for p in exp_path.glob(f"**/{dirname}") if p.is_dir()],
        key=lambda p: len(p.parts)
    )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Could not locate '{dirname}' directory in {exp_path}")


def _find_io_dirs(exp_path: Path, dirname: str) -> list[Path]:
    direct_path = exp_path / dirname
    if direct_path.is_dir():
        return [direct_path]

    logs_path = exp_path / "logs" / dirname
    if logs_path.is_dir():
        return [logs_path]

    candidates = sorted(
        [p for p in exp_path.glob(f"**/{dirname}") if p.is_dir()],
        key=lambda p: (len(p.parts), p.as_posix())
    )

    if candidates:
        return candidates

    raise FileNotFoundError(f"Could not locate '{dirname}' directory in {exp_path}")


def _resolve_exp_path(base_dir: Path, exp_paths: dict[str, Path], exp_name: str) -> Path | None:
    if "/" in exp_name or "\\" in exp_name:
        candidate = base_dir / exp_name
        if candidate.is_dir():
            return candidate

    exp_path = exp_paths.get(exp_name)
    if exp_path is not None:
        return exp_path

    candidate = base_dir / exp_name
    if candidate.is_dir():
        return candidate

    for candidate in base_dir.rglob(exp_name):
        if candidate.is_dir():
            return candidate

    return None


def _list_latency_files(sink_dir: Path) -> list[Path]:
    files_by_stem: dict[str, Path] = {}
    for pattern in ("latency_test_*.parquet", "latency_*.parquet"):
        for file_path in sink_dir.rglob(pattern):
            files_by_stem.setdefault(file_path.stem, file_path)
    return sorted(files_by_stem.values(), key=lambda path: path.as_posix())


def _read_parquet_dataframe(file_path: Path, columns: list[str] | None = None) -> DataFrame | None:
    try:
        import polars as pl  # type: ignore
    except Exception:
        pl = None

    if pl is not None:
        try:
            df = pl.read_parquet(file_path, columns=columns).to_pandas()
            if df.empty:
                return None
            return df
        except Exception:
            pass

    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pq = None

    if pq is not None:
        try:
            table = pq.read_table(file_path, columns=columns, use_pandas_metadata=False)
            df = table.to_pandas()
            if df.empty:
                return None
            return df
        except Exception:
            pass

    try:
        df = pd.read_parquet(file_path, columns=columns)
    except Exception:
        return None

    if df.empty:
        return None
    return df


def _read_tabular_dataframe(file_path: Path, columns: list[str] | None = None) -> DataFrame | None:
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return _read_parquet_dataframe(file_path, columns=columns)

    if suffix != ".csv":
        return None

    try:
        if columns is None:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, usecols=lambda col: col in columns)
    except (FileNotFoundError, EmptyDataError):
        return None
    except ValueError:
        # Some CSV readers raise when requested columns do not exist; fall back
        # to reading all columns and filtering later.
        try:
            df = pd.read_csv(file_path)
        except Exception:
            return None
    except Exception:
        return None

    if df.empty:
        return None

    return df


def _read_time_file(file_path: Path) -> DataFrame | None:
    return _read_tabular_dataframe(file_path)


def _extract_rep_id(path: Path) -> str | None:
    stem = path.stem
    if "_" not in stem:
        return None

    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None

    return parts[1]


def _read_events_file(file_path: Path, *, allow_nes_rate: bool = False) -> DataFrame | None:
    df = _read_tabular_dataframe(file_path)
    if allow_nes_rate and df is not None and "rate" not in df and "tuples_this_second" in df:
        df = df.rename(columns={"tuples_this_second": "rate"})
    if df is None or df.empty or "rate" not in df:
        return None

    time_col_candidates = ("time", "timestamp", "ts")
    time_col = next((col for col in time_col_candidates if col in df), None)
    if time_col is None:
        return None

    events_df = df[[time_col, "rate"]].copy()
    events_df[time_col] = pd.to_datetime(events_df[time_col], errors="coerce")
    events_df["rate"] = pd.to_numeric(events_df["rate"], errors="coerce")
    events_df = events_df.dropna(subset=[time_col, "rate"])

    if events_df.empty:
        return None

    events_df = events_df.rename(columns={time_col: "time"})
    return events_df.sort_values("time")


def _collect_rep_files(io_dir: Path, file_prefix: str, rep_id: str | None = None) -> list[Path]:
    stem_pattern = f"{file_prefix}_*_{rep_id}" if rep_id is not None else f"{file_prefix}_*_*"
    files_by_stem: dict[str, Path] = {}

    # Prefer CSV for topology datasets but fall back to parquet when needed.
    for ext in (".csv", ".parquet"):
        for file_path in io_dir.rglob(f"{stem_pattern}{ext}"):
            files_by_stem.setdefault(file_path.stem, file_path)

    return sorted(files_by_stem.values(), key=lambda path: path.as_posix())


def process_repetitions_topo(
    exp_path: Path,
    use_latency_based_tps: bool = False,
) -> NebulaExperimentData:
    source_dirs = _find_io_dirs(exp_path, "source")
    sink_dir = _find_io_dir(exp_path, "sink")

    repetition_ids: set[str] = set()
    for source_dir in source_dirs:
        for time_file in _collect_rep_files(source_dir, "time"):
            rep_id = _extract_rep_id(time_file)
            if rep_id is not None:
                repetition_ids.add(rep_id)

    repetitions: dict[str, NebulaRepetitionData] = {}
    throughputs: list[float] = []
    latencies: list[float] = []

    for rep_id in sorted(repetition_ids, key=lambda value: (len(value), value)):
        start_times: list[pd.Timestamp] = []
        source_end_times: list[pd.Timestamp] = []
        total_events = 0.0
        source_tps_total = 0.0
        has_source_tps = False
        events_frames: list[DataFrame] = []

        for source_dir in source_dirs:
            for time_file in _collect_rep_files(source_dir, "time", rep_id):
                df = _read_time_file(time_file)
                if df is None:
                    continue

                start_col = "first_elem_time" if "first_elem_time" in df else "start_time" if "start_time" in df else None
                if start_col is not None:
                    start_series = pd.to_datetime(df[start_col], errors="coerce").dropna()
                    if not start_series.empty:
                        start_times.append(start_series.min())

                source_end_col = "end_time" if "end_time" in df else "last_elem_time" if "last_elem_time" in df else None
                if source_end_col is not None:
                    source_end_series = pd.to_datetime(df[source_end_col], errors="coerce").dropna()
                    if not source_end_series.empty:
                        source_end_times.append(source_end_series.max())

                if "num_events" in df:
                    total_events += pd.to_numeric(df["num_events"], errors="coerce").fillna(0).sum()

                time_tps = _read_time_throughput(time_file)
                if time_tps is not None and math.isfinite(time_tps) and time_tps > 0:
                    source_tps_total += time_tps
                    has_source_tps = True

        for events_file in _collect_rep_files(sink_dir, "events", rep_id):
            events_df = _read_events_file(events_file)
            if events_df is not None:
                events_frames.append(events_df)

        if not start_times:
            continue

        earliest_start = min(start_times)

        end_times: list[pd.Timestamp] = []
        for time_file in _collect_rep_files(sink_dir, "time", rep_id):
            df = _read_time_file(time_file)
            if df is None:
                continue

            end_col = "end_time" if "end_time" in df else "last_elem_time" if "last_elem_time" in df else None
            if end_col is None:
                continue

            end_series = pd.to_datetime(df[end_col], errors="coerce").dropna()
            if not end_series.empty:
                end_times.append(end_series.max())

        if end_times:
            latest_end = max(end_times)
            sink_end_from_sink = True
        elif source_end_times:
            # Keep source-valid repetitions even when sink timing logs are missing.
            latest_end = max(source_end_times)
            sink_end_from_sink = False
        else:
            continue
        latency = latest_end - earliest_start
        latency_seconds = latency.total_seconds()

        if latency_seconds <= 0:
            continue

        total_events_int = int(total_events)
        latency_based_tps = total_events_int / latency_seconds if total_events_int else 0.0
        if use_latency_based_tps:
            throughput = latency_based_tps
        else:
            throughput = source_tps_total if has_source_tps else latency_based_tps

        if events_frames:
            combined_events = pd.concat(events_frames, ignore_index=True)
            combined_events = combined_events.groupby("time", as_index=False)["rate"].sum()
            combined_events = combined_events.sort_values("time")
            event_throughput = combined_events["rate"].tolist()
            event_throughput_time = combined_events["time"].tolist()
        else:
            event_throughput = []
            event_throughput_time = []

        repetitions[f"rep_{rep_id}"] = {
            "throughput": throughput,
            "source_tps": source_tps_total if has_source_tps else throughput,
            "latency_seconds": latency_seconds,
            "latency_ms": latency_seconds * 1000,
            "total_events": total_events_int,
            "source_start": earliest_start,
            "sink_end": latest_end,
            "sink_end_from_sink": sink_end_from_sink,
            "event_throughput": event_throughput,
            "event_throughput_time": event_throughput_time,
        }

        throughputs.append(throughput)
        latencies.append(latency_seconds)

    avg_throughput = pd.Series(throughputs).mean() if throughputs else float("nan")
    avg_latency = pd.Series(latencies).mean() if latencies else float("nan")

    return {
        "avg_throughput": avg_throughput,
        "avg_latency_seconds": avg_latency,
        "repetitions": repetitions,
    }


def _latency_unit_to_ns_scale(unit: str) -> float:
    unit_key = unit.strip().lower()
    unit_map = {
        "ns": 1.0,
        "nanosecond": 1.0,
        "nanoseconds": 1.0,
        "us": 1e3,
        "microsecond": 1e3,
        "microseconds": 1e3,
        "ms": 1e6,
        "millisecond": 1e6,
        "milliseconds": 1e6,
        "s": 1e9,
        "sec": 1e9,
        "second": 1e9,
        "seconds": 1e9,
    }
    if unit_key not in unit_map:
        raise ValueError(f"Unsupported latency unit '{unit}'. Expected one of: {', '.join(sorted(unit_map))}.")
    return unit_map[unit_key]


def _latency_col_to_ns_scale(col_name: str) -> float:
    col = col_name.lower()
    if "ns" in col:
        return 1.0
    if "us" in col or "micro" in col:
        return 1e3
    if "ms" in col or "msec" in col:
        return 1e6
    if "sec" in col or col.endswith("_s") or col.endswith("s"):
        return 1e9
    return 1.0


def _read_latency_events_file(
    file_path: Path,
    sample_stride: int | None = None,
    max_rows: int | None = None,
    sample_seed: int = 0,
    input_latency_unit: str | None = None,
) -> DataFrame | None:
    try:
        import polars as pl  # type: ignore
    except Exception:
        pl = None

    if pl is not None:
        try:
            lf = pl.scan_parquet(file_path)
            columns = lf.collect_schema().names()

            latency_col = "latency_ns" if "latency_ns" in columns else None
            if latency_col is None:
                latency_col = next((col for col in columns if "latency" in col.lower()), None)
            if latency_col is None:
                return None
            latency_scale = _latency_unit_to_ns_scale(input_latency_unit) if input_latency_unit else _latency_col_to_ns_scale(latency_col)

            time_col_candidates = ("time", "timestamp", "ts", "event_time", "event_ts")
            time_col = next((col for col in time_col_candidates if col in columns), None)
            if time_col is None:
                for col in columns:
                    if col == latency_col:
                        continue
                    col_lower = col.lower()
                    if "time" in col_lower or col_lower.endswith("ts") or col_lower.endswith("_ts"):
                        time_col = col
                        break

            sample_prob = None
            if sample_stride is not None and sample_stride > 1:
                sample_prob = 1.0 / float(sample_stride)

            if max_rows is not None and max_rows > 0:
                lf = lf.sample(n=max_rows, seed=sample_seed)
            elif sample_prob is not None:
                lf = lf.sample(frac=sample_prob, seed=sample_seed)

            select_cols = [latency_col] + ([time_col] if time_col else [])
            df = lf.select(select_cols).collect(streaming=True).to_pandas()
            if df.empty:
                return None
        except Exception:
            df = None
    else:
        df = None

    if df is None:
        df = _read_parquet_dataframe(file_path)
        if df is None or df.empty:
            return None

        if max_rows is not None and max_rows > 0 and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=sample_seed)
        elif sample_stride is not None and sample_stride > 1:
            sample_prob = 1.0 / float(sample_stride)
            if 0 < sample_prob < 1:
                df = df.sample(frac=sample_prob, random_state=sample_seed)

    latency_col = "latency_ns"
    if latency_col not in df.columns:
        fallback_col = next((col for col in df.columns if "latency" in col.lower()), None)
        if fallback_col is None:
            return None
        latency_col = fallback_col

    latency_scale = _latency_unit_to_ns_scale(input_latency_unit) if input_latency_unit else _latency_col_to_ns_scale(latency_col)
    if latency_col != "latency_ns":
        df = df.rename(columns={latency_col: "latency_ns"})
        latency_col = "latency_ns"

    time_col_candidates = ("time", "timestamp", "ts", "event_time", "event_ts")
    time_col = next((col for col in time_col_candidates if col in df.columns), None)
    if time_col is None:
        for col in df.columns:
            if col == latency_col:
                continue
            col_lower = col.lower()
            if "time" in col_lower or col_lower.endswith("ts") or col_lower.endswith("_ts"):
                time_col = col
                break

    if time_col is None:
        out = df[[latency_col]].copy()
        out[latency_col] = pd.to_numeric(out[latency_col], errors="coerce") * latency_scale
        out = out.dropna(subset=[latency_col])
        if out.empty:
            return None
        return out

    out = df[[time_col, latency_col]].copy()
    if np.issubdtype(out[time_col].dtype, np.number):
        median_val = float(np.nanmedian(out[time_col].to_numpy()))
        if median_val > 1e14:
            unit = "ns"
        elif median_val > 1e11:
            unit = "ms"
        elif median_val > 1e9:
            unit = "s"
        else:
            unit = "s"
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce", unit=unit)
    else:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[latency_col] = pd.to_numeric(out[latency_col], errors="coerce") * latency_scale
    out = out.dropna(subset=[time_col, latency_col])
    if out.empty:
        return None
    out = out.rename(columns={time_col: "time"})
    return out.sort_values("time")


def _read_latency_file(
    file_path: Path,
    sample_stride: int | None = None,
    max_rows: int | None = None,
    sample_seed: int = 0,
    input_latency_unit: str | None = None,
) -> Series | None:
    df = _read_latency_events_file(
        file_path,
        sample_stride=sample_stride,
        max_rows=max_rows,
        sample_seed=sample_seed,
        input_latency_unit=input_latency_unit,
    )
    if df is None or df.empty or "latency_ns" not in df.columns:
        return None
    series = pd.to_numeric(df["latency_ns"], errors="coerce").dropna()
    if series.empty:
        return None
    return series


def _read_numeric_series(
    file_path: Path | str,
    column: str | None = None,
    sample_stride: int | None = None,
    max_rows: int | None = None,
    sample_seed: int = 0,
) -> Series | None:
    path = Path(file_path)
    if not path.exists():
        return None

    # Fast path: use Polars if available (faster Parquet scan + random sampling).
    try:
        import polars as pl  # type: ignore
    except Exception:
        pl = None

    if pl is not None:
        try:
            lf = pl.scan_parquet(path)

            sample_prob = None
            if sample_stride is not None and sample_stride > 1:
                # Interpret stride as a random sampling ratio (1/stride).
                sample_prob = 1.0 / float(sample_stride)
            sample_size = max_rows if max_rows is not None and max_rows > 0 else None

            if sample_size is not None:
                lf = lf.sample(n=sample_size, seed=sample_seed)
            elif sample_prob is not None:
                lf = lf.sample(frac=sample_prob, seed=sample_seed)

            if column is not None:
                lf = lf.select(column)

            values_df = lf.collect(streaming=True)
            if values_df.is_empty():
                return None
            if column is None:
                col_name = values_df.columns[0]
            else:
                col_name = column if column in values_df.columns else values_df.columns[0]
            values = values_df[col_name].to_numpy()
            if values.size == 0:
                return None
            return pd.Series(values, name=col_name)
        except Exception:
            # Fall back to pandas below on any Polars failures.
            pass

    # PyArrow direct column read (avoids unsupported columns like string_view).
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pq = None

    if pq is not None:
        try:
            pf = pq.ParquetFile(path)
            schema_names = pf.schema.names
            col_name = column
            if col_name is not None and col_name not in schema_names:
                col_name = next((c for c in schema_names if column in c), None)
            if col_name is None:
                col_name = schema_names[0] if schema_names else None
            if col_name is None:
                return None

            table = pf.read(columns=[col_name])
            values = table.column(0).to_numpy()
            if values.size == 0:
                return None
            series = pd.Series(values, name=col_name)

            if max_rows is not None and max_rows > 0 and len(series) > max_rows:
                series = series.sample(n=max_rows, random_state=sample_seed)
            elif sample_stride is not None and sample_stride > 1:
                sample_prob = 1.0 / float(sample_stride)
                if 0 < sample_prob < 1:
                    series = series.sample(frac=sample_prob, random_state=sample_seed)
            return series
        except Exception:
            pass

    try:
        cols = [column] if column is not None else None
        df = _read_parquet_dataframe(path, columns=cols)
    except Exception:
        df = None

    if df is None or df.empty:
        return None

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=sample_seed)
    elif sample_stride is not None and sample_stride > 1:
        sample_prob = 1.0 / float(sample_stride)
        if 0 < sample_prob < 1:
            df = df.sample(frac=sample_prob, random_state=sample_seed)

    col = column if column in df.columns else df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None

    return series


def _read_time_throughput(file_path: Path) -> float | None:
    df = _read_time_file(file_path)
    if df is None or df.empty:
        return None

    for col in ("tps", "throughput", "rate", "avg_tps", "avg_rate"):
        if col in df:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                return None
            return float(series.mean())

    return None


def _read_events_throughput(file_path: Path, discard_first_and_last: bool = True) -> float | None:
    events_df = _read_events_file(file_path)
    if events_df is None or events_df.empty:
        return None

    series = pd.to_numeric(events_df["rate"], errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return None

    if discard_first_and_last and len(series) > 2:
        series = series.iloc[1:-1]

    if series.empty:
        return None

    return float(series.mean())


def _compute_boxplot_stats(values: Sequence[float]) -> dict[str, Any]:
    series = pd.Series(list(values)).dropna()
    if series.empty:
        return {}

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    median = float(series.quantile(0.5))
    min_val = float(series.min())
    max_val = float(series.max())

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    lower_candidates = series[series >= lower_bound]
    upper_candidates = series[series <= upper_bound]

    # Mirror matplotlib's cbook.boxplot_stats: a whisker must never cross past
    # its own quartile. With small samples the extreme point can be the only
    # one below the lower fence (or above the upper fence); excluding it as
    # an outlier would otherwise leave the whisker sitting inside the box.
    if lower_candidates.empty or lower_candidates.min() > q1:
        lower_whisker = q1
    else:
        lower_whisker = float(lower_candidates.min())

    if upper_candidates.empty or upper_candidates.max() < q3:
        upper_whisker = q3
    else:
        upper_whisker = float(upper_candidates.max())

    outlier_values = series[(series < lower_whisker) | (series > upper_whisker)].sort_values()

    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": median,
        "q1": q1,
        "q3": q3,
        "min": min_val,
        "max": max_val,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "outliers": [float(v) for v in outlier_values.tolist()],
    }


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\\textbackslash{}",
        "_": r"\\_",
        "%": r"\\%",
        "&": r"\\&",
        "#": r"\\#",
        "{": r"\\{",
        "}": r"\\}",
        "$": r"\\$",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("|", r"\\textbar{}")
    text = text.replace("\n", r"\\")
    return text


def _normalize_latency_unit(unit: str) -> tuple[str, float]:
    unit_key = unit.strip().lower()
    unit_map = {
        "ns": 1.0,
        "nanosecond": 1.0,
        "nanoseconds": 1.0,
        "us": 1e3,
        "microsecond": 1e3,
        "microseconds": 1e3,
        "ms": 1e6,
        "millisecond": 1e6,
        "milliseconds": 1e6,
        "s": 1e9,
        "sec": 1e9,
        "second": 1e9,
        "seconds": 1e9,
    }

    if unit_key not in unit_map:
        raise ValueError(f"Unsupported latency unit '{unit}'. Expected one of: {', '.join(sorted(unit_map))}.")

    return unit_key, unit_map[unit_key]


def _infer_event_step_seconds(times: Sequence[pd.Timestamp]) -> float:
    if len(times) < 2:
        return 1.0

    series = pd.to_datetime(pd.Series(times), errors="coerce").dropna()
    if len(series) < 2:
        return 1.0

    deltas = series.diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 1.0

    q50 = float(deltas.quantile(0.5))
    q90 = float(deltas.quantile(0.9))

    # Some logs contain many near-duplicate timestamps (microseconds) mixed with
    # regular reporting cadence (~1s). A plain median can collapse to the
    # microsecond deltas and explode gap filling. Detect that bimodal shape and
    # prefer the coarse cadence.
    if q50 > 0 and q90 / q50 >= 100:
        step = q90
    else:
        step = q50

    if not math.isfinite(step) or step <= 0:
        return 1.0

    return float(np.round(step, 6))


def _expand_event_throughput(
    times: Sequence[pd.Timestamp],
    values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return time offsets (s) and values with gaps filled by zeroes."""
    if times is None or values is None:
        return np.array([]), np.array([]), 1.0

    if len(times) == 0 or len(values) == 0:
        return np.array([]), np.array([]), 1.0

    if len(times) != len(values):
        values_arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
        x = np.arange(len(values_arr), dtype=float)
        return x, values_arr, 1.0

    df = pd.DataFrame({
        "time": pd.to_datetime(pd.Series(times), errors="coerce"),
        "value": pd.to_numeric(pd.Series(values), errors="coerce"),
    }).dropna()
    if df.empty:
        return np.array([]), np.array([]), 1.0

    df = df.sort_values("time")
    step = _infer_event_step_seconds(df["time"].tolist())
    if step <= 0 or not math.isfinite(step):
        step = 1.0

    offsets = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
    sample_index = np.round(offsets / step).astype(int)

    value_series = pd.Series(df["value"].to_numpy()).groupby(sample_index).sum()
    if value_series.empty:
        return np.array([]), np.array([]), step

    full_index = np.arange(int(value_series.index.min()), int(value_series.index.max()) + 1)
    filled_values = value_series.reindex(full_index, fill_value=0.0).to_numpy()
    time_offsets = full_index.astype(float) * step

    return time_offsets, filled_values, step


def _compute_mean_ci(values: Sequence[float], ci: float = 0.95) -> tuple[float, float, float, int]:
    data = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    n = int(data.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0

    mean = float(np.mean(data))
    if n == 1:
        return mean, mean, mean, 1

    std = float(np.std(data, ddof=1))
    if std == 0.0:
        return mean, mean, mean, n

    if not (0 < ci < 1):
        raise ValueError("ci must be between 0 and 1.")

    z = NormalDist().inv_cdf(0.5 + ci / 2)
    half_width = z * std / math.sqrt(n)
    return mean, mean - half_width, mean + half_width, n


def _prepare_event_series(
    rep: Mapping[str, Any],
    smooth_window: int | None = None,
    discard_first_and_last_event_tput: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    event_tput = rep.get("event_throughput", [])
    event_times = rep.get("event_throughput_time")

    if event_times:
        # Fill gaps based on timestamps so long idle periods render as zeros.
        x_values, y_values, step = _expand_event_throughput(event_times, event_tput)
    else:
        y_values = np.array(event_tput, dtype=float)
        x_values = np.arange(len(y_values), dtype=float)
        step = 1.0

    if smooth_window is not None and smooth_window > 1 and len(y_values) > 0:
        series = pd.Series(y_values, dtype=float)
        y_values = series.rolling(window=smooth_window, center=True, min_periods=1).mean().to_numpy()

    if discard_first_and_last_event_tput and len(y_values) > 1:
        x_values = x_values[1:-1]
        y_values = y_values[1:-1]

    return x_values, y_values, step


def _prepare_latency_series(
    times: Sequence[pd.Timestamp] | None,
    values: Sequence[float],
    smooth_window: int | None = None,
    discard_first_and_last: bool = False,
    sample_stride: int | None = None,
    max_points: int | None = None,
    gap_fill: str = "zero",
) -> tuple[np.ndarray, np.ndarray, float]:
    if times is not None and len(times) == len(values):
        df = pd.DataFrame({
            "time": pd.to_datetime(pd.Series(times), errors="coerce"),
            "value": pd.to_numeric(pd.Series(values), errors="coerce"),
        }).dropna()
        if df.empty:
            return np.array([]), np.array([]), 1.0

        df = df.sort_values("time")
        step = _infer_event_step_seconds(df["time"].tolist())
        if step <= 0 or not math.isfinite(step):
            step = 1.0

        offsets = (df["time"] - df["time"].iloc[0]).dt.total_seconds().to_numpy()
        sample_index = np.round(offsets / step).astype(int)
        grouped = pd.Series(df["value"].to_numpy()).groupby(sample_index).mean()
        if grouped.empty:
            return np.array([]), np.array([]), step

        min_idx = int(grouped.index.min())
        max_idx = int(grouped.index.max())
        span = max_idx - min_idx + 1

        fill_value = 0.0 if gap_fill == "zero" else np.nan
        if max_points is not None and max_points > 0 and span > max_points:
            factor = int(math.ceil(span / max_points))
            idx_arr = grouped.index.to_numpy()
            bin_idx = ((idx_arr - min_idx) // factor).astype(int)
            bin_mean = pd.Series(grouped.to_numpy()).groupby(bin_idx).mean()
            num_bins = int(math.ceil(span / factor))
            y_values = bin_mean.reindex(np.arange(num_bins), fill_value=fill_value).to_numpy()
            x_values = (min_idx + np.arange(num_bins) * factor).astype(float) * step
            step = step * factor
            if smooth_window is not None and smooth_window > 1:
                series = pd.Series(y_values, dtype=float)
                y_values = series.rolling(window=smooth_window, center=True, min_periods=1).mean().to_numpy()
        else:
            full_index = np.arange(min_idx, max_idx + 1)
            # Fill timestamp gaps with zeroes to mirror throughput handling.
            series = grouped.reindex(full_index, fill_value=fill_value)
            if smooth_window is not None and smooth_window > 1:
                series = series.rolling(window=smooth_window, center=True, min_periods=1).mean()
            x_values = full_index.astype(float) * step
            y_values = series.to_numpy()
    else:
        if values is None or len(values) == 0:
            return np.array([]), np.array([]), 1.0

        series_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
        if series_values.size == 0:
            return np.array([]), np.array([]), 1.0

        step = 1.0
        stride = sample_stride if sample_stride is not None and sample_stride > 1 else None
        if max_points is not None and max_points > 0:
            stride = max(stride or 1, int(math.ceil(series_values.size / max_points)))
        if stride and stride > 1:
            series_values = series_values[::stride]

        if smooth_window is not None and smooth_window > 1 and series_values.size > 0:
            series = pd.Series(series_values, dtype=float)
            series_values = series.rolling(window=smooth_window, center=True, min_periods=1).mean().to_numpy()

        x_values = np.arange(series_values.size, dtype=float)
        y_values = series_values

    if discard_first_and_last and y_values.size > 1:
        x_values = x_values[1:-1]
        y_values = y_values[1:-1]

    return x_values, y_values, step

def process_repetitions_nebula(exp_path: Path, *, allow_sink_only: bool = False) -> NebulaExperimentData:
    source_dirs = _find_io_dirs(exp_path, "source")
    sink_dir = _find_io_dir(exp_path, "sink")

    repetition_ids: set[str] = set()
    for source_dir in source_dirs:
        for time_file in _collect_rep_files(source_dir, "time"):
            rep_id = _extract_rep_id(time_file)
            if rep_id is not None:
                repetition_ids.add(rep_id)

    repetitions: dict[str, NebulaRepetitionData] = {}
    throughputs: list[float] = []
    latencies: list[float] = []

    for rep_id in sorted(repetition_ids, key=lambda x: (len(x), x)):
        start_times: list[pd.Timestamp] = []
        total_events = 0.0
        source_tps_total = 0.0
        has_source_tps = False
        events_frames: list[DataFrame] = []

        source_end_times: list[pd.Timestamp] = []

        for source_dir in source_dirs:
            for time_file in _collect_rep_files(source_dir, "time", rep_id):
                df = _read_time_file(time_file)
                if df is None:
                    continue

                start_col = "first_elem_time" if "first_elem_time" in df else "start_time" if "start_time" in df else None
                if start_col is not None:
                    start_series = pd.to_datetime(df[start_col], errors="coerce").dropna()
                    if not start_series.empty:
                        start_times.append(start_series.min())

                source_end_col = "end_time" if "end_time" in df else "last_elem_time" if "last_elem_time" in df else None
                if source_end_col is not None:
                    source_end_series = pd.to_datetime(df[source_end_col], errors="coerce").dropna()
                    if not source_end_series.empty:
                        source_end_times.append(source_end_series.max())

                if "num_events" in df:
                    total_events += pd.to_numeric(df["num_events"], errors="coerce").fillna(0).sum()

                time_tps = _read_time_throughput(time_file)
                if time_tps is not None and math.isfinite(time_tps) and time_tps > 0:
                    source_tps_total += time_tps
                    has_source_tps = True

        for events_file in _collect_rep_files(sink_dir, "events", rep_id):
            events_df = _read_events_file(events_file)
            if events_df is not None:
                events_frames.append(events_df)

        if not start_times:
            continue

        earliest_start = min(start_times)

        end_times: list[pd.Timestamp] = []
        for time_file in _collect_rep_files(sink_dir, "time", rep_id):
            df = _read_time_file(time_file)
            if df is None:
                continue

            end_col = "end_time" if "end_time" in df else "last_elem_time" if "last_elem_time" in df else None
            if end_col is None:
                continue

            end_series = pd.to_datetime(df[end_col], errors="coerce").dropna()
            if not end_series.empty:
                end_times.append(end_series.max())

        if end_times:
            latest_end = max(end_times)
            sink_end_from_sink = True
        elif source_end_times:
            latest_end = max(source_end_times)
            sink_end_from_sink = False
        else:
            continue
        latency = latest_end - earliest_start
        latency_seconds = latency.total_seconds()

        if latency_seconds <= 0:
            continue

        total_events_int = int(total_events)
        throughput = total_events_int / latency_seconds if total_events_int else 0.0

        event_throughput: list[float]
        event_throughput_time: list[pd.Timestamp]
        if events_frames:
            combined_events = pd.concat(events_frames, ignore_index=True)
            combined_events = combined_events.groupby("time", as_index=False)["rate"].sum()
            combined_events = combined_events.sort_values("time")
            event_throughput = combined_events["rate"].tolist()
            event_throughput_time = combined_events["time"].tolist()
        else:
            event_throughput = []
            event_throughput_time = []

        repetitions[f"rep_{rep_id}"] = {
            "throughput": throughput,
            "source_tps": source_tps_total if has_source_tps else throughput,
            "latency_seconds": latency_seconds,
            "latency_ms": latency_seconds * 1000,
            "total_events": total_events_int,
            "source_start": earliest_start,
            "sink_end": latest_end,
            "sink_end_from_sink": sink_end_from_sink,
            "event_throughput": event_throughput,
            "event_throughput_time": event_throughput_time,
        }

        throughputs.append(throughput)
        latencies.append(latency_seconds)

    if allow_sink_only:
        # Add only repetitions the original timing-based loader could not load.
        sink_rep_ids = {
            rep_id for path in _collect_rep_files(sink_dir, "events")
            if (rep_id := _extract_rep_id(path)) is not None
        }
        for rep_id in sorted(sink_rep_ids, key=lambda value: (len(value), value)):
            if f"rep_{rep_id}" in repetitions:
                continue
            frames = [
                frame for path in _collect_rep_files(sink_dir, "events", rep_id)
                if (frame := _read_events_file(path, allow_nes_rate=True)) is not None
            ]
            if not frames:
                continue
            events = pd.concat(frames, ignore_index=True)
            events = events.groupby("time", as_index=False)["rate"].sum().sort_values("time")
            # Sink samples cannot establish end-to-end latency or source totals.
            repetitions[f"rep_{rep_id}"] = {
                "throughput": float("nan"),
                "source_tps": float("nan"),
                "latency_seconds": float("nan"),
                "latency_ms": float("nan"),
                "total_events": 0,
                "source_start": pd.NaT,
                "sink_end": events["time"].iloc[-1],
                "sink_end_from_sink": True,
                "event_throughput": events["rate"].tolist(),
                "event_throughput_time": events["time"].tolist(),
            }

    avg_throughput = pd.Series(throughputs).mean() if throughputs else float("nan")
    avg_latency = pd.Series(latencies).mean() if latencies else float("nan")

    return {
        "avg_throughput": avg_throughput,
        "avg_latency_seconds": avg_latency,
        "repetitions": repetitions,
    }


def process_experiments_nebula(
    path: str = "data_nebula",
    names: Sequence[str] | None = None,
    *,
    allow_sink_only: bool = False,
) -> dict[str, NebulaExperimentData]:
    """Load experiments; opt in to sink-only event curves without timing metrics."""
    base_path = Path(path)
    experiments: dict[str, NebulaExperimentData] = {}

    experiment_names = names if names is not None else ["hom_hom", "hom_het", "het_het"]

    for exp_name in experiment_names:
        exp_path = base_path / exp_name
        if not exp_path.is_dir():
            continue

        label = exp_name_mapping.get(exp_name, exp_name)
        result = process_repetitions_nebula(exp_path, allow_sink_only=allow_sink_only)
        result["exp_key"] = exp_name
        experiments[label] = result

    return experiments


def process_experiments_topo(
    path: str = "data_topo",
    names: Sequence[str] | None = None,
    use_latency_based_tps: bool = False,
) -> dict[str, NebulaExperimentData]:
    base_path = Path(path)
    experiments: dict[str, NebulaExperimentData] = {}

    experiment_names = list(names) if names is not None else sorted(
        exp_path.name for exp_path in base_path.iterdir() if exp_path.is_dir()
    )

    for exp_name in experiment_names:
        exp_path = base_path / exp_name
        if not exp_path.is_dir():
                print(f"Warning: Experiment directory '{exp_path}' does not exist. Skipping.")
                continue

        label = exp_name_mapping.get(exp_name, exp_name)
        result = process_repetitions_topo(
            exp_path,
            use_latency_based_tps=use_latency_based_tps,
        )
        result["exp_key"] = exp_name
        experiments[label] = result

    return experiments

def process_repetition_nm(repetition_path: Path | str) -> NmRepetitionData:
    repetition_dir = Path(repetition_path)
    throughput, runtime = process_tput_and_runtime(repetition_dir)

    return {
        "runtime": runtime,
        "throughput": throughput,
        "operator_graph": process_operator_diagram(str(repetition_dir)),
    }


def process_repetition(repetition_path: Path | str) -> StreamingRepetitionData:
    repetition_dir = Path(repetition_path)
    source_tput = process_source_throughput(repetition_dir)
    sink_tput = process_sink_throughput(repetition_dir)
    runtime = process_runtime(repetition_dir)

    return {
        "runtime": runtime,
        "source_throughput": source_tput,
        "avg_source_throughput": float(source_tput.mean()),
        "sink_throughput": sink_tput,
        "avg_sink_throughput": float(sink_tput.mean()),
        "operator_graph": process_operator_diagram(str(repetition_dir)),
    }


def process_repetitions_nm(exp_path: Path) -> NmExperimentData:
    repetitions: dict[str, NmRepetitionData] = {}
    for repetition_path in exp_path.iterdir():
        if repetition_path.is_dir():
            repetitions[repetition_path.name] = process_repetition_nm(repetition_path)

    avg_throughput = pd.Series([rep["throughput"] for rep in repetitions.values()]).mean()
    avg_runtime = float(pd.Series([
        rep["runtime"] for rep in repetitions.values()
    ]).mean())

    return {
        "avg_throughput": avg_throughput,
        "avg_runtime": avg_runtime,
        "repetitions": repetitions,
    }


def process_repetitions(exp_path: Path) -> ExperimentData:
    repetitions: dict[str, StreamingRepetitionData] = {}
    for repetition_path in exp_path.iterdir():
        if repetition_path.is_dir():
            repetitions[repetition_path.name] = process_repetition(repetition_path)

    avg_exp_source_throughput = float(pd.concat([
        rep["source_throughput"] for rep in repetitions.values()
    ]).mean())
    avg_exp_sink_throughput = float(pd.concat([
        rep["sink_throughput"] for rep in repetitions.values()
    ]).mean())
    avg_runtime = float(pd.Series([
        rep["runtime"] for rep in repetitions.values()
    ]).mean())

    return {
        "avg_avg_source_throughput": avg_exp_source_throughput,
        "avg_avg_sink_throughput": avg_exp_sink_throughput,
        "avg_runtime": avg_runtime,
        "repetitions": repetitions
    }


def process_experiments_nm(path: str = "data_nm", names: Sequence[str] | None = None) -> dict[str, NmExperimentData]:
    return cast(dict[str, NmExperimentData], process_experiments(path, names))


def process_experiments(path: str = "data", names: Sequence[str] | None = None) -> dict[str, ExperimentSummary]:
    base_path = Path(path)
    experiments: dict[str, ExperimentSummary] = {}

    exp_paths = {exp_path.name: exp_path for exp_path in base_path.iterdir() if exp_path.is_dir()}

    experiment_order = list(names) if names is not None else list(exp_name_mapping.keys())

    for exp_name in experiment_order:
        if path == "data_nes":
            exp_path = exp_paths.get(exp_name)
            if exp_path is None:
                continue
            label = exp_name_mapping.get(exp_name, exp_name)
            result = process_repetitions_nes(exp_path)
            result["exp_key"] = exp_name
            experiments[label] = result
            continue

        exp_path = exp_paths.get(exp_name)
        if exp_path is None:
            continue

        label = exp_name_mapping.get(exp_name, exp_name)
        experiment_result: ExperimentWithDag
        if path in {"data", "data_chained"}:
            experiment_result = process_repetitions(exp_path)
        else:
            experiment_result = process_repetitions_nm(exp_path)

        experiment_result["dag"] = process_flink_dag(exp_path)
        experiment_result["exp_key"] = exp_name

        experiments[label] = experiment_result

    return experiments


def prepare_exp_data(
        exp_data: Mapping[str, ExperimentSummary],
        key: str,
        exclude_repetitions: Sequence[int | str] | None = (0,10),
        use_source_num_tuples_tps: bool = True,
        require_matching_sink_time: bool = True,
        use_sink_event_tps: bool = False,
) -> DataFrame:
    plot_data: list[PlotRow] = []

    def _is_zero_value(value: float) -> bool:
        return math.isfinite(value) and value == 0.0

    def _rep_tokens(value: int | str) -> set[str]:
        text = str(value).strip()
        if not text:
            return set()

        tokens = {text}
        rep_id = _extract_rep_id(Path(text))
        if rep_id is not None:
            tokens.add(rep_id)
            tokens.add(f"rep_{rep_id}")
            tokens.add(f"_{rep_id}")

        if text.startswith("rep_") and len(text) > 4:
            tokens.add(text[4:])
        if text.startswith("_") and len(text) > 1:
            tokens.add(text[1:])
        if text.isdigit():
            tokens.add(f"rep_{text}")
            tokens.add(f"_{text}")

        return tokens

    exclude_set: set[str] = set()
    for rep in exclude_repetitions or []:
        exclude_set.update(_rep_tokens(rep))

    def _resolve_value(rep: Mapping[str, Any]) -> float | None:
        base_value = float(rep[key])
        if key != "throughput":
            return base_value

        if use_sink_event_tps:
            return None

        # Require sink-side timing to ensure src/sink repetition ids match.
        if require_matching_sink_time and rep.get("sink_end_from_sink", True) is False:
            return None

        source_tps_raw = rep.get("source_tps")
        source_tps_value = pd.to_numeric(pd.Series([source_tps_raw]), errors="coerce").iloc[0]
        if not use_source_num_tuples_tps:
            if pd.notna(source_tps_value) and math.isfinite(float(source_tps_value)):
                return float(source_tps_value)
            return base_value

        if rep.get("sink_end_from_sink", True) is False:
            return None

        source_tuples_raw = rep.get("source_num_tuples")
        if source_tuples_raw is None:
            source_tuples_raw = rep.get("source_total_events")
        if source_tuples_raw is None:
            source_tuples_raw = rep.get("num_events")
        if source_tuples_raw is None:
            # For topology/nebula data, total_events is sourced from source num_events.
            source_tuples_raw = rep.get("total_events")

        source_first_elem_ts = pd.to_datetime(rep.get("source_start"), errors="coerce")
        sink_last_elem_ts = pd.to_datetime(rep.get("sink_end"), errors="coerce")

        source_tuples_value = pd.to_numeric(pd.Series([source_tuples_raw]), errors="coerce").iloc[0]
        if pd.isna(source_tuples_value) or pd.isna(source_first_elem_ts) or pd.isna(sink_last_elem_ts):
            return None

        duration_seconds = (sink_last_elem_ts - source_first_elem_ts).total_seconds()
        if duration_seconds <= 0:
            return None

        source_tps = float(source_tuples_value) / duration_seconds
        if not math.isfinite(source_tps):
            return None
        return source_tps

    def _resolve_series_values(rep: Mapping[str, Any]) -> list[float]:
        if key != "throughput" or not use_sink_event_tps:
            return []

        event_values = pd.to_numeric(pd.Series(rep.get("event_throughput", [])), errors="coerce").dropna()
        if event_values.empty:
            return []

        values: list[float] = []
        for value in event_values.tolist():
            value_f = float(value)
            if math.isfinite(value_f) and not _is_zero_value(value_f):
                values.append(value_f)
        return values

    for exp_name, info in exp_data.items():
        exp_key = info.get("exp_key")
        repetitions = cast(Mapping[str, Mapping[str, Any]], info["repetitions"])
        for rep_name, rep in repetitions.items():
            if exclude_set.intersection(_rep_tokens(rep_name)):
                continue

            if key == "throughput" and use_sink_event_tps:
                series_values = _resolve_series_values(rep)
                if not series_values:
                    continue
                for index, value in enumerate(series_values):
                    record: dict[str, Any] = {
                        "exp_name": exp_name,
                        "value": value,
                        "repetition": rep_name,
                        "index": index,
                    }
                    if exp_key is not None:
                        record["exp_key"] = exp_key
                    plot_data.append(record)
                continue

            value = _resolve_value(rep)
            if value is None:
                continue
            if _is_zero_value(value):
                continue
            record: dict[str, Any] = {
                "exp_name": exp_name,
                "value": value,
            }
            if exp_key is not None:
                record["exp_key"] = exp_key
            plot_data.append(record)

    if not plot_data:
        return pd.DataFrame(columns=["exp_name", "value"])

    return pd.DataFrame(plot_data)


def prepare_relative_exp_data(
        exp_data: Mapping[str, ExperimentSummary],
        key: str,
        exclude_repetitions: Sequence[int | str] | None = (0,),
        use_source_num_tuples_tps: bool = True,
        require_matching_sink_time: bool = True,
        use_sink_event_tps: bool = False,
) -> DataFrame:
    plot_data: list[PlotRow] = []

    def _rep_tokens(value: int | str) -> set[str]:
        text = str(value).strip()
        if not text:
            return set()

        tokens = {text}
        rep_id = _extract_rep_id(Path(text))
        if rep_id is not None:
            tokens.add(rep_id)
            tokens.add(f"rep_{rep_id}")
            tokens.add(f"_{rep_id}")

        if text.startswith("rep_") and len(text) > 4:
            tokens.add(text[4:])
        if text.startswith("_") and len(text) > 1:
            tokens.add(text[1:])
        if text.isdigit():
            tokens.add(f"rep_{text}")
            tokens.add(f"_{text}")

        return tokens

    exclude_set: set[str] = set()
    for rep in exclude_repetitions or []:
        exclude_set.update(_rep_tokens(rep))

    def _resolve_value(rep: Mapping[str, Any]) -> float | None:
        base_value = float(rep[key])
        if key != "throughput":
            return base_value

        if use_sink_event_tps:
            return None

        # For relative throughput, require sink-side timing to ensure src/sink repetition ids match.
        if require_matching_sink_time and rep.get("sink_end_from_sink", True) is False:
            return None

        source_tps_raw = rep.get("source_tps")
        source_tps_value = pd.to_numeric(pd.Series([source_tps_raw]), errors="coerce").iloc[0]
        if not use_source_num_tuples_tps:
            if pd.notna(source_tps_value) and math.isfinite(float(source_tps_value)):
                return float(source_tps_value)
            return base_value

        if rep.get("sink_end_from_sink", True) is False:
            return None

        source_tuples_raw = rep.get("source_num_tuples")
        if source_tuples_raw is None:
            source_tuples_raw = rep.get("source_total_events")
        if source_tuples_raw is None:
            source_tuples_raw = rep.get("num_events")
        if source_tuples_raw is None:
            # For topology/nebula data, total_events is sourced from source num_events.
            source_tuples_raw = rep.get("total_events")

        source_first_elem_ts = pd.to_datetime(rep.get("source_start"), errors="coerce")
        sink_last_elem_ts = pd.to_datetime(rep.get("sink_end"), errors="coerce")

        source_tuples_value = pd.to_numeric(pd.Series([source_tuples_raw]), errors="coerce").iloc[0]
        if pd.isna(source_tuples_value) or pd.isna(source_first_elem_ts) or pd.isna(sink_last_elem_ts):
            return None

        duration_seconds = (sink_last_elem_ts - source_first_elem_ts).total_seconds()
        if duration_seconds <= 0:
            return None

        source_tps = float(source_tuples_value) / duration_seconds
        if not math.isfinite(source_tps):
            return None
        return source_tps

    def _resolve_series_values(rep: Mapping[str, Any]) -> list[float]:
        if key != "throughput" or not use_sink_event_tps:
            return []

        event_values = pd.to_numeric(pd.Series(rep.get("event_throughput", [])), errors="coerce").dropna()
        if event_values.empty:
            return []

        values: list[float] = []
        for value in event_values.tolist():
            value_f = float(value)
            if math.isfinite(value_f):
                values.append(value_f)
        return values

    for exp_name, info in exp_data.items():
        exp_key = info.get("exp_key")
        repetitions = cast(Mapping[str, Mapping[str, Any]], info["repetitions"])
        for rep_name, rep in repetitions.items():
            if exclude_set.intersection(_rep_tokens(rep_name)):
                continue

            if key == "throughput" and use_sink_event_tps:
                series_values = _resolve_series_values(rep)
                if not series_values:
                    continue
                for index, value in enumerate(series_values):
                    record: dict[str, Any] = {
                        "exp_name": exp_name,
                        "value": value,
                        "abs_value": round(value),
                        "repetition": rep_name,
                        "index": index,
                    }
                    if exp_key is not None:
                        record["exp_key"] = exp_key
                    plot_data.append(record)
                continue

            value = _resolve_value(rep)
            if value is None:
                continue
            record: dict[str, Any] = {
                "exp_name": exp_name,
                "value": value,
                "abs_value": round(value),
            }
            if exp_key is not None:
                record["exp_key"] = exp_key
            plot_data.append(record)

    if not plot_data:
        return pd.DataFrame(columns=["exp_name", "value", "abs_value"])

    df = pd.DataFrame(plot_data)
    group_max = df.groupby("exp_name")["value"].transform("max")
    df["value"] = df["value"] / group_max

    return df


def prepare_nebula_event_throughput_data(
        exp_data: Mapping[str, NebulaExperimentData],
        smooth_window: int | None = None,
        discard_first_and_last_event_tput: bool = True, # First and last event throughput can be an outlier due to startup effects, so we may want to discard it for plotting purposes.
) -> DataFrame:
    plot_data: list[dict[str, Any]] = []

    for exp_name, info in exp_data.items():
        exp_key = info.get("exp_key")
        repetitions = info["repetitions"]
        for rep_name, rep in repetitions.items():
            x_values, y_values, _ = _prepare_event_series(
                rep,
                smooth_window=smooth_window,
                discard_first_and_last_event_tput=discard_first_and_last_event_tput,
            )

            for pos, (x_value, value) in enumerate(zip(x_values, y_values)):
                record: dict[str, Any] = {
                    "exp_name": exp_name,
                    "repetition": rep_name,
                    "exp_name_tex": _tex_escape(exp_name),
                    "repetition_tex": _tex_escape(rep_name),
                    "index": float(x_value),
                    "value": float(value),
                }
                if exp_key is not None:
                    record["exp_key"] = exp_key

                plot_data.append(record)

    return pd.DataFrame(plot_data)


def prepare_nebula_event_throughput_mean_ci_data(
        exp_data: Mapping[str, NebulaExperimentData],
        smooth_window: int | None = None,
        discard_first_and_last_event_tput: bool = True,
        ci: float = 0.95,
) -> DataFrame:
    plot_data: list[dict[str, Any]] = []

    for exp_name, info in exp_data.items():
        exp_key = info.get("exp_key")
        repetitions = info["repetitions"]

        rep_series: list[tuple[np.ndarray, np.ndarray, float]] = []
        steps: list[float] = []

        for rep in repetitions.values():
            x_values, y_values, step = _prepare_event_series(
                rep,
                smooth_window=smooth_window,
                discard_first_and_last_event_tput=discard_first_and_last_event_tput,
            )
            if len(x_values) == 0 or len(y_values) == 0:
                continue
            rep_series.append((x_values, y_values, step))
            if step > 0 and math.isfinite(step):
                steps.append(step)

        if not rep_series:
            continue

        common_step = float(np.median(steps)) if steps else 1.0
        if common_step <= 0 or not math.isfinite(common_step):
            common_step = 1.0

        per_rep_values: list[pd.Series] = []
        max_idx = -1
        for x_values, y_values, _ in rep_series:
            idx = np.round(x_values / common_step).astype(int)
            series = pd.Series(y_values).groupby(idx).sum()
            if series.empty:
                continue
            per_rep_values.append(series)
            max_idx = max(max_idx, int(series.index.max()))

        if max_idx < 0 or not per_rep_values:
            continue

        for idx in range(0, max_idx + 1):
            values = [series.get(idx, np.nan) for series in per_rep_values]
            mean, ci_low, ci_high, n = _compute_mean_ci(values, ci=ci)
            if n == 0:
                continue
            record: dict[str, Any] = {
                "exp_name": exp_name,
                "exp_name_tex": _tex_escape(exp_name),
                "index": float(idx * common_step),
                "value": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
            }
            if exp_key is not None:
                record["exp_key"] = exp_key
            plot_data.append(record)

    return pd.DataFrame(plot_data)


def prepare_sink_latency_mean_ci_data(
        base_path: str,
        names: Sequence[str] | None = None,
        unit: str = "ms",
        smooth_window: int | None = None,
        discard_first_and_last: bool = False,
        ci: float = 0.95,
        sample_stride: int | None = None,
        max_points: int | None = None,
        exclude_repetitions: Sequence[int | str] | None = None,
        max_rows_per_file: int | None = None,
        sample_seed: int = 0,
        input_latency_unit: str | None = None,
        gap_fill: str = "zero",
) -> DataFrame:
    base_dir = Path(base_path)
    exp_paths = {exp_path.name: exp_path for exp_path in base_dir.iterdir() if exp_path.is_dir()}

    if names is None:
        ordered_names = [name for name in exp_name_mapping if name in exp_paths]
        ordered_names += [name for name in sorted(exp_paths) if name not in exp_name_mapping]
    else:
        ordered_names = list(names)

    unit_key, unit_scale = _normalize_latency_unit(unit)
    plot_data: list[dict[str, Any]] = []
    exclude_set = {str(rep) for rep in exclude_repetitions or []}

    for exp_name in ordered_names:
        exp_path = _resolve_exp_path(base_dir, exp_paths, exp_name)
        if exp_path is None:
            continue

        sink_dir = _find_io_dir(exp_path, "sink")
        rep_files: dict[str, list[Path]] = {}
        for latency_file in _list_latency_files(sink_dir):
            rep_id = _extract_rep_id(latency_file)
            if rep_id is None:
                continue
            if rep_id in exclude_set:
                continue
            rep_files.setdefault(rep_id, []).append(latency_file)

        rep_series: list[tuple[np.ndarray, np.ndarray, float]] = []
        steps: list[float] = []

        for files in rep_files.values():
            frames: list[DataFrame] = []
            has_time = True
            for latency_file in files:
                df = _read_latency_events_file(
                    latency_file,
                    sample_stride=sample_stride,
                    max_rows=max_rows_per_file,
                    sample_seed=sample_seed,
                    input_latency_unit=input_latency_unit,
                )
                if df is None or df.empty:
                    continue
                if "time" not in df.columns:
                    has_time = False
                frames.append(df)

            if not frames:
                continue

            combined = pd.concat(frames, ignore_index=True)
            combined["latency_ns"] = pd.to_numeric(combined["latency_ns"], errors="coerce")
            combined = combined.dropna(subset=["latency_ns"])
            if combined.empty:
                continue

            combined["latency_ns"] = combined["latency_ns"] / unit_scale

            if has_time and "time" in combined.columns:
                combined = combined.dropna(subset=["time"])
                if combined.empty:
                    continue
                combined = combined.sort_values("time")
                times = combined["time"].tolist()
            else:
                times = None

            x_values, y_values, step = _prepare_latency_series(
                times,
                combined["latency_ns"].to_numpy(),
                smooth_window=smooth_window,
                discard_first_and_last=discard_first_and_last,
                sample_stride=sample_stride,
                max_points=max_points,
                gap_fill=gap_fill,
            )
            if len(x_values) == 0 or len(y_values) == 0:
                continue

            rep_series.append((x_values, y_values, step))
            if step > 0 and math.isfinite(step):
                steps.append(step)

        if not rep_series:
            continue

        common_step = float(np.median(steps)) if steps else 1.0
        if common_step <= 0 or not math.isfinite(common_step):
            common_step = 1.0

        per_rep_values: list[pd.Series] = []
        max_idx = -1
        for x_values, y_values, _ in rep_series:
            idx = np.round(x_values / common_step).astype(int)
            series = pd.Series(y_values).groupby(idx).mean()
            if series.empty:
                continue
            per_rep_values.append(series)
            max_idx = max(max_idx, int(series.index.max()))

        if max_idx < 0 or not per_rep_values:
            continue

        for idx in range(0, max_idx + 1):
            values = [series.get(idx, np.nan) for series in per_rep_values]
            mean, ci_low, ci_high, n = _compute_mean_ci(values, ci=ci)
            if n == 0:
                if gap_fill == "nan":
                    record: dict[str, Any] = {
                        "exp_name": exp_name,
                        "exp_name_tex": _tex_escape(exp_name),
                        "index": float(idx * common_step),
                        "value": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n": 0,
                        "latency_unit": unit_key,
                    }
                    plot_data.append(record)
                continue
            record: dict[str, Any] = {
                "exp_name": exp_name,
                "exp_name_tex": _tex_escape(exp_name),
                "index": float(idx * common_step),
                "value": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
                "latency_unit": unit_key,
            }
            plot_data.append(record)

    return pd.DataFrame(plot_data)


def prepare_nebula_event_boxplot_summary(
        exp_data: Mapping[str, NebulaExperimentData]
) -> DataFrame:
    plot_data: list[dict[str, Any]] = []

    for exp_name, info in exp_data.items():
        exp_key = info.get("exp_key")
        repetitions = info["repetitions"]
        for rep_name, rep in repetitions.items():
            stats = _compute_boxplot_stats(rep.get("event_throughput", []))
            if not stats:
                continue

            record: dict[str, Any] = {
                "exp_name": exp_name,
                "repetition": rep_name,
                "exp_name_tex": _tex_escape(exp_name),
                "repetition_tex": _tex_escape(rep_name),
            }
            if exp_key is not None:
                record["exp_key"] = exp_key
            record.update(stats)
            plot_data.append(record)

    return pd.DataFrame(plot_data)


def prepare_grouped_boxplot_summary(
        df: DataFrame,
        group_col: str = "exp_name",
        value_col: str = "value",
        key_col: str | None = "exp_key",
) -> DataFrame:
    """Compute boxplot stats from an already prepared DataFrame (e.g. per-repetition throughput)."""
    required = {group_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for grouped boxplot summary: {', '.join(sorted(missing))}")

    records: list[dict[str, Any]] = []
    for group_name, group in df.groupby(group_col):
        values = pd.to_numeric(group[value_col], errors="coerce").dropna().to_numpy()
        stats = _compute_boxplot_stats(values)
        if not stats:
            continue

        record: dict[str, Any] = {
            group_col: group_name,
        }
        if key_col is not None and key_col in group.columns:
            keys = group[key_col].dropna().unique()
            if len(keys) > 0:
                record[key_col] = keys[0]
        record.update(stats)
        records.append(record)

    return pd.DataFrame(records)


def prepare_sink_latency_percentile_data(
        base_path: str,
        names: Sequence[str] | None = None,
        percentiles: Sequence[float] = (50, 90, 95, 99),
        exclude_repetitions: Sequence[int | str] | None = (0,),
        unit: str = "ms",
        per_repetition: bool = False,
) -> DataFrame:
    base_dir = Path(base_path)
    exp_paths = {exp_path.name: exp_path for exp_path in base_dir.iterdir() if exp_path.is_dir()}

    if names is None:
        ordered_names = [name for name in exp_name_mapping if name in exp_paths]
        ordered_names += [name for name in sorted(exp_paths) if name not in exp_name_mapping]
    else:
        ordered_names = list(names)

    exclude_set = {str(rep) for rep in exclude_repetitions or []}
    percentile_list = sorted({float(p) for p in percentiles})
    if any(p < 0 or p > 100 for p in percentile_list):
        raise ValueError("Percentiles must be within [0, 100].")

    unit_key, unit_scale = _normalize_latency_unit(unit)

    exp_entries: list[tuple[str, str, list[Path]]] = []
    all_files: list[Path] = []

    for exp_name in ordered_names:
        exp_path = _resolve_exp_path(base_dir, exp_paths, exp_name)
        if exp_path is None:
            continue

        label = exp_name_mapping.get(exp_name, exp_name)
        try:
            sink_dir = _find_io_dir(exp_path, "sink")
        except FileNotFoundError:
            continue

        latency_files = []
        for latency_file in _list_latency_files(sink_dir):
            rep_id = _extract_rep_id(latency_file)
            if rep_id is None or rep_id in exclude_set:
                continue
            latency_files.append(latency_file)

        if not latency_files:
            continue

        exp_entries.append((exp_name, label, latency_files))
        all_files.extend(latency_files)

    records: list[dict[str, Any]] = []

    for _, label, latency_files in exp_entries:
        if per_repetition:
            for latency_file in latency_files:
                rep_id = _extract_rep_id(latency_file)
                if rep_id is None or rep_id in exclude_set:
                    continue

                series = _read_latency_file(latency_file)
                if series is None or series.empty:
                    continue

                series = series.astype("float64") / unit_scale
                quantiles = series.quantile([p / 100 for p in percentile_list])

                for percentile in percentile_list:
                    value = float(quantiles.loc[percentile / 100])
                    records.append({
                        "exp_name": label,
                        "repetition": f"rep_{rep_id}",
                        "percentile": percentile,
                        "percentile_label": f"p{percentile:g}",
                        "value": value,
                        "unit": unit_key,
                    })
        else:
            series_list: list[Series] = []
            for latency_file in latency_files:
                rep_id = _extract_rep_id(latency_file)
                if rep_id is None or rep_id in exclude_set:
                    continue

                series = _read_latency_file(latency_file)
                if series is None or series.empty:
                    continue

                series_list.append(series)

            if not series_list:
                continue

            combined = pd.concat(series_list, ignore_index=True)
            combined = combined.astype("float64") / unit_scale
            quantiles = combined.quantile([p / 100 for p in percentile_list])

            for percentile in percentile_list:
                value = float(quantiles.loc[percentile / 100])
                records.append({
                    "exp_name": label,
                    "percentile": percentile,
                    "percentile_label": f"p{percentile:g}",
                    "value": value,
                    "unit": unit_key,
                })

    df = pd.DataFrame(records)

    return df


def prepare_sink_latency_summary_data(
        base_path: str,
        names: Sequence[str] | None = None,
        exclude_repetitions: Sequence[int | str] | None = (0,),
        unit: str = "ms",
        sample_stride: int | None = None,
        max_points: int | None = None,
        sample_seed: int = 0,
        input_latency_unit: str | None = None,
        aggregation: str = "pooled",
        quantile: float = 0.99,
) -> DataFrame:
    """Latency summaries computed one experiment at a time, to keep memory low."""
    base_dir = Path(base_path)
    exp_paths = {exp_path.name: exp_path for exp_path in base_dir.iterdir() if exp_path.is_dir()}

    if names is None:
        ordered_names = [name for name in exp_name_mapping if name in exp_paths]
        ordered_names += [name for name in sorted(exp_paths) if name not in exp_name_mapping]
    else:
        ordered_names = list(names)

    exclude_set = {str(rep) for rep in exclude_repetitions or []}
    unit_key, unit_scale = _normalize_latency_unit(unit)
    aggregation_key = aggregation.strip().lower()
    if aggregation_key not in {"pooled", "mean_of_repetitions"}:
        raise ValueError("aggregation must be either 'pooled' or 'mean_of_repetitions'.")

    q = float(quantile)
    if q > 1:
        q = q / 100.0
    if q < 0 or q > 1:
        raise ValueError("quantile must be in [0, 1] or [0, 100].")

    def _load_latency_values(latency_file: Path) -> np.ndarray | None:
        series = _read_numeric_series(
            latency_file,
            column="latency_ns",
            sample_stride=sample_stride,
            max_rows=max_points,
            sample_seed=sample_seed,
        )
        if series is None:
            series = _read_latency_file(
                latency_file,
                sample_stride=sample_stride,
                max_rows=max_points,
                sample_seed=sample_seed,
                input_latency_unit=input_latency_unit,
            )
        if series is None or series.empty:
            return None
        values = (series.astype("float64") / unit_scale).to_numpy()
        if values.size == 0:
            return None
        return values

    records: list[dict[str, Any]] = []

    for exp_name in ordered_names:
        exp_path = _resolve_exp_path(base_dir, exp_paths, exp_name)
        if exp_path is None:
            continue

        label = exp_name_mapping.get(exp_name, exp_name)
        try:
            sink_dir = _find_io_dir(exp_path, "sink")
        except FileNotFoundError:
            continue

        latency_files: list[Path] = []
        for latency_file in _list_latency_files(sink_dir):
            rep_id = _extract_rep_id(latency_file)
            if rep_id is None or rep_id in exclude_set:
                continue
            latency_files.append(latency_file)

        if not latency_files:
            continue

        if aggregation_key == "pooled":
            arrays: list[np.ndarray] = []
            for latency_file in latency_files:
                values = _load_latency_values(latency_file)
                if values is None:
                    continue
                arrays.append(values)

            if not arrays:
                continue

            combined = np.concatenate(arrays)
            if combined.size == 0:
                continue

            records.append({
                "exp_name": label,
                "exp_key": exp_name,
                "p50": float(np.quantile(combined, 0.5, method="linear")),
                "p99": float(np.quantile(combined, q, method="linear")),
                "count": int(combined.size),
                "unit": unit_key,
            })
            continue

        rep_p50s: list[float] = []
        rep_qs: list[float] = []
        count = 0
        for latency_file in latency_files:
            values = _load_latency_values(latency_file)
            if values is None:
                continue
            rep_p50s.append(float(np.quantile(values, 0.5, method="linear")))
            rep_qs.append(float(np.quantile(values, q, method="linear")))
            count += int(values.size)

        if not rep_p50s or not rep_qs:
            continue

        records.append({
            "exp_name": label,
            "exp_key": exp_name,
            "p50": float(np.mean(rep_p50s)),
            "p99": float(np.mean(rep_qs)),
            "count": count,
            "unit": unit_key,
        })

    return pd.DataFrame(records)


def prepare_sink_latency_distribution_data(
        base_path: str,
        names: Sequence[str] | None = None,
        exclude_repetitions: Sequence[int | str] | None = (0,),
        unit: str = "ms",
        per_repetition: bool = False,
        max_points: int | None = None,
        sample_stride: int | None = None,
        sample_seed: int = 0,
        input_latency_unit: str | None = None,
) -> DataFrame:
    base_dir = Path(base_path)
    exp_paths = {exp_path.name: exp_path for exp_path in base_dir.iterdir() if exp_path.is_dir()}

    if names is None:
        ordered_names = [name for name in exp_name_mapping if name in exp_paths]
        ordered_names += [name for name in sorted(exp_paths) if name not in exp_name_mapping]
    else:
        ordered_names = list(names)

    exclude_set = {str(rep) for rep in exclude_repetitions or []}
    unit_key, unit_scale = _normalize_latency_unit(unit)
    exp_entries: list[tuple[str, str, list[Path]]] = []
    all_files: list[Path] = []

    for exp_name in ordered_names:
        exp_path = _resolve_exp_path(base_dir, exp_paths, exp_name)
        if exp_path is None:
            continue

        label = exp_name_mapping.get(exp_name, exp_name)
        try:
            sink_dir = _find_io_dir(exp_path, "sink")
        except FileNotFoundError:
            continue

        latency_files = []
        for latency_file in _list_latency_files(sink_dir):
            rep_id = _extract_rep_id(latency_file)
            if rep_id is None or rep_id in exclude_set:
                continue
            latency_files.append(latency_file)

        if not latency_files:
            continue

        exp_entries.append((exp_name, label, latency_files))
        all_files.extend(latency_files)

    tasks: list[tuple[str, str, str, Path]] = []
    for exp_name, label, latency_files in exp_entries:
        for latency_file in latency_files:
            rep_id = _extract_rep_id(latency_file)
            if rep_id is None or rep_id in exclude_set:
                continue
            tasks.append((exp_name, label, rep_id, latency_file))

    if not tasks:
        return pd.DataFrame()

    def _load_latency_task(task: tuple[str, str, str, Path]) -> tuple[str, str, str, np.ndarray] | None:
        exp_name, label, rep_id, latency_file = task
        series = _read_numeric_series(
            latency_file,
            column="latency_ns",
            sample_stride=sample_stride,
            max_rows=max_points,
            sample_seed=sample_seed,
        )
        if series is None:
            series = _read_latency_file(
                latency_file,
                sample_stride=sample_stride,
                max_rows=max_points,
                sample_seed=sample_seed,
                input_latency_unit=input_latency_unit,
            )
        if series is None or series.empty:
            return None

        values = (series.astype("float64") / unit_scale).to_numpy()
        if values.size == 0:
            return None
        return exp_name, label, rep_id, values

    max_workers = min(8, len(tasks))
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            task_results = list(executor.map(_load_latency_task, tasks))
    else:
        task_results = [_load_latency_task(task) for task in tasks]

    exp_name_arrays: list[np.ndarray] = []
    exp_key_arrays: list[np.ndarray] = []
    repetition_arrays: list[np.ndarray] = []
    value_arrays: list[np.ndarray] = []
    unit_arrays: list[np.ndarray] = []

    for result in task_results:
        if result is None:
            continue
        exp_name, label, rep_id, values = result
        exp_name_arrays.append(np.full(values.size, label, dtype=object))
        exp_key_arrays.append(np.full(values.size, exp_name, dtype=object))
        if per_repetition:
            repetition_arrays.append(np.full(values.size, f"rep_{rep_id}", dtype=object))
        value_arrays.append(values)
        unit_arrays.append(np.full(values.size, unit_key, dtype=object))

    if not value_arrays:
        return pd.DataFrame()

    data: dict[str, np.ndarray] = {
        "exp_name": np.concatenate(exp_name_arrays),
        "exp_key": np.concatenate(exp_key_arrays),
        "value": np.concatenate(value_arrays),
        "unit": np.concatenate(unit_arrays),
    }
    if per_repetition:
        data["repetition"] = np.concatenate(repetition_arrays)

    return pd.DataFrame(data)


def prepare_sink_latency_ecdf_data(
        base_path: str,
        names: Sequence[str] | None = None,
        exclude_repetitions: Sequence[int | str] | None = (0,),
        unit: str = "ms",
        per_repetition: bool = False,
        max_points: int | None = None,
        sample_stride: int | None = None,
        sample_seed: int = 0,
        relative: bool = True,
        input_latency_unit: str | None = None,
) -> DataFrame:
    df = prepare_sink_latency_distribution_data(
        base_path=base_path,
        names=names,
        exclude_repetitions=exclude_repetitions,
        unit=unit,
        per_repetition=per_repetition,
        max_points=max_points,
        sample_stride=sample_stride,
        sample_seed=sample_seed,
        input_latency_unit=input_latency_unit,
    )

    if df.empty:
        return df

    if relative:
        df = df.copy()
        df["abs_value"] = df["value"]
        group_max = df.groupby("exp_name")["value"].transform("max")
        group_max = group_max.replace(0, np.nan)
        df["value"] = df["value"] / group_max
        df = df.dropna(subset=["value"])

    return df


def prepare_sink_latency_boxplot_summary(
        base_path: str,
        names: Sequence[str] | None = None,
        exclude_repetitions: Sequence[int | str] | None = (0,),
        unit: str = "ms",
        per_repetition: bool = False,
        max_points: int | None = None,
        sample_stride: int | None = None,
        sample_seed: int = 0,
) -> DataFrame:
    unit_key, unit_scale = _normalize_latency_unit(unit)
    return prepare_metric_boxplot_summary(
        base_path=base_path,
        names=names,
        exclude_repetitions=exclude_repetitions,
        per_repetition=per_repetition,
        max_points=max_points,
        sample_stride=sample_stride,
        sample_seed=sample_seed,
        io_dir="sink",
        file_glob="latency_test_*.parquet",
        value_column="latency_ns",
        unit_key=unit_key,
        unit_scale=unit_scale,
    )

def prepare_metric_boxplot_summary(
        base_path: str,
        names: Sequence[str] | None = None,
        exclude_repetitions: Sequence[int | str] | None = (0,),
        per_repetition: bool = False,
        max_points: int | None = None,
        sample_stride: int | None = None,
        sample_seed: int = 0,
        io_dir: str = "sink",
        file_glob: str | Sequence[str] = "latency_test_*.parquet",
        value_column: str | None = None,
        unit_key: str | None = None,
        unit_scale: float = 1.0,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        discard_first_and_last: bool = False,
) -> DataFrame:
    base_dir = Path(base_path)
    exp_paths = {exp_path.name: exp_path for exp_path in base_dir.iterdir() if exp_path.is_dir()}

    if names is None:
        ordered_names = [name for name in exp_name_mapping if name in exp_paths]
        ordered_names += [name for name in sorted(exp_paths) if name not in exp_name_mapping]
    else:
        ordered_names = list(names)

    exclude_set = {str(rep) for rep in exclude_repetitions or []}

    exp_entries: list[tuple[str, str, list[Path]]] = []
    for exp_name in ordered_names:
        exp_path = _resolve_exp_path(base_dir, exp_paths, exp_name)
        if exp_path is None:
            continue

        label = exp_name_mapping.get(exp_name, exp_name)
        try:
            io_path = _find_io_dir(exp_path, io_dir)
        except FileNotFoundError:
            continue

        patterns = [file_glob] if isinstance(file_glob, str) else list(file_glob)
        files_by_stem: dict[str, Path] = {}
        for pattern in patterns:
            for metric_file in sorted(io_path.rglob(pattern)):
                rep_id = _extract_rep_id(metric_file)
                if rep_id is not None and rep_id in exclude_set:
                    continue
                files_by_stem.setdefault(metric_file.stem, metric_file)
        files = sorted(files_by_stem.values(), key=lambda path: path.as_posix())

        if not files:
            continue

        exp_entries.append((exp_name, label, files))

    records: list[dict[str, Any]] = []

    def _apply_series_options(series: pd.Series) -> np.ndarray:
        values = series.astype("float64")
        if discard_first_and_last and len(values) > 2:
            values = values.iloc[1:-1]
        arr = values.to_numpy()
        if unit_scale and abs(unit_scale - 1.0) > 1e-9:
            arr = arr / unit_scale
        if transform is not None:
            arr = transform(arr)
        return arr

    for exp_name, label, files in exp_entries:
        if per_repetition:
            for metric_file in files:
                rep_id = _extract_rep_id(metric_file) or metric_file.stem
                if rep_id in exclude_set:
                    continue

                series = _read_numeric_series(
                    metric_file,
                    column=value_column,
                    sample_stride=sample_stride,
                    max_rows=max_points,
                    sample_seed=sample_seed,
                )
                if series is None or series.empty:
                    continue

                values = _apply_series_options(series)
                stats = _compute_boxplot_stats(values)
                if not stats:
                    continue

                stats.update({
                    "exp_name": label,
                    "exp_key": exp_name,
                    "repetition": f"rep_{rep_id}",
                })
                if unit_key is not None:
                    stats["unit"] = unit_key
                records.append(stats)
        else:
            values_list: list[np.ndarray] = []
            for metric_file in files:
                rep_id = _extract_rep_id(metric_file)
                if rep_id is not None and rep_id in exclude_set:
                    continue

                series = _read_numeric_series(
                    metric_file,
                    column=value_column,
                    sample_stride=sample_stride,
                    max_rows=max_points,
                    sample_seed=sample_seed,
                )
                if series is None or series.empty:
                    continue

                values = _apply_series_options(series)

                if values.size:
                    values_list.append(values)

            if not values_list:
                continue

            combined = np.concatenate(values_list)
            stats = _compute_boxplot_stats(combined)
            if not stats:
                continue

            stats.update({
                "exp_name": label,
                "exp_key": exp_name,
            })
            if unit_key is not None:
                stats["unit"] = unit_key
            records.append(stats)

    return pd.DataFrame(records)


def prepare_throughput_boxplot_summary(
        base_path: str,
        names: Sequence[str] | None = None,
        exclude_repetitions: Sequence[int | str] | None = None,
        per_repetition: bool = False,
        max_points: int | None = None,
        sample_stride: int | None = None,
        sample_seed: int = 0,
        file_glob: str | Sequence[str] = ("events_test_*.parquet", "events_*.parquet", "events_test_*.csv", "events_*.csv"),
        value_column: str = "rate",
        discard_first_and_last: bool = True,
) -> DataFrame:
    return prepare_metric_boxplot_summary(
        base_path=base_path,
        names=names,
        exclude_repetitions=exclude_repetitions,
        per_repetition=per_repetition,
        max_points=max_points,
        sample_stride=sample_stride,
        sample_seed=sample_seed,
        io_dir="sink",
        file_glob=file_glob,
        value_column=value_column,
        unit_key="tps",
        unit_scale=1.0,
        discard_first_and_last=discard_first_and_last,
    )


def prepare_latency_throughput_data(
        base_path: str,
        names: Sequence[str] | None = None,
        exclude_repetitions: Sequence[int | str] | None = (0,),
        latency_percentile: float = 99,
        unit: str = "ms",
        throughput_source: str = "auto",
        max_points: int | None = None,
        sample_stride: int | None = None,
        sample_seed: int = 0,
        per_test: bool = False,
) -> DataFrame:
    base_dir = Path(base_path)
    exp_paths = {exp_path.name: exp_path for exp_path in base_dir.iterdir() if exp_path.is_dir()}

    if names is None:
        ordered_names = [name for name in exp_name_mapping if name in exp_paths]
        ordered_names += [name for name in sorted(exp_paths) if name not in exp_name_mapping]
    else:
        ordered_names = list(names)

    percentile = float(latency_percentile)
    if percentile < 0 or percentile > 100:
        raise ValueError("Latency percentile must be within [0, 100].")

    exclude_set = {str(rep) for rep in exclude_repetitions or []}
    unit_key, unit_scale = _normalize_latency_unit(unit)
    throughput_source = throughput_source.strip().lower()
    if throughput_source not in {"auto", "time", "events"}:
        raise ValueError("throughput_source must be one of: 'auto', 'time', 'events'.")

    exp_entries: list[tuple[str, list[Path], list[Path], list[Path], str, str]] = []

    for exp_name in ordered_names:
        exp_path = _resolve_exp_path(base_dir, exp_paths, exp_name)
        if exp_path is None:
            continue

        label = exp_name_mapping.get(exp_name, exp_name)

        rep_dirs = [
            p for p in exp_path.iterdir()
            if p.is_dir() and (
                (p / "sink").is_dir()
                or (p / "source").is_dir()
                or any(p.glob("**/sink"))
            )
        ]
        if not rep_dirs:
            rep_dirs = [exp_path]

        for rep_dir in sorted(rep_dirs, key=lambda p: (len(p.name), p.name)):
            if rep_dir.name in exclude_set:
                continue
            try:
                sink_dir = _find_io_dir(rep_dir, "sink")
            except FileNotFoundError:
                continue

            latency_files = [
                lf for lf in _list_latency_files(sink_dir)
                if _extract_rep_id(lf) is None or _extract_rep_id(lf) not in exclude_set
            ]
            if not latency_files:
                continue

            rep_time_files = sorted(sink_dir.rglob("time_*.parquet"))
            rep_events_files = sorted(sink_dir.rglob("events_*.parquet"))

            exp_entries.append((label, latency_files, rep_time_files, rep_events_files, rep_dir.name, exp_name))

    records: list[dict[str, Any]] = []

    for label, latency_files, rep_time_files, rep_events_files, rep_dir_name, exp_key in exp_entries:
        parallelism = rep_dir_name
        try:
            parallelism = int(rep_dir_name)
        except Exception:
            pass

        if per_test:
            time_by_rep: dict[str, float] = {}
            event_by_rep: dict[str, float] = {}
            time_values_all: list[float] = []
            event_values_all: list[float] = []

            for time_file in rep_time_files:
                value = _read_time_throughput(time_file)
                if value is None:
                    continue
                value_f = float(value)
                time_values_all.append(value_f)
                rep_id = _extract_rep_id(time_file)
                if rep_id is not None:
                    time_by_rep[rep_id] = value_f

            for events_file in rep_events_files:
                value = _read_events_throughput(events_file)
                if value is None:
                    continue
                value_f = float(value)
                event_values_all.append(value_f)
                rep_id = _extract_rep_id(events_file)
                if rep_id is not None:
                    event_by_rep[rep_id] = value_f

            for idx, latency_file in enumerate(latency_files):
                series = _read_latency_file(
                    latency_file,
                    sample_stride=sample_stride,
                    max_rows=max_points,
                    sample_seed=sample_seed,
                )
                if series is None or series.empty:
                    continue

                latency_ms = series.astype("float64") / unit_scale
                latency_value = float(latency_ms.quantile(percentile / 100))

                rep_id = _extract_rep_id(latency_file)
                if rep_id is None:
                    rep_id = str(idx)

                throughput_value = None
                if throughput_source in {"auto", "time"}:
                    throughput_value = time_by_rep.get(rep_id)
                    if throughput_value is None and len(time_values_all) == 1:
                        throughput_value = time_values_all[0]

                if throughput_value is None and throughput_source in {"auto", "events"}:
                    throughput_value = event_by_rep.get(rep_id)
                    if throughput_value is None and len(event_values_all) == 1:
                        throughput_value = event_values_all[0]

                if throughput_value is None:
                    continue

                repetition = rep_id
                try:
                    repetition = int(rep_id)
                except Exception:
                    pass

                records.append({
                    "exp_name": label,
                    "exp_key": exp_key,
                    "parallelism": parallelism,
                    "repetition": repetition,
                    "throughput": float(throughput_value),
                    "latency": latency_value,
                    "latency_percentile": percentile,
                    "latency_unit": unit_key,
                })
        else:
            latency_series_list: list[Series] = []
            for latency_file in latency_files:
                series = _read_latency_file(
                    latency_file,
                    sample_stride=sample_stride,
                    max_rows=max_points,
                    sample_seed=sample_seed,
                )
                if series is None or series.empty:
                    continue
                latency_series_list.append(series)

            if not latency_series_list:
                continue

            latency_series = pd.concat(latency_series_list, ignore_index=True)
            latency_ms = latency_series.astype("float64") / unit_scale
            latency_value = float(latency_ms.quantile(percentile / 100))

            throughput_value = None
            if throughput_source in {"auto", "time"}:
                time_values: list[float] = []
                for time_file in rep_time_files:
                    value = _read_time_throughput(time_file)
                    if value is not None:
                        time_values.append(float(value))
                if time_values:
                    throughput_value = float(np.mean(time_values))

            if throughput_value is None and throughput_source in {"auto", "events"}:
                event_values: list[float] = []
                for events_file in rep_events_files:
                    value = _read_events_throughput(events_file)
                    if value is not None:
                        event_values.append(float(value))
                if event_values:
                    throughput_value = float(np.mean(event_values))

            if throughput_value is None:
                continue

            records.append({
                "exp_name": label,
                "exp_key": exp_key,
                "parallelism": parallelism,
                "throughput": float(throughput_value),
                "latency": latency_value,
                "latency_percentile": percentile,
                "latency_unit": unit_key,
            })

    df = pd.DataFrame(records)

    return df


class OpType(Enum):
    MIN = 1
    MAX = 2


def get_rep_with(data: ExperimentWithDag, key: str, op_type: OpType) -> OperatorRepetition:
    def fn(item: OperatorRepetition) -> Any:
        return item[key]

    repetitions = cast(dict[str, OperatorRepetition], data["repetitions"])

    match op_type:
        case OpType.MIN:
            return min(repetitions.values(), key=fn)
        case OpType.MAX:
            return max(repetitions.values(), key=fn)

    raise ValueError(f"Unsupported OpType: {op_type}")
