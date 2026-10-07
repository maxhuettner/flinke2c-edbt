#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import exp_name_mapping
from exp_utils import prepare_exp_data, prepare_grouped_boxplot_summary, process_experiments_topo
from gen_plot_data import gen_boxplot_summary_data, gen_ecdf_data
from plot_utils import ecdf_plot, repetition_throughput_boxplot

SYSTEM_ORDER = ["flink", "flinke2c_td", "flinke2c_bu", "nes"]
ENV_ORDER = {"cloud": 0, "e2c": 1, "eh": 2, "wan": 3}

# flink / flinke2c_td / flinke2c_bu names carry a "_nes_" infix before the env
# (e.g. flink_q1_nes_cloud); the bare "nes" system does not (e.g. nes_q1_cloud).
_SUFFIXED_PATTERN = re.compile(
    r"^(?P<system>flink|flinke2c_td|flinke2c_bu)_q(?P<query>\d+[a-z0-9]*)_nes_(?P<env>cloud|e2c|eh|wan)$"
)
_NES_PATTERN = re.compile(
    r"^nes_q(?P<query>\d+[a-z0-9]*)_(?P<env>cloud|e2c|eh|wan)$"
)


def _match_name(name: str) -> tuple[str, str, str] | None:
    match = _SUFFIXED_PATTERN.match(name)
    if match:
        return match.group("system"), match.group("query"), match.group("env")

    match = _NES_PATTERN.match(name)
    if match:
        return "nes", match.group("query"), match.group("env")

    return None


# Edit these variables directly before running the script.
PATH = "data_new"
NAMES = [
    "flink_q1_nes_cloud", "flink_q1_nes_e2c", "flink_q1_nes_eh", "flink_q1_nes_wan",
    "flink_q4_nes_cloud", "flink_q4_nes_e2c", "flink_q4_nes_eh", "flink_q4_nes_wan",
    "flink_q5_nes_cloud", "flink_q5_nes_e2c", "flink_q5_nes_eh", "flink_q5_nes_wan",
    "flinke2c_td_q1_nes_e2c", "flinke2c_td_q1_nes_eh",
    "flinke2c_td_q4_nes_e2c", "flinke2c_td_q4_nes_eh",
    "flinke2c_td_q5_nes_e2c", "flinke2c_td_q5_nes_eh",
    "flinke2c_bu_q1_nes_e2c", "flinke2c_bu_q1_nes_eh",
    "flinke2c_bu_q4_nes_e2c", "flinke2c_bu_q4_nes_eh",
    "flinke2c_bu_q5_nes_e2c", "flinke2c_bu_q5_nes_eh",
    "nes_q1_cloud", "nes_q1_e2c", "nes_q1_eh", "nes_q1_wan",
    "nes_q4_cloud", "nes_q4_e2c", "nes_q4_eh", "nes_q4_wan",
    "nes_q5_cloud", "nes_q5_e2c", "nes_q5_eh", "nes_q5_wan",
]
EXCLUDE_REPETITIONS = ("0", "10")
OUTPUT_SUFFIX = "nes"
ECDF_FILE_PREFIX = "nes_abs"
BOXPLOT_FILE_PREFIX = "nes_abs"
FIG_DIR = Path("figs/auto_nes")
FIG_FORMAT = "pdf"
MAX_WORKERS = 4


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


def _series_file_key(exp_name: str) -> str:
    return exp_name.split("_q", 1)[0]


def _ordered_system_keys(values: list[str]) -> list[str]:
    unique_values = list(dict.fromkeys(values))
    known = [system for system in SYSTEM_ORDER if system in unique_values]
    remaining = sorted(value for value in unique_values if value not in SYSTEM_ORDER)
    return known + remaining


def _ensure_system_order_mapping() -> None:
    for system in SYSTEM_ORDER:
        exp_name_mapping.setdefault(system, system)


def _ordered_exp_labels(df: pd.DataFrame, exp_names: list[str]) -> list[str]:
    if "exp_key" not in df.columns or "exp_name" not in df.columns:
        return []

    label_by_key = (
        df[["exp_key", "exp_name"]]
        .dropna(subset=["exp_key", "exp_name"])
        .drop_duplicates(subset=["exp_key"], keep="first")
        .set_index("exp_key")["exp_name"]
        .to_dict()
    )
    return [label_by_key[exp_name] for exp_name in exp_names if exp_name in label_by_key]


def _apply_exp_label_order(df: pd.DataFrame, ordered_labels: list[str]) -> pd.DataFrame:
    if not ordered_labels or "exp_name" not in df.columns:
        return df

    ordered = df.copy()
    ordered["exp_name"] = pd.Categorical(
        ordered["exp_name"],
        categories=ordered_labels,
        ordered=True,
    )
    return ordered.sort_values("exp_name", kind="stable").reset_index(drop=True)


def _rewrite_ecdf_artifact_names(
        *,
        output_suffix: str,
        file_prefix: str,
        query: str,
        env: str,
        series_keys: list[str],
) -> None:
    suffix = f"_{output_suffix}" if output_suffix else ""
    plot_data_dir = Path(f"plot_data{suffix}")
    plot_code_dir = Path(f"plot_code{suffix}")
    old_base = f"{file_prefix}_{query}_{env}"

    replacements: dict[str, str] = {}
    for system_key in series_keys:
        old_name = f"{old_base}_{system_key}_ecdf.csv"
        new_name = f"{file_prefix}_{system_key}_{query}_{env}_ecdf.csv"
        old_path = plot_data_dir / old_name
        new_path = plot_data_dir / new_name
        if old_path.exists():
            if new_path.exists() and new_path != old_path:
                new_path.unlink()
            old_path.rename(new_path)
        replacements[old_name] = new_name

    plot_code_path = plot_code_dir / f"{old_base}_ecdf.tex"
    if plot_code_path.exists():
        content = plot_code_path.read_text(encoding="utf-8")
        for old_name, new_name in replacements.items():
            content = content.replace(old_name, new_name)
        plot_code_path.write_text(content, encoding="utf-8")


def _group_experiment_names(names: list[str]) -> list[tuple[str, str, list[str]]]:
    grouped: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    ignored: list[str] = []

    for name in names:
        match = _match_name(name)
        if match is None:
            ignored.append(name)
            continue
        system, query, env = match
        grouped[(query, env)][system] = name

    if ignored:
        print(f"Ignoring {len(ignored)} names with unsupported pattern: {', '.join(sorted(ignored))}")

    jobs: list[tuple[str, str, list[str]]] = []
    for (query, env), systems in grouped.items():
        exp_names = [systems[system] for system in SYSTEM_ORDER if system in systems]
        if len(exp_names) >= 2:
            jobs.append((query, env, exp_names))

    def _query_sort_key(query: str) -> tuple[int, str]:
        match = re.match(r"(\d+)", query)
        if match:
            return (int(match.group(1)), query)
        return (10_000, query)

    jobs.sort(key=lambda item: (_query_sort_key(item[0]), ENV_ORDER.get(item[1], 99)))
    return jobs


def _run_one(
        *,
        base_path: str,
        query: str,
        env: str,
        exp_names: list[str],
        exclude_repetitions: tuple[str, ...],
        output_suffix: str,
        ecdf_file_prefix: str,
        boxplot_file_prefix: str,
        fig_dir: Path,
        fig_format: str,
) -> str:
    env_label = env.replace("_", "-")
    ecdf_base_name = f"{ecdf_file_prefix}_q{query}_{env}"
    boxplot_base_name = f"{boxplot_file_prefix}_q{query}_{env}"

    _ensure_system_order_mapping()

    data = process_experiments_topo(path=base_path, names=exp_names)
    throughput_data = prepare_exp_data(
        data,
        "throughput",
        exclude_repetitions=exclude_repetitions,
    )
    if throughput_data.empty:
        return f"Q{query} {env}: no throughput data"

    ordered_labels = _ordered_exp_labels(throughput_data, exp_names)
    throughput_data = _apply_exp_label_order(throughput_data, ordered_labels)

    ecdf_fig = ecdf_plot(
        throughput_data,
        x_name="Relative Throughput",
        title=f"Throughput (NES | Q{query} | {env_label} | higher is better)",
        is_int_axis=False,
    )
    _save_figure(ecdf_fig, fig_dir / f"{ecdf_base_name}_ecdf.{fig_format}")
    plt.close(ecdf_fig)

    ecdf_export_data = throughput_data.copy()
    key_source = ecdf_export_data["exp_key"] if "exp_key" in ecdf_export_data.columns else ecdf_export_data["exp_name"]
    ecdf_export_data["exp_key"] = key_source.map(_series_file_key)
    gen_ecdf_data(
        ecdf_export_data,
        ecdf_base_name,
        output_suffix=output_suffix,
        x_label="Throughput (tps)",
    )
    _rewrite_ecdf_artifact_names(
        output_suffix=output_suffix,
        file_prefix=ecdf_file_prefix,
        query=f"q{query}",
        env=env,
        series_keys=_ordered_system_keys(ecdf_export_data["exp_key"].dropna().unique().tolist()),
    )

    tput_boxplot_data = prepare_grouped_boxplot_summary(
        throughput_data,
        group_col="exp_name",
        value_col="value",
        key_col="exp_key",
    )
    if tput_boxplot_data.empty:
        return f"Q{query} {env}: no throughput boxplot data"

    tput_boxplot_data = _apply_exp_label_order(tput_boxplot_data, ordered_labels)

    boxplot_fig, _ = repetition_throughput_boxplot(
        throughput_data,
        x="exp_name",
        x_name="Experiment",
        title=f"Throughput (Q{query})",
        hue="exp_name",
        legend=False,
    )
    _save_figure(boxplot_fig, fig_dir / f"{boxplot_base_name}_boxplot.{fig_format}")
    plt.close(boxplot_fig)

    gen_boxplot_summary_data(
        tput_boxplot_data,
        boxplot_base_name,
        output_suffix=output_suffix,
        x_col="exp_name",
    )

    return f"Q{query} {env}: done"


def main() -> int:
    jobs = _group_experiment_names(list(NAMES))
    if not jobs:
        raise RuntimeError("No valid grouped experiment sets found in NAMES.")

    fig_dir = FIG_DIR
    exclude_repetitions = tuple(str(rep) for rep in EXCLUDE_REPETITIONS)
    max_workers = max(1, min(int(MAX_WORKERS), len(jobs)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_one,
                base_path=PATH,
                query=query,
                env=env,
                exp_names=exp_names,
                exclude_repetitions=exclude_repetitions,
                output_suffix=OUTPUT_SUFFIX,
                ecdf_file_prefix=ECDF_FILE_PREFIX,
                boxplot_file_prefix=BOXPLOT_FILE_PREFIX,
                fig_dir=fig_dir,
                fig_format=FIG_FORMAT,
            )
            for query, env, exp_names in jobs
        ]

        for future in as_completed(futures):
            print(future.result())

    print(f"Saved figures under {fig_dir}")
    print(f"Generated data under plot_data_{OUTPUT_SUFFIX}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
