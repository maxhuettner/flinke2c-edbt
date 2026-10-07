from io import StringIO
from pathlib import Path
from typing import Protocol, Optional, Sequence

from matplotlib import cm
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import numpy as np

from constants import exp_name_mapping
from exp_utils import get_rep_with, OpType


class PlotFunction(Protocol):
    def __call__(self, data: DataFrame, x: str, ax: Axes, is_higher_better: bool, hue: str, log_scale: bool) -> Axes:
        pass


def imputation_perf_plot(
        name: str,
        average: bool = True,
        path: str | Path = "perf_data",
        percentage: bool = True,
        grouped: bool = True,
):
    """Plot GPU imputation pipeline timing as horizontal stacked bars.

    The input ``name`` identifies a folder under ``path`` containing one CSV
    file, for example ``imputation_perf_plot("q1i_direct_gpu")``.
    With ``average=True`` (the default), one bar shows the mean across runs.
    With ``average=False``, one bar is shown for each repetition.
    With ``percentage=True`` (the default), each bar is normalized to 100%.
    With ``grouped=True`` (the default), percentage bars retain every stage
    while annotating the aggregate ``Flink`` and ``GPU`` sections.
    """
    exp_path = Path(path) / name
    csv_paths = sorted(exp_path.glob("*.csv"))
    if len(csv_paths) != 1:
        raise ValueError(
            f"Expected exactly one CSV file in {exp_path}, found {len(csv_paths)}"
        )
    csv_path = csv_paths[0]
    csv_lines = csv_path.read_text().splitlines()
    try:
        header_line = next(
            index for index, line in enumerate(csv_lines)
            if line.startswith("timestamp,")
        )
    except StopIteration as error:
        raise ValueError(f"Could not find the CSV header in {csv_path}") from error

    perf = pd.read_csv(StringIO("\n".join(csv_lines[header_line:])))
    total_columns = [
        column for column in perf.columns if column.endswith("TotalMillis")
    ]
    if not total_columns:
        raise ValueError(f"No *TotalMillis columns found in {csv_path}")

    step_names = {
        column: column.removesuffix("TotalMillis")
        for column in total_columns
    }
    stage_data = perf[total_columns].rename(columns=step_names).drop(
        columns=["wait"],
        errors="ignore",
    )
    flink_steps = [
        "laneWait", "extract", "fill", "frame", "submit", "decode", "collect",
    ]
    gpu_steps = [
        "gpuH2D", "gpuPrepare", "gpuProcess", "gpuKernel", "gpuCommit", "gpuD2H",
    ]
    operation_order = flink_steps[:5] + gpu_steps + flink_steps[5:]
    known_steps = set(flink_steps + gpu_steps)
    remaining_steps = [step for step in stage_data.columns if step not in known_steps]
    group_columns = {}
    if percentage and grouped:
        flink_columns = [step for step in flink_steps if step in stage_data.columns]
        gpu_columns = [step for step in gpu_steps if step in stage_data.columns]
        gpu_columns += [step for step in remaining_steps if step not in flink_columns]
        stage_data = stage_data[flink_columns + gpu_columns]
        group_columns = {"Flink": flink_columns, "GPU": gpu_columns}
    else:
        step_order = [step for step in operation_order if step in stage_data.columns]
        step_order += [step for step in remaining_steps if step not in step_order]
        stage_data = stage_data[step_order]

    if average:
        stage_data = stage_data.mean().to_frame().T
        repetition_labels = [""]
        title = (
            "Average GPU imputation runtime by pipeline step"
            if percentage and grouped
            else "Average GPU imputation time by pipeline step"
        )
        y_label = ""
    else:
        repetition_labels = list(range(1, len(stage_data) + 1))
        title = (
            "GPU imputation runtime by repetition and pipeline step"
            if percentage and grouped
            else "GPU imputation time by repetition and pipeline step"
        )
        y_label = "Repetition"

    y_positions = np.arange(len(stage_data))

    if percentage:
        totals = stage_data.sum(axis=1)
        stage_data = stage_data.div(totals.replace(0, np.nan), axis=0).fillna(0) * 100
        x_label = "Runtime (%)"
    else:
        x_label = "Accumulated time (ms)"

    fig, ax = plt.subplots(figsize=(12, 4 if average else 6))
    colors = [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#000000",  # black
    ]
    hatches = ["///", r"\\\\", "...", "xxx", "---", "|||", "+++", "ooo", "OOO", "***", "..", "||//"]
    left = np.zeros(len(stage_data))
    legend_handles = []

    for index, step in enumerate(stage_data.columns):
        values = stage_data[step].to_numpy(dtype=float)
        stage_label = (
            f"{step} ({stage_data[step].mean():.1f}%)"
            if percentage
            else step
        )
        ax.barh(
            y_positions,
            values,
            left=left,
            color=colors[index % len(colors)],
            edgecolor="black",
            linewidth=0.3,
            label="_nolegend_",
        )
        ax.barh(
            y_positions,
            values,
            left=left,
            facecolor="none",
            hatch=hatches[index % len(hatches)],
            edgecolor=(0, 0, 0, 0.35),
            linewidth=0.6,
            label="_nolegend_",
        )
        legend_handles.append(
            Patch(
                facecolor=colors[index % len(colors)],
                edgecolor=(0, 0, 0, 0.35),
                hatch=hatches[index % len(hatches)],
                label=stage_label,
            )
        )
        left += values

    if percentage and grouped:
        group_colors = {"Flink": "#333333", "GPU": "#777777"}
        group_left = np.zeros(len(stage_data))
        for group_name, columns in group_columns.items():
            group_width = stage_data[columns].sum(axis=1).to_numpy(dtype=float)
            for y_position, start, width in zip(y_positions, group_left, group_width):
                group_y = y_position + 0.47
                group_right = start + width
                group_color = group_colors[group_name]
                ax.plot(
                    [start, group_right],
                    [group_y, group_y],
                    color=group_color,
                    linewidth=1.5,
                    clip_on=False,
                )
                ax.plot(
                    [start, start],
                    [group_y - 0.07, group_y],
                    color=group_color,
                    linewidth=1.5,
                    clip_on=False,
                )
                ax.plot(
                    [group_right, group_right],
                    [group_y - 0.07, group_y],
                    color=group_color,
                    linewidth=1.5,
                    clip_on=False,
                )
                ax.text(
                    (start + group_right) / 2,
                    group_y + 0.04,
                    f"{group_name} {width:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=group_color,
                )
            group_left += group_width

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yticks(y_positions, labels=repetition_labels)
    if percentage:
        ax.set_xlim(0, 100)
    if percentage and grouped:
        ax.set_ylim(-0.5, len(stage_data) - 0.5 + 0.8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        handles=legend_handles,
        title="Pipeline step",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "frame: encode rows  •  decode: decode output  •  collect: emit rows",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 0.82, 1))
    plt.show()
    return fig, ax


def rdma_gpu_perf_plot(
        name: str,
        average: bool = True,
        path: str | Path = "perf_data",
        percentage: bool = True,
        grouped: bool = True,
):
    """Plot paired RDMA/Flink and GPU server performance files together.

    The split RDMA pre/post rows are summed per repetition before being
    aligned with the one GPU row for that repetition.
    """
    exp_path = Path(path) / name
    rdma_pre_path = exp_path / "rdma_perf_pre.csv"
    rdma_post_path = exp_path / "rdma_perf_post.csv"
    gpu_path = exp_path / "perf_gpu.csv"
    if not rdma_pre_path.is_file() or not rdma_post_path.is_file() or not gpu_path.is_file():
        raise FileNotFoundError(
            f"Expected rdma_perf_pre.csv, rdma_perf_post.csv, and perf_gpu.csv "
            f"in {exp_path}"
        )

    gpu = pd.read_csv(gpu_path)
    gpu_columns = [column for column in gpu.columns if column.endswith("TotalMillis")]
    rdma_pre = pd.read_csv(rdma_pre_path)
    rdma_post = pd.read_csv(rdma_post_path)
    rdma_pre_columns = [column for column in rdma_pre.columns if column.endswith("TotalMillis")]
    rdma_post_columns = [column for column in rdma_post.columns if column.endswith("TotalMillis")]
    if len(rdma_pre) != len(rdma_post):
        raise ValueError("RDMA pre/post files have different repetition counts")
    if len(rdma_pre) != len(gpu):
        raise ValueError(
            f"RDMA repetitions ({len(rdma_pre)}) do not match GPU repetitions ({len(gpu)})"
        )
    def prefixed_names(columns, prefix):
        return [
            prefix + column.removesuffix("TotalMillis")[0].upper()
            + column.removesuffix("TotalMillis")[1:]
            for column in columns
        ]

    rdma_pre_names = prefixed_names(rdma_pre_columns, "rdmaPre")
    rdma_post_names = prefixed_names(rdma_post_columns, "rdmaPost")
    gpu_names = prefixed_names(gpu_columns, "gpu")
    rdma_pre_runs = rdma_pre[rdma_pre_columns].reset_index(drop=True)
    rdma_post_runs = rdma_post[rdma_post_columns].reset_index(drop=True)
    rdma_pre_runs = rdma_pre_runs.set_axis(rdma_pre_names, axis="columns")
    rdma_post_runs = rdma_post_runs.set_axis(rdma_post_names, axis="columns")
    gpu_runs = gpu[gpu_columns].reset_index(drop=True)
    gpu_runs = gpu_runs.set_axis(gpu_names, axis="columns")
    stage_data = pd.concat(
        [rdma_pre_runs, rdma_post_runs, gpu_runs],
        axis=1,
    )
    rdma_names = rdma_pre_names + rdma_post_names

    if average:
        stage_data = stage_data.mean().to_frame().T
        repetition_labels = [""]
        title = "Average RDMA/GPU runtime by pipeline step"
    else:
        repetition_labels = list(range(1, len(stage_data) + 1))
        title = "RDMA/GPU runtime by repetition and pipeline step"

    if grouped:
        group_columns = {
            "RDMA pre": rdma_pre_names,
            "GPU": gpu_names,
            "RDMA post": rdma_post_names,
        }
        stage_order = rdma_pre_names + gpu_names + rdma_post_names
    else:
        group_columns = {}
        stage_order = rdma_pre_names + gpu_names + rdma_post_names
    stage_data = stage_data[stage_order]

    if percentage:
        totals = stage_data.sum(axis=1)
        stage_data = stage_data.div(totals.replace(0, np.nan), axis=0).fillna(0) * 100
        x_label = "Runtime (%)"
    else:
        x_label = "Accumulated time (ms)"

    y_positions = np.arange(len(stage_data))
    fig, ax = plt.subplots(figsize=(12, 4 if average else 6))
    colors = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000",
    ]
    hatches = ["///", r"\\\\", "...", "xxx", "---", "|||", "+++", "ooo", "OOO", "***", "..", "||//"]
    left = np.zeros(len(stage_data))
    legend_handles = []

    for index, step in enumerate(stage_data.columns):
        values = stage_data[step].to_numpy(dtype=float)
        stage_label = f"{step} ({stage_data[step].mean():.1f}%)" if percentage else step
        color = colors[index % len(colors)]
        hatch = hatches[index % len(hatches)]
        ax.barh(
            y_positions, values, left=left, color=color,
            edgecolor="black", linewidth=0.3, label="_nolegend_",
        )
        ax.barh(
            y_positions, values, left=left, facecolor="none",
            hatch=hatch, edgecolor=(0, 0, 0, 0.35), linewidth=0.6,
            label="_nolegend_",
        )
        legend_handles.append(
            Patch(facecolor=color, edgecolor=(0, 0, 0, 0.35), hatch=hatch, label=stage_label)
        )
        left += values

    if percentage and grouped:
        group_left = np.zeros(len(stage_data))
        for group_name, columns in group_columns.items():
            group_width = stage_data[columns].sum(axis=1).to_numpy(dtype=float)
            for y_position, start, width in zip(y_positions, group_left, group_width):
                group_y = y_position + 0.47
                group_right = start + width
                group_color = "#333333" if group_name != "GPU" else "#777777"
                ax.plot([start, group_right], [group_y, group_y], color=group_color, linewidth=1.5, clip_on=False)
                ax.plot([start, start], [group_y - 0.07, group_y], color=group_color, linewidth=1.5, clip_on=False)
                ax.plot([group_right, group_right], [group_y - 0.07, group_y], color=group_color, linewidth=1.5, clip_on=False)
                ax.text((start + group_right) / 2, group_y + 0.04, f"{group_name} {width:.1f}%", ha="center", va="bottom", fontsize=8, color=group_color)
            group_left += group_width

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Repetition" if not average else "")
    ax.set_yticks(y_positions, labels=repetition_labels)
    if percentage:
        ax.set_xlim(0, 100)
    if percentage and grouped:
        ax.set_ylim(-0.5, len(stage_data) - 0.5 + 0.8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(handles=legend_handles, title="Pipeline step", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.text(0.5, 0.01, "RDMA pre: encode, publish, collect  •  GPU: receive, flush, submit, cudaWait, output, outputDrain  •  RDMA post: receive, decode, collect", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 0.82, 1))
    plt.show()
    return fig, ax


def __basic_plot(plot_fn: PlotFunction, data: DataFrame, x: str, x_name: str, y_name: str, title: str,
                 is_higher_better: bool = True, hue: str = "", log_scale: bool = False):
    if not data.empty:
        num_series = data[hue].nunique() if hue and hue in data.columns else 1
        fig_width = min(14, max(6, 4 + num_series * 1.5))
        fig, ax = plt.subplots(figsize=(fig_width, 6))
    else:
        fig, ax = plt.subplots(figsize=(6, 6))

    if not data.empty:
        plot_fn(data, x, ax, is_higher_better, hue, log_scale)

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(ymin=0)

    return fig, ax


def __basic_plot_by_categories(
        plot_fn: PlotFunction,
        data: DataFrame,
        x: str,
        x_name: str,
        y_name: str,
        title: str,
        is_higher_better: bool = True,
        hue: str = "",
        log_scale: bool = False,
):
    num_categories = data[x].nunique() if x in data.columns else len(data)
    fig_width = min(18, max(6, num_categories * 1.5))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    if not data.empty:
        plot_fn(data, x, ax, is_higher_better, hue, log_scale)

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(ymin=0)

    return fig, ax


def __errorbar_plot(data: DataFrame, x: str, ax: Axes, _is_higher_better: bool, hue: str, _log_scale: bool) -> Axes:
    return sns.pointplot(data=data, x=x, y="value", hue=hue, errorbar=lambda v: (v.min(), v.max()), capsize=.2, ax=ax)


def errorbar_plot(data: DataFrame, title: str, y_name: str, x: str = "exp_name", x_name: str = "Experiment Name"):
    data = data.rename(columns={"exp_name": "Experiment Name"})
    if x == "exp_name":
        x = "Experiment Name"

    fig, ax = __basic_plot(__errorbar_plot, data, x,
                           x_name, y_name, title, hue=x)
    return fig


def __hist_plot(data: DataFrame, x: str, ax: Axes, _is_higher_better: bool, hue: str, _log_scale: bool) -> Axes:
    return sns.histplot(data=data, x=x, hue=hue, ax=ax, stat="probability")


def hist_plot(data: DataFrame, title: str, x_name: str, y_name: str = "Probability"):
    data = data.rename(columns={"exp_name": "Experiment Name"})
    x = "value"
    fig, ax = __basic_plot(__hist_plot, data, x, x_name,
                           y_name, title, hue="Experiment Name")

    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"{int(val):,}")
    )

    ax.set_xlim(xmin=0)

    return fig, ax


def __ecdf_plot(data: DataFrame, x: str, ax: Axes, _is_higher_better: bool, hue: str, log_scale: bool) -> Axes:
    return sns.ecdfplot(
        data=data,
        x=x,
        hue=hue,
        ax=ax,
        linewidth=2,
        log_scale=log_scale,
    )


def ecdf_plot(data: DataFrame, title: str, x_name: str, y_name: str = "Cumulative Probability", log_scale: bool = False, is_int_axis: bool = True):
    data = data.rename(columns={"exp_name": "Experiment Name"})
    x = "value"
    fig, ax = __basic_plot(__ecdf_plot, data, x, x_name, y_name,
                           title, hue="Experiment Name", log_scale=log_scale)

    if is_int_axis:
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda val, _: f"{int(val):,}")
        )

    ax.set_xlim(xmin=0)

    return fig


def ecdf_latency_plot(
    data: DataFrame,
    title: str,
    x_name: str = "Latency (ms)",
    y_name: str = "Cumulative Probability",
    log_scale: bool = False,
    is_int_axis: bool = False,
    percentiles: Sequence[float] = (99,),
    show_percentiles: bool = True,
    clip_percentile: float | None = None,
    legend_loc: str = "upper left",
):
    data = data.rename(columns={"exp_name": "Experiment Name"})
    x = "value"
    group_col = "Experiment Name" if "Experiment Name" in data.columns else "exp_name"
    if not data.empty:
        num_series = data[group_col].nunique() if group_col in data.columns else 1
        fig_width = min(14, max(6, 4 + num_series * 1.5))
        fig, ax = plt.subplots(figsize=(fig_width, 6))
    else:
        fig, ax = plt.subplots(figsize=(6, 6))

    hue_order = list(pd.unique(data[group_col])) if group_col in data.columns else None
    palette = sns.color_palette(n_colors=len(hue_order)) if hue_order else None

    sns.ecdfplot(
        data=data,
        x=x,
        hue=group_col if group_col in data.columns else None,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        linewidth=2,
        log_scale=log_scale,
    )

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(ymin=0)

    if is_int_axis:
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda val, _: f"{int(val):,}")
        )

    group_values = list(pd.unique(data[group_col])) if group_col in data.columns else []
    handles, labels = ax.get_legend_handles_labels()

    if show_percentiles and percentiles:
        color_by_label: dict[str, str] = {}
        for handle, label in zip(handles, labels):
            if not label:
                continue
            if hasattr(handle, "get_color"):
                color_by_label[str(label)] = handle.get_color()

        if not color_by_label:
            if palette and group_values:
                for label, color in zip(group_values, palette):
                    color_by_label[str(label)] = color

        percentiles_sorted = sorted({float(p) for p in percentiles if 0 <= float(p) <= 100})
        if percentiles_sorted:
            for p in percentiles_sorted:
                qs = data.groupby(group_col)[x].quantile(p / 100)
                for label, value in qs.items():
                    color = color_by_label.get(str(label), None)
                    ax.axvline(
                        value,
                        color=color or "gray",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.7,
                        label="_nolegend_",
                    )

    if clip_percentile is not None:
        clip_val = float(clip_percentile)
        if 0 < clip_val < 100:
            qs = data.groupby(group_col)[x].quantile(clip_val / 100)
            max_clip = float(qs.max()) if not qs.empty else None
            if max_clip is not None:
                ax.set_xlim(right=max_clip)

    if log_scale:
        positive = data[x][data[x] > 0]
        if not positive.empty:
            min_pos = float(positive.min())
            ax.set_xlim(left=max(min_pos * 0.9, min_pos * 0.1))
    else:
        ax.set_xlim(xmin=0)

    if legend_loc and labels:
        ax.legend(loc=legend_loc)

    return fig


def __kde_plot(data: DataFrame, x: str, ax: Axes, _is_higher_better: bool, hue: str, log_scale: bool) -> Axes:
    return sns.kdeplot(
        data=data,
        x=x,
        hue=hue,
        ax=ax,
        fill=True,
        linewidth=2,
        bw_adjust=.5,
        cut=0,  # TODO: Check if should be used
        log_scale=log_scale,
    )


def kde_plot(data: DataFrame, title: str, x_name: str, y_name: str = "Density", log_scale=False):
    data = data.rename(columns={"exp_name": "Experiment Name"})
    x = "value"
    fig, ax = __basic_plot(__kde_plot, data, x, x_name, y_name,
                           title, hue="Experiment Name", log_scale=log_scale)

    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"{int(val):,}")
    )

    ax.set_xlim(xmin=0)

    return fig


def __box_plot(data: DataFrame, x: str, ax: Axes, _is_higher_better: bool, _hue: str, _log_scale: bool) -> Axes:
    return sns.boxplot(data=data, x=x, y="value", ax=ax)


def boxplot(data: DataFrame, y_name: str, title: str, x: str = "exp_name", x_name: str = "Experiment Name"):
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        ax.set_ylim(bottom=0)
        return fig

    fig, _ = __basic_plot_by_categories(__box_plot, data, x, x_name, y_name, title)
    return fig


def __summary_box_plot(data: DataFrame, x: str, ax: Axes, _is_higher_better: bool, _hue: str, _log_scale: bool) -> Axes:
    order = data[x].tolist()
    records = []
    for _, row in data.iterrows():
        label = row[x]
        records.extend([
            {x: label, "value": row["lower_whisker"]},
            {x: label, "value": row["q1"]},
            {x: label, "value": row["median"]},
            {x: label, "value": row["q3"]},
            {x: label, "value": row["upper_whisker"]},
        ])

    synth = pd.DataFrame.from_records(records)
    return sns.boxplot(data=synth, x=x, y="value", order=order, ax=ax, showfliers=False)


def boxplot_from_summary(
        data: DataFrame,
        y_name: str,
        title: str,
        x: str = "exp_name",
        x_name: str = "Experiment Name",
):
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        ax.set_ylim(bottom=0)
        return fig

    required = {x, "q1", "q3", "median", "lower_whisker", "upper_whisker"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for summary boxplot: {', '.join(sorted(missing))}")

    fig, _ = __basic_plot_by_categories(__summary_box_plot, data, x, x_name, y_name, title)
    return fig


def __stacked_bar_plot(data: DataFrame, x, ax: Axes, is_higher_better: bool, _hue: str, _log_scale: bool):
    stats = (
        data
        .groupby(x)["value"]
        .agg(min_val="min", avg="mean", max_val="max")
        .reset_index()
    )

    max_color_idx = 0 if is_higher_better else 1
    min_color_idx = 1 if is_higher_better else 0
    min_alpha = 1 if is_higher_better else 0.3
    max_alpha = 0.3 if is_higher_better else 1
    min_line_style = "solid" if is_higher_better else "dashed"
    max_line_style = "dashed" if is_higher_better else "solid"
    min_hatch = "/" if is_higher_better else "o"
    max_hatch = "o" if is_higher_better else "/"

    cmap = cm.get_cmap("Set2")
    max_color = mcolors.to_hex(cmap(max_color_idx))
    min_color = mcolors.to_hex(cmap(min_color_idx))

    min_bar_height = stats["min_val"]
    max_bar_bottom = min_bar_height if is_higher_better else 0
    max_bar_height = stats["max_val"] - \
        stats["min_val"] if is_higher_better else stats["max_val"]
    max_bar_label = "Best-case" if is_higher_better else "Worst-case"
    min_bar_label = "Worst-case" if is_higher_better else "Best-case"

    max_bars = ax.bar(stats[x], max_bar_height, color=max_color, alpha=max_alpha, edgecolor="black", hatch=max_hatch,
                      linestyle=max_line_style, bottom=max_bar_bottom,
                      label=max_bar_label)

    min_bars = ax.bar(stats[x], min_bar_height, color=min_color, alpha=min_alpha, edgecolor="black", hatch=min_hatch,
                      linestyle=min_line_style, label=min_bar_label)
    for min_bar in min_bars:
        min_bar.set_linewidth(1)
    for max_bar in max_bars:
        max_bar.set_linewidth(1)

    for text in ax.legend(fontsize=12, facecolor="white", edgecolor="black", frameon=True).get_texts():
        text.set_color("black")

    return ax


def stacked_bar_plot(data: DataFrame, y_name: str, title: str, x: str = "exp_name", x_name: str = "Experiment Name",
                     base_name: str = None,
                     is_higher_better: bool = True):
    base_data = None

    if base_name is not None:
        base_name = exp_name_mapping[base_name] if base_name is not None else None
        base_data = data.loc[data[x] == base_name]

        data = data[data[x] != base_name]
    fig, ax = __basic_plot(__stacked_bar_plot, data, x,
                           x_name, y_name, title, is_higher_better)

    baseline_color = 'red'
    baseline_error_alpha = 0.2
    baseline_line_style = 'dotted'
    baseline_line_width = 1.5

    if base_data is not None:
        ax.axhspan(base_data["value"].min(), base_data["value"].max(
        ), color=baseline_color, alpha=baseline_error_alpha)

        ax.axhline(xmin=ax.get_xlim()[0], y=base_data["value"].mean(), color=baseline_color,
                   linestyle=baseline_line_style,
                   linewidth=baseline_line_width)

        handles, labels = ax.get_legend_handles_labels()

        labels = [base_name] + labels
        baseline_legend_line = Line2D([0], [1], color=baseline_color, linestyle=baseline_line_style,
                                      linewidth=baseline_line_width)
        baseline_legend_rect = Patch(
            facecolor=baseline_color, alpha=baseline_error_alpha, label=base_name)
        handles = [(baseline_legend_line, baseline_legend_rect)] + handles

        ax.legend(handles, labels)

    return fig


def event_throughput_line_plot(
        data: DataFrame,
        title: str,
        x_name: str = "Sample",
        y_name: str = "Event Throughput",
        hue: str = "repetition"
):
    if hue not in data.columns and not data.empty:
        raise ValueError(f"Column '{hue}' not found in provided data")

    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        unique_series = data[hue].nunique() if hue in data else 1
        fig_width = max(6, unique_series * 2)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        sns.lineplot(
            data=data,
            x="index",
            y="value",
            hue=hue,
            units=hue,
            estimator=None,
            linewidth=2,
            ax=ax,
        )

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if hue in data.columns and not data.empty:
        ax.legend(title=hue, fontsize=12)

    return fig, ax


def event_throughput_mean_ci_line_plot(
        data: DataFrame,
        title: str,
        x_name: str = "Time (s)",
        y_name: str = "Event Throughput",
        hue: str = "exp_name",
        ci_alpha: float = 0.2,
        line_width: float = 2.0,
):
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        return fig, ax

    required_columns = {"index", "value", "ci_low", "ci_high"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for mean/CI plot: {', '.join(sorted(missing))}")

    unique_series = data[hue].nunique() if hue in data.columns else 1
    fig_width = max(6, unique_series * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    if hue in data.columns:
        for label, group in data.groupby(hue):
            group = group.sort_values("index")
            ax.plot(group["index"], group["value"], label=label, linewidth=line_width)
            ax.fill_between(group["index"], group["ci_low"], group["ci_high"], alpha=ci_alpha)
        ax.legend(title=hue, fontsize=12)
    else:
        group = data.sort_values("index")
        ax.plot(group["index"], group["value"], linewidth=line_width)
        ax.fill_between(group["index"], group["ci_low"], group["ci_high"], alpha=ci_alpha)

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    return fig, ax


def latency_mean_ci_line_plot(
        data: DataFrame,
        title: str,
        x_name: str = "Time (s)",
        y_name: str = "Latency (ms)",
        hue: str = "exp_name",
        ci_alpha: float = 0.2,
        line_width: float = 2.0,
):
    return event_throughput_mean_ci_line_plot(
        data,
        title=title,
        x_name=x_name,
        y_name=y_name,
        hue=hue,
        ci_alpha=ci_alpha,
        line_width=line_width,
    )


def event_throughput_boxplot(
        data: DataFrame,
        title: str,
        x: str = "repetition",
        x_name: str = "Repetition",
        y_name: str = "Event Throughput",
        hue: str | None = None,
        legend: bool = True,
):
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        ax.set_ylim(bottom=0)
        return fig, ax

    required_columns = {x, "value"}
    if hue:
        required_columns.add(hue)

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for boxplot: {', '.join(sorted(missing))}")

    num_categories = data[x].nunique()
    fig_width = max(6, num_categories * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.boxplot(data=data, x=x, y="value", hue=hue, ax=ax)

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(bottom=0)

    if hue and legend:
        ax.legend(title=hue, fontsize=12)

    return fig, ax


def repetition_throughput_boxplot(
        data: DataFrame,
        title: str,
        x: str = "exp_name",
        x_name: str = "Experiment",
        y_name: str = "Throughput (tps)",
        hue: str | None = None,
        legend: bool = True,
):
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        ax.set_ylim(bottom=0)
        return fig, ax

    required_columns = {x, "value"}
    if hue:
        required_columns.add(hue)

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for boxplot: {', '.join(sorted(missing))}")

    num_categories = data[x].nunique()
    fig_width = max(6, num_categories * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    sns.boxplot(data=data, x=x, y="value", hue=hue, ax=ax)

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(bottom=0)

    if hue and legend:
        ax.legend(title=hue, fontsize=12)

    return fig, ax


def latency_percentile_plot(
        data: DataFrame,
        title: str,
        x_name: str = "Percentile",
        y_name: str = "Latency (ms)",
        hue: str | None = "exp_name",
        show_repetitions: bool = False,
        log_scale: bool = False,
        legend: bool = True,
):
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        return fig, ax

    required_columns = {"percentile", "value"}
    if hue:
        required_columns.add(hue)
    if show_repetitions:
        required_columns.add("repetition")

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for latency percentile plot: {', '.join(sorted(missing))}")

    percentiles = sorted(data["percentile"].unique())
    fig_width = max(6, len(percentiles) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    if show_repetitions and "repetition" in data.columns:
        sns.lineplot(
            data=data,
            x="percentile",
            y="value",
            hue=hue,
            style="repetition",
            markers=True,
            dashes=False,
            estimator=None,
            ax=ax,
        )
        ax.set_xticks(percentiles)
        ax.set_xticklabels([f"p{p:g}" for p in percentiles])
    else:
        sns.pointplot(
            data=data,
            x="percentile",
            y="value",
            hue=hue,
            order=percentiles,
            errorbar=None,
            ax=ax,
        )
        ax.set_xticklabels([f"p{p:g}" for p in percentiles])

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)

    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)

    if hue and legend:
        ax.legend(title=hue, fontsize=12)

    return fig, ax


def latency_distribution_plot(
        data: DataFrame,
        title: str,
        x_name: str = "Latency (ms)",
        y_name: str = "Frequency",
        hue: str | None = "exp_name",
        bins: str | int = "auto",
        stat: str = "count",
        kde: bool = False,
        log_scale: bool = False,
        percentile_lines: Sequence[float] | None = (95, 99),
        percentile_color: str = "red",
        percentile_style: str = "--",
        annotate_percentiles: bool = True,
        annotation_offset_points: int = 6,
        legend: bool = False,
):
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        if log_scale:
            ax.set_xscale("log")
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        return fig, ax

    required_columns = {"value"}
    if hue:
        required_columns.add(hue)

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for latency distribution plot: {', '.join(sorted(missing))}")

    values_all = pd.to_numeric(data["value"], errors="coerce").dropna().to_numpy()
    if values_all.size == 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        if log_scale:
            ax.set_xscale("log")
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        return fig, ax

    stat_key = stat.strip().lower()
    if stat_key not in {"count", "probability", "density", "percent"}:
        raise ValueError("stat must be one of: count, probability, density, percent")

    if isinstance(bins, (str, int)):
        bin_edges = np.histogram_bin_edges(values_all, bins=bins)
    else:
        bin_edges = np.asarray(list(bins), dtype=float)
        if bin_edges.ndim != 1 or bin_edges.size < 2:
            raise ValueError("bins must define at least two edges")

    num_categories = data[hue].nunique() if hue in data.columns else 1
    fig_width = max(6, num_categories * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    groups = data.groupby(hue) if hue and hue in data.columns else [(None, data)]
    max_y = 0.0

    for label, group in groups:
        values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue

        counts, _ = np.histogram(values, bins=bin_edges)
        counts = counts.astype(float)

        if stat_key == "probability":
            total = counts.sum()
            if total > 0:
                counts = counts / total
        elif stat_key == "percent":
            total = counts.sum()
            if total > 0:
                counts = counts / total * 100
        elif stat_key == "density":
            total = counts.sum()
            widths = np.diff(bin_edges)
            if total > 0:
                counts = counts / (total * widths)

        max_y = max(max_y, float(counts.max()) if counts.size else 0.0)

        y_step = np.r_[counts, counts[-1] if counts.size else 0.0]
        if label is None or not hue:
            ax.step(bin_edges, y_step, where="post")
        else:
            ax.step(bin_edges, y_step, where="post", label=str(label))

    if kde:
        rng = np.random.default_rng(0)
        for label, group in groups:
            values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                continue
            if values.size > 20000:
                values = rng.choice(values, size=20000, replace=False)
            sns.kdeplot(
                values,
                ax=ax,
                label=None if label is None or not hue else str(label),
                common_norm=False,
            )

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)
    if max_y > 0:
        ax.set_ylim(0, max_y * 1.02)
    else:
        ax.set_ylim(bottom=0)

    if log_scale:
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=0)

    if percentile_lines:
        percentiles = np.nanpercentile(values_all, percentile_lines)
        for percentile, value in zip(percentile_lines, percentiles):
            if log_scale and value <= 0:
                continue
            ax.axvline(
                value,
                color=percentile_color,
                linestyle=percentile_style,
                linewidth=1.5,
            )
            if annotate_percentiles:
                ax.annotate(
                    f"p{percentile:g}",
                    xy=(value, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -max(1, annotation_offset_points)),
                    textcoords="offset points",
                    rotation=90,
                    ha="right",
                    va="top",
                    color=percentile_color,
                )

    if hue and legend:
        ax.legend(title=hue, fontsize=12)

    return fig, ax


def latency_throughput_plot(
        data: DataFrame,
        title: str,
        x: str = "throughput",
        y: str = "latency",
        x_name: str = "Throughput (tps)",
        y_name: str = "Latency (ms)",
        hue: str | None = "exp_name",
        style: str | None = None,
        log_x: bool = False,
        log_y: bool = False,
        legend: bool = True,
        connect: bool = True,
):
    if data.empty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlabel(x_name, fontsize=18)
        ax.set_ylabel(y_name, fontsize=18)
        ax.set_title(title, fontsize=20)
        if log_x:
            ax.set_xscale("log")
        else:
            ax.set_xlim(left=0)
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        return fig, ax

    required_columns = {x, y}
    if hue:
        required_columns.add(hue)
    if style:
        required_columns.add(style)

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for latency throughput plot: {', '.join(sorted(missing))}")

    num_categories = data[hue].nunique() if hue in data.columns else 1
    fig_width = max(6, num_categories * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    hue_order = list(pd.unique(data[hue])) if hue and hue in data.columns else None
    palette = sns.color_palette(n_colors=len(hue_order)) if hue_order else None

    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        style=style,
        ax=ax,
    )

    if connect and hue and hue in data.columns:
        palette_map = dict(zip(hue_order or [], palette or []))
        for label, group in data.groupby(hue):
            group = group.sort_values(x)
            ax.plot(
                group[x],
                group[y],
                linewidth=1.5,
                alpha=0.7,
                color=palette_map.get(label),
            )

    ax.set_xlabel(x_name, fontsize=18)
    ax.set_ylabel(y_name, fontsize=18)
    ax.set_title(title, fontsize=20)

    if log_x:
        ax.set_xscale("log")
    else:
        ax.set_xlim(left=0)

    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)

    if hue and legend:
        ax.legend(title=hue, fontsize=12)

    return fig, ax


def latency_throughput_dual_line_plot(
        data: DataFrame,
        throughput_title: str = "Throughput",
        latency_title: str = "Latency",
        x: str = "parallelism",
        throughput_col: str = "throughput",
        latency_col: str = "latency",
        x_name: str = "Parallelism",
        throughput_name: str = "Throughput (tps)",
        latency_name: str = "Latency (ms)",
        hue: str | None = "exp_name",
        marker: str | None = "o",
        legend: bool = True,
        sharex: bool = True,
        throughput_ci: float | None = 95,
        latency_ci: float | None = None,
        ci_alpha: float = 0.2,
):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=sharex)

    if data.empty:
        axs[0].set_title(throughput_title)
        axs[0].set_xlabel(x_name, fontsize=14)
        axs[0].set_ylabel(throughput_name, fontsize=14)
        axs[0].set_ylim(bottom=0)

        axs[1].set_title(latency_title)
        axs[1].set_xlabel(x_name, fontsize=14)
        axs[1].set_ylabel(latency_name, fontsize=14)
        axs[1].set_ylim(bottom=0)

        return fig, axs

    required_columns = {x, throughput_col, latency_col}
    if hue:
        required_columns.add(hue)
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns for dual line plot: {', '.join(sorted(missing))}")

    plot_data = data.copy()
    x_numeric = pd.to_numeric(plot_data[x], errors="coerce")
    if x_numeric.notna().all():
        plot_data["_plot_x"] = x_numeric.astype(float)
    else:
        plot_data["_plot_x"] = plot_data[x].astype(str)

    sort_cols = ["_plot_x"]
    if hue and hue in plot_data.columns:
        sort_cols = [hue] + sort_cols
    plot_data = plot_data.sort_values(sort_cols)

    sns.lineplot(
        data=plot_data,
        x="_plot_x",
        y=throughput_col,
        hue=hue,
        marker=marker,
        estimator="mean",
        errorbar=None if throughput_ci is None else ("ci", throughput_ci),
        err_style="band",
        err_kws={"alpha": ci_alpha},
        sort=False,
        ax=axs[0],
    )
    axs[0].set_title(throughput_title)
    axs[0].set_xlabel(x_name, fontsize=14)
    axs[0].set_ylabel(throughput_name, fontsize=14)
    axs[0].set_ylim(bottom=0)

    if hue and legend:
        axs[0].legend(title=hue, fontsize=12)
    elif axs[0].get_legend() is not None:
        axs[0].get_legend().remove()

    sns.lineplot(
        data=plot_data,
        x="_plot_x",
        y=latency_col,
        hue=hue,
        marker=marker,
        estimator="mean",
        errorbar=None if latency_ci is None else ("ci", latency_ci),
        err_style="band",
        err_kws={"alpha": ci_alpha},
        legend=False,
        sort=False,
        ax=axs[1],
    )
    axs[1].set_title(latency_title)
    axs[1].set_xlabel(x_name, fontsize=14)
    axs[1].set_ylabel(latency_name, fontsize=14)
    axs[1].set_ylim(bottom=0)
    if axs[1].get_legend() is not None:
        axs[1].get_legend().remove()

    return fig, axs


def __operator_img(data: DataFrame, graph_name: str, key: str, op_type: OpType):
    graph = get_rep_with(data, key, op_type)["operator_graph"]

    graph.render(f"plots/{graph_name}", format="png")

    img = mpimg.imread(f"plots/{graph_name}.png")

    return img


def operator_plot(exp_data: dict, name: str, key: str, op_type: OpType):
    exp_name = exp_name_mapping[name]
    data = exp_data[exp_name]

    graph_name = f"{name}_graph"

    img = __operator_img(data, graph_name, key, op_type)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"{exp_name}\nOperator Graph")
    ax.imshow(img)
    ax.axis('off')
    return fig


def dag_plot(exp_data: dict, name: str):
    exp_name = exp_name_mapping[name]
    data = exp_data[exp_name]

    graph = data["dag"]
    graph.render(f"plots/{name}_dag", format="png")
    img = mpimg.imread(f"plots/{name}_dag.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"{exp_name}\nDAG")
    ax.imshow(img)
    ax.axis('off')

    return fig


def min_max_operator_plot(exp_data: dict, name: str, key: str):
    exp_name = exp_name_mapping[name]
    data = exp_data[exp_name]

    min_graph_name = f"{name}_min_graph"
    max_graph_name = f"{name}_max_graph"

    min_img = __operator_img(data, min_graph_name, key, OpType.MIN)
    max_img = __operator_img(data, max_graph_name, key, OpType.MAX)

    fig, axs = plt.subplots(1, 2, figsize=(8, 6))

    fig.suptitle(f"{exp_name}\nOperator Graphs")

    axs[0].imshow(min_img)
    axs[0].set_title(f"Min {key}")
    axs[0].axis('off')

    axs[1].imshow(max_img)
    axs[1].set_title(f"Max {key}")
    axs[1].axis('off')

    fig.tight_layout()
    return fig
