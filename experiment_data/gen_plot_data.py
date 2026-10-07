from enum import Enum
import ast
import csv

import pandas as pd
import numpy as np
from constants import exp_name_mapping
import os
import re
from scipy import stats


def gen_ecdf_data(
    df: pd.DataFrame,
    file_name: str,
    output_suffix: str = "",
    x_label: str = "Throughput (tps)",
    legend_strip: list[str] = None,
    max_points: int | None = None,
    tail_focus: float = 0.99,
    tail_share: float = 0.6,
    keep_quantiles: list[float] | None = None,
    uniform: bool = False,
    clip_percentile: float | None = None,
    marker_percentiles: list[float] | None = None,
):
    """
    Generate ECDF data for pgfplots visualization.
    
    Args:
        df: DataFrame containing the data with 'exp_name' and 'value' columns.
            If present, 'exp_key' will be used for output filenames.
        file_name: Base name for the output CSV files
        x_label: Label for x-axis
        legend_strip: List of substrings to remove from legend entries
        max_points: Cap points per series by quantile downsampling
        tail_focus: Quantile where tail sampling starts (e.g., 0.99)
        tail_share: Fraction of points dedicated to [tail_focus, 1]
        keep_quantiles: Additional quantiles to force include (values in [0, 1] or [0, 100])
        uniform: If True, use uniform quantile spacing across [0, 1]
        clip_percentile: If set, cap output to this percentile (values in [0, 100])
        marker_percentiles: If set, emit small CSVs with vertical lines for these percentiles
    """
    def _safe_key(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")

    def _normalize_quantiles(values: list[float]) -> list[float]:
        out: list[float] = []
        for q in values:
            q = float(q)
            if q > 1:
                q = q / 100.0
            out.append(q)
        return out

    output_dir = f"plot_data_{output_suffix}" if output_suffix else "plot_data"
    os.makedirs(output_dir, exist_ok=True)

    def _quantile_grid(n_points: int) -> np.ndarray:
        n_points = max(4, int(n_points))
        max_q = None
        if clip_percentile is not None:
            max_q = float(clip_percentile)
            if max_q > 1:
                max_q = max_q / 100.0
            max_q = min(max(max_q, 0.0), 1.0)

        if uniform:
            qs = np.linspace(0.0, 1.0, n_points, endpoint=True)
            if keep_quantiles:
                qs = np.concatenate([qs, np.array(_normalize_quantiles(keep_quantiles))])
            qs = np.clip(qs, 0.0, 1.0)
            if max_q is not None:
                qs = qs[qs <= max_q]
            qs = np.unique(qs)
            qs.sort()
            return qs

        tf = float(tail_focus)
        tf = min(max(tf, 0.5), 0.9999)
        ts = float(tail_share)
        ts = min(max(ts, 0.1), 0.9)
        n_tail = max(2, int(n_points * ts))
        n_body = max(2, n_points - n_tail)
        qs = np.concatenate([
            np.linspace(0.0, tf, n_body, endpoint=False),
            np.linspace(tf, 1.0, n_tail, endpoint=True),
        ])
        if keep_quantiles:
            qs = np.concatenate([qs, np.array(_normalize_quantiles(keep_quantiles))])
        qs = np.clip(qs, 0.0, 1.0)
        if max_q is not None:
            qs = qs[qs <= max_q]
        qs = np.unique(qs)
        qs.sort()
        return qs

    group_key = "exp_key" if "exp_key" in df.columns and df["exp_key"].notna().any() else "exp_name"
    for key_name, group in df.groupby(group_key):
        # Sort values for ECDF
        sorted_values = np.sort(group["value"].to_numpy())
        marker_qs = _normalize_quantiles(marker_percentiles) if marker_percentiles else []
        if max_points is not None and len(sorted_values) > max_points:
            qs = _quantile_grid(max_points)
            x_vals = np.quantile(sorted_values, qs, method="linear")
            y_vals = qs
        else:
            if clip_percentile is not None:
                max_q = float(clip_percentile)
                if max_q > 1:
                    max_q = max_q / 100.0
                max_q = min(max(max_q, 0.0), 1.0)
                if max_q < 1.0:
                    cut_idx = max(1, int(np.ceil(max_q * len(sorted_values))))
                    sorted_values = sorted_values[:cut_idx]
            x_vals = sorted_values
            y_vals = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        
        # Create step-like pattern
        x_step = []
        y_step = []
        
        # Add starting point at y=0
        x_step.append(x_vals[0])
        y_step.append(0)
        
        for i in range(len(x_vals)):
            # Add point for current step
            x_step.append(x_vals[i])
            y_step.append(y_vals[i])
            # Add point for next x value with same y (horizontal step)
            if i < len(x_vals) - 1:
                x_step.append(x_vals[i+1])
                y_step.append(y_vals[i])
        
        key = _safe_key(key_name)
        label = key_name
        if "exp_name" in group.columns and group["exp_name"].notna().any():
            label = group["exp_name"].iloc[0]
        records = [{
            "x": x,
            "ecdf": y,
            "key": key,
            "label": str(label).replace("\n", r"\\"),
        } for x, y in zip(x_step, y_step)]

        ecdf_df = pd.DataFrame(records)
        suffix = f"_{output_suffix}" if output_suffix else ""
        ecdf_df.to_csv(f"plot_data{suffix}/{file_name}_{key}_ecdf.csv", index=False)

        if marker_qs:
            for q in marker_qs:
                q = float(min(max(q, 0.0), 1.0))
                xq = float(np.quantile(sorted_values, q, method="linear"))
                marker_df = pd.DataFrame({
                    "x": [xq, xq],
                    "ecdf": [0.0, 1.0],
                    "key": [key, key],
                    "label": [str(label).replace("\n", r"\\")] * 2,
                    "percentile": [q * 100.0, q * 100.0],
                })
                p_tag = f"p{int(round(q * 100))}"
                marker_df.to_csv(
                    f"plot_data{suffix}/{file_name}_{key}_{p_tag}_line.csv",
                    index=False,
                )
    
    # Generate PGFPlots code
    gen_pgfplots_code(file_name, x_label, output_suffix=output_suffix, plot_type=PlotType.ECDF, legend_strip=legend_strip)


def gen_kde_data(df: pd.DataFrame, file_name: str, x_label: str = "Throughput (tps)", legend_strip: list[str] = None):
    """
    Generate KDE data for pgfplots visualization.
    
    Args:
        df: DataFrame containing the data with 'exp_name' and 'value' columns.
            If present, 'exp_key' will be used for output filenames.
        file_name: Base name for the output CSV files
        x_label: Label for x-axis
        legend_strip: List of substrings to remove from legend entries
    """
    def _safe_key(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")

    for name, group in df.groupby("exp_name"):
        # Get values for KDE
        values = group["value"].values
        
        # Skip if there's only one data point
        if len(values) <= 1:
            print(f"Warning: Skipping KDE for {name} as it has only {len(values)} data point(s)")
            continue
        
        # Create a range of x values for the KDE
        x_min = values.min()
        x_max = values.max()
        x = np.linspace(x_min, x_max, 1000)
        
        # Calculate KDE
        kde = stats.gaussian_kde(values, bw_method=0.5)
        y = kde(x)
        
        # Ensure the curve starts and ends at 0
        y[0] = 0
        y[-1] = 0
        
        key = None
        if "exp_key" in group.columns:
            keys = group["exp_key"].dropna().unique()
            if len(keys) == 1:
                key = keys[0]
            elif len(keys) > 1:
                key = keys[0]
        if key is None:
            key = name
        key = _safe_key(key)
        records = [{
            "x": x_val,
            "kde": y_val,
            "key": key,
            "label": name.replace("\n", r"\\"),
        } for x_val, y_val in zip(x, y)]

        kde_df = pd.DataFrame(records)
        kde_df.to_csv(f"plot_data/{file_name}_{key}_kde.csv", index=False)
    
    # Generate PGFPlots code
    gen_pgfplots_code(file_name, x_label, plot_type=PlotType.KDE, legend_strip=legend_strip)


class PlotType(Enum):
    ECDF = "ECDF"
    KDE = "KDE"


def gen_pgfplots_code(file_name: str, x_label: str, output_suffix: str = "",
                      width: str = "\\linewidth", height: str = "0.6\\linewidth",
                      plot_type: PlotType = PlotType.ECDF, legend_strip: list[str] = None):
    """
    Generate pgfplots code for visualization.
    
    Args:
        file_name: Base name of the CSV files
        x_label: Label for x-axis
        width: Width of the plot
        height: Height of the plot
        plot_type: Type of plot (ECDF or KDE)
        legend_strip: List of substrings to remove from legend entries
    """
    match plot_type:
        case PlotType.ECDF:
            y_label = "Cumulative Probability"
        case PlotType.KDE:
            y_label = "Density"

    # Find all matching CSV files
    folder_suffix = f"_{output_suffix}" if output_suffix else ""
    plot_data_dir = f"plot_data{folder_suffix}"
    plot_code_dir = f"plot_code{folder_suffix}"
    suffix = "_ecdf.csv" if plot_type == PlotType.ECDF else "_kde.csv"

    if not os.path.isdir(plot_data_dir):
        raise ValueError(f"Data directory does not exist: {plot_data_dir}")

    os.makedirs(plot_code_dir, exist_ok=True)

    matching_files = []
    for f in os.listdir(plot_data_dir):
        if f.startswith(f"{file_name}_") and f.endswith(suffix):
            matching_files.append(f)
    
    if not matching_files:
        raise ValueError(f"No valid data files found for {file_name} with type {plot_type.value}")
    
    # Define line styles and fill patterns for the plots
    line_styles = ['solid', 'dashed', 'dotted', 'dashdotted', 'densely dashed', 'densely dotted', 
                  'loosely dashed', 'loosely dotted', 'loosely dashdotted', 'densely dashdotted']
    fill_patterns = ['north east lines', 'north west lines', 'crosshatch', 'crosshatch dots',
                    'horizontal lines', 'vertical lines', 'grid', 'dots',
                    'bricks', 'checkerboard']
    
    # For KDE plots, find the maximum y value across all files
    ymax = 1.0  # Default for ECDF
    ymin = 0.0  # Default for ECDF
    if plot_type == PlotType.KDE:
        max_y = 0
        min_y = float('inf')
        for csv_file in matching_files:
            df = pd.read_csv(f"{plot_data_dir}/{csv_file}")
            max_y = max(max_y, df["kde"].max())
            min_y = min(min_y, df["kde"].min())
        ymax = max_y * 1.1  # Add 10% padding

    # Regular plot
    code = f"""\\begin{{tikzpicture}}
% Define colors
\\definecolor{{color1}}{{RGB}}{{31,119,180}}
\\definecolor{{color2}}{{RGB}}{{255,127,14}}
\\definecolor{{color3}}{{RGB}}{{44,160,44}}
\\definecolor{{color4}}{{RGB}}{{214,39,40}}
\\definecolor{{color5}}{{RGB}}{{148,103,189}}
\\definecolor{{color6}}{{RGB}}{{140,86,75}}
\\definecolor{{color7}}{{RGB}}{{227,119,194}}
\\definecolor{{color8}}{{RGB}}{{127,127,127}}
\\definecolor{{color9}}{{RGB}}{{188,189,34}}
\\definecolor{{color10}}{{RGB}}{{23,190,207}}

\\begin{{axis}}[
    width={width},
    height={height},
    xlabel={{{x_label}}},
    ylabel={{{y_label}}},
    xmin=0,
    ymin={ymin},
    ymax={ymax},
    ymajorgrids=true,
    xmajorgrids=true,
    legend pos=north west,
    legend style={{nodes={{scale=0.5, transform shape}}, legend cell align=left}},
    every axis plot/.append style={{very thick}},
    tick label style={{/pgf/number format/fixed}},
    x tick label style={{/pgf/number format/1000 sep=,}},
    scaled x ticks=false"""
    code += "\n]\n\n"

    # Sort files according to exp_name_mapping order
    file_order = {key: i for i, key in enumerate(exp_name_mapping.keys())}
    def _sort_key(csv_name: str) -> tuple[int, int | float, str]:
        key = csv_name[len(file_name) + 1:-len(suffix)]
        if key in file_order:
            return (0, file_order[key], key)
        return (1, float("inf"), key)

    matching_files.sort(key=_sort_key)

    # Add plots
    for i, csv_file in enumerate(matching_files):
        key = csv_file[len(file_name) + 1:-len(suffix)]
        color_name = f"color{i+1}"
        y_col = "ecdf" if plot_type == PlotType.ECDF else "kde"
        
        if plot_type == PlotType.ECDF:
            line_style = line_styles[i % len(line_styles)]
            legend_text = exp_name_mapping.get(key, key)
            try:
                label_df = pd.read_csv(f"{plot_data_dir}/{csv_file}", usecols=["label"], nrows=1)
                if not label_df.empty and "label" in label_df.columns:
                    raw_label = str(label_df["label"].iloc[0]).strip()
                    if raw_label:
                        legend_text = raw_label.replace("\\\\", " ")
            except Exception:
                pass
            legend_text = legend_text.replace("\n", " ").replace("%", "\\%")
            if legend_strip:
                for s in legend_strip:
                    legend_text = legend_text.replace(s, "")
            code += f"\\addplot[mark=none, color={color_name}, {line_style}] table[x=x, y={y_col}, col sep=comma] {{{plot_data_dir}/{csv_file}}};\n"
            code += f"\\addlegendentry{{{legend_text.strip()}}}\n"
        else:
            pattern = fill_patterns[i % len(fill_patterns)]
            legend_text = exp_name_mapping.get(key, key).replace("\n", " ").replace("%", "\\%")
            if legend_strip:
                for s in legend_strip:
                    legend_text = legend_text.replace(s, "")
            code += f"\\addlegendimage{{area legend, line width=0pt, fill={color_name}!20, pattern={pattern}, pattern color={color_name}}}\n"
            code += f"\\addplot[mark=none, color={color_name}, fill={color_name}!20, pattern={pattern}, pattern color={color_name}, forget plot] table[x=x, y={y_col}, col sep=comma] {{{plot_data_dir}/{csv_file}}};\n"
            code += f"\\addlegendentry{{{legend_text.strip()}}}\n"
    
    code += "\\end{axis}\n\\end{tikzpicture}"

    with open(f"{plot_code_dir}/{file_name}_{plot_type.value.lower()}.tex", "w", encoding="utf8") as f:
        f.write(code)


def gen_event_line_data(
        df: pd.DataFrame,
        file_name: str,
        output_suffix: str = "",
        use_smoothed: bool = False
) -> list[dict[str, str]]:
    """Generate per-repetition CSV data suitable for pgfplots line plots.

    Args:
        df: DataFrame containing `exp_name`, `repetition`, `index`, and
            either `value` or `value_smoothed` columns. Optional columns
            `exp_name_tex`/`repetition_tex` improve legend handling.
        file_name: Base name for the output CSV files.
        output_suffix: Optional suffix that selects `plot_data_<suffix>`.
        use_smoothed: When True, prefer the `value_smoothed` column if present.

    Returns:
        A list of metadata dicts with the csv `path`, original `exp_name`,
        `repetition`, and TeX-ready labels for convenience when
        constructing pgfplots commands.
    """

    if df.empty:
        return []

    value_column = "value_smoothed" if use_smoothed and "value_smoothed" in df.columns else "value"
    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in DataFrame")

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_dir = f"plot_data{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    metadata: list[dict[str, str]] = []

    def _sanitize(part: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_-]+", "_", part)
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean or "unnamed"

    grouped = df.groupby(["exp_name", "repetition"])
    for (exp_name, repetition), group in grouped:
        exp_part = None
        if "exp_key" in group.columns:
            keys = group["exp_key"].dropna().unique()
            if len(keys) == 1:
                exp_part = keys[0]
            elif len(keys) > 1:
                exp_part = keys[0]
        if exp_part is None:
            exp_part = exp_name
        exp_part = _sanitize(str(exp_part))
        rep_part = _sanitize(repetition)

        output_path = os.path.join(output_dir, f"{file_name}_{exp_part}_{rep_part}.csv")

        export_df = group.sort_values("index")[["index", value_column]].rename(columns={
            "index": "sample",
            value_column: "value",
        })

        export_df.to_csv(output_path, index=False, na_rep="nan")

        label_tex = group["exp_name_tex"].iloc[0] if "exp_name_tex" in group.columns else exp_name
        repetition_tex = group["repetition_tex"].iloc[0] if "repetition_tex" in group.columns else repetition

        metadata.append({
            "path": output_path,
            "exp_name": exp_name,
            "repetition": repetition,
            "label_tex": label_tex,
            "repetition_tex": repetition_tex,
        })

    return metadata


def gen_event_line_ci_data(
        df: pd.DataFrame,
        file_name: str,
        output_suffix: str = "",
        value_col: str = "value",
        lower_col: str = "ci_low",
        upper_col: str = "ci_high",
        stop_at_last_above_for: list[str] | tuple[str, ...] | set[str] | None = None,
        stop_match_col: str = "exp_key",
        stop_threshold: float = 0.0,
        marker_suffix: str | None = None,
) -> list[dict[str, str]]:
    """Generate per-experiment CSV data with mean and CI bands for pgfplots.

    Optional behavior:
      - stop_at_last_above_for: identifiers to trim trailing tail points.
        Matching uses stop_match_col when present (default exp_key), else exp_name.
      - stop_threshold: last point is selected from value_col > stop_threshold.
      - marker_suffix: if set and trimming is applied, emit a one-row marker CSV
        named: {file_name}_{exp_part}_{marker_suffix}.csv.
    """
    if df.empty:
        return []

    required = {"exp_name", "index", value_col, lower_col, upper_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for CI CSV: {', '.join(sorted(missing))}")

    normalized_suffix = output_suffix.strip().lstrip("_")
    suffix = f"_{normalized_suffix}" if normalized_suffix else ""
    output_dir = f"plot_data{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    metadata: list[dict[str, str]] = []
    stop_set = {str(v) for v in (stop_at_last_above_for or [])}

    def _sanitize(part: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_-]+", "_", part)
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean or "unnamed"

    grouped = df.groupby("exp_name")
    for exp_name, group in grouped:
        exp_part = None
        if "exp_key" in group.columns:
            keys = group["exp_key"].dropna().unique()
            if len(keys) == 1:
                exp_part = keys[0]
            elif len(keys) > 1:
                exp_part = keys[0]
        if exp_part is None:
            exp_part = exp_name
        exp_part = _sanitize(str(exp_part))

        output_path = os.path.join(output_dir, f"{file_name}_{exp_part}.csv")
        group_sorted = group.sort_values("index")

        match_value = str(exp_name)
        if stop_match_col in group.columns:
            match_values = group[stop_match_col].dropna().unique()
            if len(match_values) >= 1:
                match_value = str(match_values[0])

        marker_path: str | None = None
        export_group = group_sorted
        should_trim = bool(stop_set) and (match_value in stop_set or str(exp_name) in stop_set)
        if should_trim and not group_sorted.empty:
            value_series = pd.to_numeric(group_sorted[value_col], errors="coerce")
            valid_mask = value_series > float(stop_threshold)
            if bool(valid_mask.any()):
                last_valid_pos = int(np.flatnonzero(valid_mask.to_numpy())[-1])
                export_group = group_sorted.iloc[:last_valid_pos + 1]
                if marker_suffix:
                    marker_row = export_group.iloc[-1]
                    marker_path = os.path.join(output_dir, f"{file_name}_{exp_part}_{_sanitize(marker_suffix)}.csv")
                    marker_df = pd.DataFrame([{
                        "sample": marker_row["index"],
                        "value": marker_row[value_col],
                        "ci_low": marker_row[lower_col],
                        "ci_high": marker_row[upper_col],
                    }])
                    marker_df.to_csv(marker_path, index=False, na_rep="nan")

        export_df = export_group[["index", value_col, lower_col, upper_col]].rename(columns={
            "index": "sample",
            value_col: "value",
            lower_col: "ci_low",
            upper_col: "ci_high",
        })

        export_df.to_csv(output_path, index=False, na_rep="nan")

        label_tex = group["exp_name_tex"].iloc[0] if "exp_name_tex" in group.columns else exp_name

        metadata.append({
            "path": output_path,
            "exp_name": exp_name,
            "label_tex": label_tex,
            "marker_path": marker_path,
        })

    return metadata


def gen_latency_throughput_data(
        df: pd.DataFrame,
        file_name: str,
        output_suffix: str = "",
        x_col: str = "throughput",
        y_col: str = "latency",
) -> list[dict[str, str]]:
    """Generate per-experiment CSV data for latency-throughput scatter plots (pgfplots)."""
    if df.empty:
        return []

    required = {x_col, y_col}
    if "exp_name" in df.columns:
        required.add("exp_name")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for latency-throughput CSV: {', '.join(sorted(missing))}")

    normalized_suffix = output_suffix.strip().lstrip("_")
    suffix = f"_{normalized_suffix}" if normalized_suffix else ""
    output_dir = f"plot_data{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    metadata: list[dict[str, str]] = []

    def _sanitize(part: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_-]+", "_", part)
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean or "unnamed"

    if "exp_name" in df.columns:
        grouped = df.groupby("exp_name")
    else:
        grouped = [(file_name, df)]

    optional_cols = [col for col in ("parallelism", "repetition", "latency_percentile", "latency_unit") if col in df.columns]

    for exp_name, group in grouped:
        exp_part = None
        if "exp_key" in group.columns:
            keys = group["exp_key"].dropna().unique()
            if len(keys) >= 1:
                exp_part = keys[0]
        if exp_part is None:
            exp_part = exp_name
        exp_part = _sanitize(str(exp_part))

        output_path = os.path.join(output_dir, f"{file_name}_{exp_part}.csv")

        export_cols = [x_col, y_col] + optional_cols
        export_df = group[export_cols].rename(columns={
            x_col: "throughput",
            y_col: "latency",
        })

        export_df.to_csv(output_path, index=False, na_rep="nan")

        label_tex = group["exp_name_tex"].iloc[0] if "exp_name_tex" in group.columns else exp_name

        metadata.append({
            "path": output_path,
            "exp_name": exp_name,
            "label_tex": label_tex,
        })

    return metadata


def gen_parallelism_line_ci_data(
        df: pd.DataFrame,
        file_name: str,
        output_suffix: str = "",
        x_col: str = "parallelism",
        y_col: str = "throughput",
        ci: float | None = 0.95,
) -> list[dict[str, str]]:
    """Generate per-experiment line CSV data aggregated by x_col with optional CI bands.

    Output columns:
      - index: x value
      - value: mean(y)
      - ci_low / ci_high: CI bounds (or value when CI is disabled / n<=1)
      - n: sample count used for aggregation
    """
    if df.empty:
        return []

    required = {"exp_name", x_col, y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for parallelism line CI CSV: {', '.join(sorted(missing))}")

    normalized_suffix = output_suffix.strip().lstrip("_")
    suffix = f"_{normalized_suffix}" if normalized_suffix else ""
    output_dir = f"plot_data{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    metadata: list[dict[str, str]] = []

    def _sanitize(part: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_-]+", "_", part)
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean or "unnamed"

    ci_level = None if ci is None else float(ci)
    if ci_level is not None:
        if ci_level <= 0:
            ci_level = None
        elif ci_level > 1:
            ci_level = ci_level / 100.0
        ci_level = min(max(ci_level, 0.0), 1.0)

    for exp_name, group in df.groupby("exp_name"):
        exp_part = None
        if "exp_key" in group.columns:
            keys = group["exp_key"].dropna().unique()
            if len(keys) >= 1:
                exp_part = keys[0]
        if exp_part is None:
            exp_part = exp_name
        exp_part = _sanitize(str(exp_part))

        agg_rows: list[dict[str, float | int | str]] = []
        for x_value, x_group in group.groupby(x_col):
            values = pd.to_numeric(x_group[y_col], errors="coerce").dropna().to_numpy()
            n = int(values.size)
            if n == 0:
                continue

            mean = float(np.mean(values))
            if ci_level is None or n <= 1:
                ci_low = mean
                ci_high = mean
            else:
                sem = stats.sem(values, nan_policy="omit")
                if not np.isfinite(sem) or sem == 0:
                    ci_low = mean
                    ci_high = mean
                else:
                    ci_low_f, ci_high_f = stats.t.interval(ci_level, df=n - 1, loc=mean, scale=sem)
                    if not np.isfinite(ci_low_f) or not np.isfinite(ci_high_f):
                        ci_low = mean
                        ci_high = mean
                    else:
                        ci_low = float(ci_low_f)
                        ci_high = float(ci_high_f)

            agg_rows.append({
                "index": x_value,
                "value": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
            })

        if not agg_rows:
            continue

        export_df = pd.DataFrame(agg_rows)
        index_numeric = pd.to_numeric(export_df["index"], errors="coerce")
        if index_numeric.notna().all():
            export_df["index"] = index_numeric.astype(float)
            export_df = export_df.sort_values("index")
        else:
            export_df["index"] = export_df["index"].astype(str)
            export_df = export_df.sort_values("index")

        output_path = os.path.join(output_dir, f"{file_name}_{exp_part}.csv")
        export_df.to_csv(output_path, index=False, na_rep="nan")

        label_tex = group["exp_name_tex"].iloc[0] if "exp_name_tex" in group.columns else exp_name
        metadata.append({
            "path": output_path,
            "exp_name": exp_name,
            "label_tex": label_tex,
        })

    return metadata


def gen_tput_table_data(
        df: pd.DataFrame,
        file_name: str,
        output_suffix: str = "",
        value_col: str = "value",
        split_flinke2c_variants: bool = False,
) -> str:
    """Generate throughput summary table CSV grouped by query/system.

    Output columns are multi-level:
      - (Query, ""), (System, "")
      - (Environment, mean|min|max|std|rel_std) for each discovered environment in order
    """
    required = {"exp_name", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for throughput table CSV: {', '.join(sorted(missing))}")

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_dir = f"plot_data{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    def _safe_text(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value)

    def _infer_system(raw_key: str, label: str) -> str | None:
        key = raw_key.lower()
        lbl = label.lower()
        if "flinke2c_bu" in key or "bottom-up" in lbl:
            return "flinke2c_bu" if split_flinke2c_variants else "flinke2c"
        if "flinke2c_td" in key or "top-down" in lbl:
            return "flinke2c_td" if split_flinke2c_variants else "flinke2c"
        if "flinke2c" in key or "flinke2c" in lbl:
            return "flinke2c"
        if key.startswith("nes") or re.search(r"\bnes\b", lbl):
            return "nes"
        if key.startswith("flink") or key.startswith("base_") or "baseline" in lbl:
            return "flink"
        return None

    def _infer_query(raw_key: str, label: str) -> str:
        key = raw_key.lower()
        lbl = label.lower()
        key_match = re.search(r"(?:^|_)q(\d+[a-z0-9]*)", key)
        if key_match:
            return f"Q{key_match.group(1).upper()}"

        label_match = re.search(r"\bq(\d+[a-z0-9]*)\b", lbl)
        if label_match:
            return f"Q{label_match.group(1).upper()}"
        return "Unknown"

    def _infer_env(raw_key: str, label: str) -> str:
        key = raw_key.lower()
        lbl = label.lower()
        if "_wan" in key or "(wan)" in lbl or " wide area" in lbl or "wide-area" in lbl:
            return "WAN"
        if "edge_heavy" in key or "edge-heavy" in lbl:
            return "Edge-heavy"
        if "_e2c" in key or "edge_to_cloud" in key or "edge-to-cloud" in lbl or "het_het" in key:
            return "Balanced"
        if "_cloud" in key or "(cloud)" in lbl or "hom_hom" in key:
            return "Cloud"
        if "on_prem" in key or "on-prem" in lbl:
            return "On-prem"
        if "hom_het" in key or "(mixed)" in lbl:
            return "Mixed"
        return "Unknown"

    parsed_rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        value = pd.to_numeric(getattr(row, value_col), errors="coerce")
        if pd.isna(value):
            continue

        exp_name = _safe_text(getattr(row, "exp_name"))
        exp_key = _safe_text(getattr(row, "exp_key")) if hasattr(row, "exp_key") else ""
        raw_key = exp_key or exp_name

        system = _infer_system(raw_key, exp_name)
        if system is None:
            continue

        parsed_rows.append({
            "env": _infer_env(raw_key, exp_name),
            "query": _infer_query(raw_key, exp_name),
            "system": system,
            "value": float(value),
        })

    if not parsed_rows:
        raise ValueError("No valid throughput values found to build table data.")

    parsed_df = pd.DataFrame(parsed_rows)

    metric_order = ["mean", "min", "max", "std", "rel_std"]
    stats_by_group: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for (env, query, system), group in parsed_df.groupby(["env", "query", "system"]):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rel_std_value = (std_value / mean_value) * 100.0 if mean_value != 0 else np.nan
        stats_by_group[(str(env), str(query), str(system))] = {
            "mean": round(mean_value, 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
            "std": round(std_value, 2),
            "rel_std": round(rel_std_value, 2) if not pd.isna(rel_std_value) else np.nan,
        }

    env_order = {
        "On-prem": 0,
        "Cloud": 1,
        "Mixed": 2,
        "Balanced": 3,
        "Edge-heavy": 4,
        "WAN": 5,
        "Unknown": 99,
    }

    def _query_sort_key(query: str) -> tuple[int, str]:
        match = re.fullmatch(r"Q(\d+)([A-Z0-9]*)", query.upper())
        if match:
            return (int(match.group(1)), match.group(2))
        return (10_000, query)

    def _format_value(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):,.2f}"

    def _format_percent(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):.2f}%"

    query_order = sorted({query for _, query, _ in stats_by_group.keys()}, key=_query_sort_key)
    envs = sorted({env for env, _, _ in stats_by_group.keys()}, key=lambda env: (env_order.get(env, 99), env))
    system_order = [("flink", "Flink")]
    if split_flinke2c_variants:
        system_order.extend([
            ("flinke2c_td", "FlinkE2C (TD)"),
            ("flinke2c_bu", "FlinkE2C (BU)"),
            ("flinke2c", "FlinkE2C"),
        ])
    else:
        system_order.append(("flinke2c", "FlinkE2C"))
    system_order.append(("nes", "NES"))

    columns: list[tuple[str, str]] = [("Query", ""), ("System", "")]
    columns.extend((env, metric) for env in envs for metric in metric_order)

    export_rows: list[dict[tuple[str, str], object]] = []
    for query in query_order:
        first_system_row = True
        for system_key, system_label in system_order:
            has_any_env = any((env, query, system_key) in stats_by_group for env in envs)
            if not has_any_env:
                continue

            row: dict[tuple[str, str], object] = {
                ("Query", ""): query if first_system_row else "",
                ("System", ""): system_label,
            }

            for env in envs:
                stats = stats_by_group.get((env, query, system_key))
                for metric in metric_order:
                    if not stats:
                        row[(env, metric)] = ""
                    elif metric == "rel_std":
                        row[(env, metric)] = _format_percent(stats[metric])
                    else:
                        row[(env, metric)] = _format_value(stats[metric])

            export_rows.append(row)
            first_system_row = False

    if not export_rows:
        raise ValueError("No throughput summary rows available after normalization.")

    export = pd.DataFrame(export_rows, columns=columns)
    export.columns = pd.MultiIndex.from_tuples(columns)

    output_path = f"{output_dir}/{file_name}_tput_table.csv"
    export.to_csv(output_path, index=False)
    return output_path


def gen_tput_table_latex(input_csv: str, output_tex: str, ignore_nes: bool = False) -> str:
    """Generate LaTeX tabular code from throughput table CSV."""
    with open(input_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise RuntimeError("CSV must contain at least 3 rows (2 header rows + data).")

    header_q = rows[0]
    header_m = rows[1]

    def _parse_float(value: str) -> float:
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        if not cleaned:
            return float("nan")
        return float(cleaned)

    def _tex_num(value: str) -> str:
        raw = value.strip().replace(",", "")
        if raw.endswith("%"):
            raw = raw[:-1]
        if not raw:
            return ""
        return f"\\num{{{raw}}}"

    def _tex_pm_std(std_value: str, rel_std: str, *, bold_std: bool = False, bold_rel: bool = False) -> str:
        std_tex = _tex_num(std_value)
        rel_clean = rel_std.strip().rstrip("%")
        rel_tex = _tex_num(rel_clean)
        if not std_tex:
            return ""
        std_content = _maybe_bold(std_tex, bold_std)
        if not rel_tex:
            return f"$\\pm$ {std_content}"
        rel_content = _maybe_bold(f"({rel_tex}\\%)", bold_rel)
        return f"$\\pm$ {std_content}~{rel_content}"

    def _maybe_bold(content: str, is_bold: bool) -> str:
        if not content:
            return content
        return "{\\bfseries " + content + "}" if is_bold else content

    def _maybe_dim(content: str, is_dimmed: bool) -> str:
        if not content:
            return content
        return "{\\color[gray]{0.20} " + content + "}" if is_dimmed else content

    def _is_nes_system(system: str) -> bool:
        normalized = system.strip().lower().replace("-", "").replace(" ", "")
        return normalized in {"nes", "nebulastream"}

    def _should_ignore_nes(env: str, system: str) -> bool:
        return ignore_nes and _is_nes_system(system)

    env_cols: dict[str, dict[str, int]] = {}
    for idx in range(2, len(header_q)):
        env = header_q[idx].strip()
        metric = header_m[idx].strip()
        if not env or not metric:
            continue
        env_cols.setdefault(env, {})[metric] = idx

    envs = list(env_cols.keys())
    if not envs:
        raise RuntimeError("No environment columns found in CSV header.")

    base_metrics = ["mean", "std", "min", "max"]
    required_metrics = [*base_metrics, "rel_std"]
    for env in envs:
        for metric in required_metrics:
            if metric not in env_cols[env]:
                raise RuntimeError(f"Missing metric '{metric}' for environment '{env}'.")

    query_rows: list[tuple[str, list[tuple[str, dict[str, dict[str, str]]]]]] = []
    current_query = ""
    current_system_rows: list[tuple[str, dict[str, dict[str, str]]]] = []

    for row in rows[2:]:
        if len(row) < len(header_q):
            row = row + [""] * (len(header_q) - len(row))

        query = row[0].strip()
        system = row[1].strip()
        if not system:
            continue

        if query:
            if current_query:
                query_rows.append((current_query, current_system_rows))
            current_query = query
            current_system_rows = []

        if not current_query:
            continue

        metrics_by_env: dict[str, dict[str, str]] = {}
        for env in envs:
            metrics_by_env[env] = {}
            for metric in required_metrics:
                metrics_by_env[env][metric] = row[env_cols[env][metric]].strip()

        current_system_rows.append((system, metrics_by_env))

    if current_query:
        query_rows.append((current_query, current_system_rows))

    if not query_rows:
        raise RuntimeError("No data rows found in CSV.")

    col_count = 2 + len(envs) * len(base_metrics)
    env_align: list[str] = []
    for _ in envs:
        env_align.extend(["c", "l", "c", "c"])  # std column left, others centered
    align = "@{} c c " + " ".join(env_align) + " @{}"

    top_header_parts = [r"\multirow{2}{*}{\textbf{Query}}", r"\multirow{2}{*}{\textbf{System}}"]
    for env in envs:
        top_header_parts.append(rf"\multicolumn{{{len(base_metrics)}}}{{c}}{{\textbf{{{env}}}}}")

    cmid_parts = []
    start = 3
    for _ in envs:
        end = start + len(base_metrics) - 1
        cmid_parts.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1

    sub_header_parts = ["", ""]
    for _ in envs:
        sub_header_parts.extend([
            r"\textbf{mean}",
            r"\multicolumn{1}{l}{\textbf{$\pm$ std (RSD)}}",
            r"\textbf{min}",
            r"\textbf{max}",
        ])

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append(r"  \toprule")
    lines.append("  " + " & ".join(top_header_parts) + r" \\")
    lines.append("  " + " ".join(cmid_parts))
    lines.append("  " + " & ".join(sub_header_parts) + r" \\")
    lines.append(r"  \midrule")

    for query_idx, (query, systems) in enumerate(query_rows):
        winner: dict[tuple[str, str], set[int]] = {}
        excluded_winner_rows = {
            idx for idx, (system, _) in enumerate(systems)
            if _should_ignore_nes(query, system)
        }
        if systems:
            for env in envs:
                for metric in required_metrics:
                    values = [
                        _parse_float(system_data[env][metric])
                        for _, system_data in systems
                    ]
                    valid = [
                        (idx, value)
                        for idx, value in enumerate(values)
                        if not np.isnan(value) and idx not in excluded_winner_rows
                    ]
                    if not valid:
                        winner[(env, metric)] = set()
                        continue

                    target = (
                        min(value for _, value in valid)
                        if metric in {"std", "rel_std"}
                        else max(value for _, value in valid)
                    )
                    winner[(env, metric)] = {
                        idx for idx, value in valid if value == target
                    }

        for sys_idx, (system, data) in enumerate(systems):
            query_cell = rf"\multirow{{{len(systems)}}}{{*}}{{{query}}}" if sys_idx == 0 else ""
            ignore_row = _should_ignore_nes(query, system)
            row_parts = [query_cell, _maybe_dim(system, ignore_row)]

            for env in envs:
                for metric in base_metrics:
                    if metric == "std":
                        content = _tex_pm_std(
                            data[env]["std"],
                            data[env]["rel_std"],
                            bold_std=(not ignore_row) and sys_idx in winner.get((env, "std"), set()),
                            bold_rel=(not ignore_row) and sys_idx in winner.get((env, "rel_std"), set()),
                        )
                        row_parts.append(_maybe_dim(content, ignore_row))
                    else:
                        content = _tex_num(data[env][metric])
                        is_bold = (not ignore_row) and sys_idx in winner.get((env, metric), set())
                        row_parts.append(_maybe_dim(_maybe_bold(content, is_bold), ignore_row))

            add_gap_before_next = (
                sys_idx + 1 < len(systems)
                and _should_ignore_nes(query, systems[sys_idx + 1][0])
            )
            row_suffix = r" \\[0.15em]" if add_gap_before_next else r" \\"
            lines.append("  " + " & ".join(row_parts) + row_suffix)

        if query_idx < len(query_rows) - 1:
            lines.append(rf"  \cmidrule(lr){{1-{col_count}}}")

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    output_dir = os.path.dirname(output_tex)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output_tex


def gen_latency_table_data(
        df: pd.DataFrame,
        file_name: str,
        output_suffix: str = "",
        value_col: str = "value",
        split_flinke2c_variants: bool = False,
        delta_metric: str = "p99",
        aggregation: str = "pooled",
) -> str:
    """Generate latency summary table CSV grouped by query/system.

    This is intended for raw per-event latency samples where lower is better.
    Input should typically come from
    ``prepare_sink_latency_distribution_data(..., per_repetition=True)``.

    By default, metrics are computed in ``pooled`` mode, i.e. from the full
    pooled latency distribution per system. That matches the semantics used by
    ECDF plots and ``groupby(...).quantile(...)`` on the flattened data.
    ``mean_of_repetitions`` is also available when you explicitly want to first
    compute per-repetition summaries and then average those summaries.

    Output columns are multi-level:
      - (Query, ""), (System, "")
      - (Environment, metric) where metric is one of: p50, p99, delta_best
    """
    summary_mode = {"exp_name", "p50", "p99"}.issubset(df.columns)
    if not summary_mode:
        required = {"exp_name", value_col, "repetition"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for latency table CSV: {', '.join(sorted(missing))}")

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_dir = f"plot_data{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    def _safe_text(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value)

    def _infer_system(raw_key: str, label: str) -> str | None:
        key = raw_key.lower()
        lbl = label.lower()
        if "flinke2c_bu" in key or "bottom-up" in lbl:
            return "flinke2c_bu" if split_flinke2c_variants else "flinke2c"
        if "flinke2c_td" in key or "top-down" in lbl:
            return "flinke2c_td" if split_flinke2c_variants else "flinke2c"
        if "flinke2c" in key or "flinke2c" in lbl:
            return "flinke2c"
        if "capsys" in key or "capsys" in lbl:
            return "capsys"
        if key.startswith("nes") or re.search(r"\bnes\b", lbl):
            return "nes"
        if key.startswith("flink") or key.startswith("base_") or "baseline" in lbl:
            return "flink"
        return None

    def _infer_query(raw_key: str, label: str) -> str:
        key = raw_key.lower()
        lbl = label.lower()
        key_match = re.search(r"(?:^|_)q(\d+[a-z0-9]*)", key)
        if key_match:
            return f"Q{key_match.group(1).upper()}"

        label_match = re.search(r"\bq(\d+[a-z0-9]*)\b", lbl)
        if label_match:
            return f"Q{label_match.group(1).upper()}"
        return "Unknown"

    def _infer_env(raw_key: str, label: str) -> str:
        key = raw_key.lower()
        lbl = label.lower()
        if "_wan" in key or "(wan)" in lbl or " wide area" in lbl or "wide-area" in lbl:
            return "WAN"
        if "edge_heavy" in key or "edge-heavy" in lbl:
            return "Edge-heavy"
        if "_e2c" in key or "edge_to_cloud" in key or "edge-to-cloud" in lbl or "het_het" in key:
            return "Balanced"
        if "_cloud" in key or "(cloud)" in lbl or "hom_hom" in key:
            return "Cloud"
        if "on_prem" in key or "on-prem" in lbl:
            return "On-prem"
        if "hom_het" in key or "(mixed)" in lbl:
            return "Mixed"
        return "Unknown"

    def _query_sort_key(query: str) -> tuple[int, str]:
        match = re.fullmatch(r"Q(\d+)([A-Z0-9]*)", query.upper())
        if match:
            return (int(match.group(1)), match.group(2))
        return (10_000, query)

    def _format_value(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):,.2f}"

    def _format_percent(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):+.2f}%"

    summary_by_group: dict[tuple[str, str, str], dict[str, float]] = {}
    if summary_mode:
        for row in df.itertuples(index=False):
            exp_name = _safe_text(getattr(row, "exp_name"))
            exp_key = _safe_text(getattr(row, "exp_key")) if hasattr(row, "exp_key") else ""
            raw_key = exp_key or exp_name
            system = _infer_system(raw_key, exp_name)
            if system is None:
                continue

            p50_value = pd.to_numeric(pd.Series([getattr(row, "p50")]), errors="coerce").iloc[0]
            p99_value = pd.to_numeric(pd.Series([getattr(row, "p99")]), errors="coerce").iloc[0]
            if pd.isna(p50_value) or pd.isna(p99_value):
                continue

            env = _infer_env(raw_key, exp_name)
            query = _infer_query(raw_key, exp_name)
            summary_by_group[(str(env), str(query), str(system))] = {
                "p50": round(float(p50_value), 2),
                "p99": round(float(p99_value), 2),
            }
    else:
        parsed_rows: list[dict[str, object]] = []
        for row in df.itertuples(index=False):
            value = pd.to_numeric(getattr(row, value_col), errors="coerce")
            if pd.isna(value):
                continue

            exp_name = _safe_text(getattr(row, "exp_name"))
            exp_key = _safe_text(getattr(row, "exp_key")) if hasattr(row, "exp_key") else ""
            raw_key = exp_key or exp_name

            system = _infer_system(raw_key, exp_name)
            if system is None:
                continue

            parsed_rows.append({
                "env": _infer_env(raw_key, exp_name),
                "query": _infer_query(raw_key, exp_name),
                "system": system,
                "repetition": _safe_text(getattr(row, "repetition")),
                "value": float(value),
            })

        if not parsed_rows:
            raise ValueError("No valid latency values found to build table data.")

        parsed_df = pd.DataFrame(parsed_rows)
        aggregation_key = aggregation.strip().lower()
        if aggregation_key == "pooled":
            for (env, query, system), group in parsed_df.groupby(["env", "query", "system"]):
                values = pd.to_numeric(group["value"], errors="coerce").dropna()
                if values.empty:
                    continue
                summary_by_group[(str(env), str(query), str(system))] = {
                    "p50": round(float(np.quantile(values.to_numpy(), 0.5, method="linear")), 2),
                    "p99": round(float(np.quantile(values.to_numpy(), 0.99, method="linear")), 2),
                }
        elif aggregation_key == "mean_of_repetitions":
            rep_metrics: list[dict[str, object]] = []
            for (env, query, system, repetition), group in parsed_df.groupby(["env", "query", "system", "repetition"]):
                values = pd.to_numeric(group["value"], errors="coerce").dropna()
                if values.empty:
                    continue
                rep_metrics.append({
                    "env": str(env),
                    "query": str(query),
                    "system": str(system),
                    "repetition": str(repetition),
                    "p50": float(np.quantile(values.to_numpy(), 0.5, method="linear")),
                    "p99": float(np.quantile(values.to_numpy(), 0.99, method="linear")),
                })

            if not rep_metrics:
                raise ValueError("No latency summary rows available after normalization.")

            rep_metrics_df = pd.DataFrame(rep_metrics)
            for (env, query, system), group in rep_metrics_df.groupby(["env", "query", "system"]):
                p50_latency = pd.to_numeric(group["p50"], errors="coerce").dropna()
                p99_latency = pd.to_numeric(group["p99"], errors="coerce").dropna()
                if p50_latency.empty or p99_latency.empty:
                    continue
                summary_by_group[(str(env), str(query), str(system))] = {
                    "p50": round(float(p50_latency.mean()), 2),
                    "p99": round(float(p99_latency.mean()), 2),
                }
        else:
            raise ValueError("aggregation must be either 'pooled' or 'mean_of_repetitions'.")

    if not summary_by_group:
        raise ValueError("No latency summary rows available after aggregation.")

    delta_metric_key = delta_metric.strip().lower()
    if delta_metric_key not in {"p50", "p99"}:
        raise ValueError("delta_metric must be either 'p50' or 'p99'.")

    best_by_env_query: dict[tuple[str, str], float] = {}
    for env, query, _ in summary_by_group:
        best_value = min(
            stats[delta_metric_key]
            for (cur_env, cur_query, _), stats in summary_by_group.items()
            if cur_env == env and cur_query == query
        )
        best_by_env_query[(env, query)] = best_value

    env_order = {
        "On-prem": 0,
        "Cloud": 1,
        "Mixed": 2,
        "Balanced": 3,
        "Edge-heavy": 4,
        "WAN": 5,
        "Unknown": 99,
    }

    query_order = sorted({query for _, query, _ in summary_by_group.keys()}, key=_query_sort_key)
    envs = sorted({env for env, _, _ in summary_by_group.keys()}, key=lambda env: (env_order.get(env, 99), env))
    system_order = [("flink", "Flink")]
    if split_flinke2c_variants:
        system_order.extend([
            ("flinke2c_td", "FlinkE2C (TD)"),
            ("flinke2c_bu", "FlinkE2C (BU)"),
            ("flinke2c", "FlinkE2C"),
        ])
    else:
        system_order.append(("flinke2c", "FlinkE2C"))
    system_order.append(("nes", "NES"))

    metric_order = ["p50", "p99", "delta_best"]
    columns: list[tuple[str, str]] = [("Query", ""), ("System", "")]
    for env in envs:
        for metric in metric_order:
            columns.append((env, metric))

    export_rows: list[dict[tuple[str, str], object]] = []
    for query in query_order:
        first_system_row = True
        for system_key, system_label in system_order:
            has_any_env = any(
                (env, query, system_key) in summary_by_group
                for env in envs
            )
            if not has_any_env:
                continue

            row: dict[tuple[str, str], object] = {
                ("Query", ""): query if first_system_row else "",
                ("System", ""): system_label,
            }

            for env in envs:
                stats = summary_by_group.get((env, query, system_key))
                best_value = best_by_env_query.get((env, query))
                if not stats:
                    row[(env, "p50")] = ""
                    row[(env, "p99")] = ""
                    row[(env, "delta_best")] = ""
                    continue

                delta_best = np.nan
                if best_value not in (None, 0):
                    delta_best = ((stats[delta_metric_key] - best_value) / best_value) * 100.0

                row[(env, "p50")] = _format_value(stats["p50"])
                row[(env, "p99")] = _format_value(stats["p99"])
                row[(env, "delta_best")] = _format_percent(delta_best)

            export_rows.append(row)
            first_system_row = False

    if not export_rows:
        raise ValueError("No latency export rows available after normalization.")

    export = pd.DataFrame(export_rows, columns=columns)
    export.columns = pd.MultiIndex.from_tuples(columns)

    output_path = f"{output_dir}/{file_name}_lat_table.csv"
    export.to_csv(output_path, index=False)
    return output_path


def gen_latency_table_latex(input_csv: str, output_tex: str, ignore_nes: bool = False) -> str:
    """Generate LaTeX tabular code from latency table CSV."""
    with open(input_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise RuntimeError("CSV must contain at least 3 rows (2 header rows + data).")

    header_q = rows[0]
    header_m = rows[1]

    def _parse_float(value: str) -> float:
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        if cleaned.startswith("+"):
            cleaned = cleaned[1:]
        if not cleaned:
            return float("nan")
        return float(cleaned)

    def _tex_num(value: str) -> str:
        raw = value.strip().replace(",", "")
        if raw.endswith("%"):
            raw = raw[:-1]
        if raw.startswith("+"):
            raw = raw[1:]
        if not raw:
            return ""
        return f"\\num{{{raw}}}"

    def _maybe_bold(content: str, is_bold: bool) -> str:
        if not content:
            return content
        return "{\\bfseries " + content + "}" if is_bold else content

    def _maybe_dim(content: str, is_dimmed: bool) -> str:
        if not content:
            return content
        return "{\\color[gray]{0.20} " + content + "}" if is_dimmed else content

    def _is_nes_system(system: str) -> bool:
        normalized = system.strip().lower().replace("-", "").replace(" ", "")
        return normalized in {"nes", "nebulastream"}

    def _should_ignore_nes(system: str) -> bool:
        return ignore_nes and _is_nes_system(system)

    env_cols: dict[str, dict[str, int]] = {}
    for idx in range(2, len(header_q)):
        env = header_q[idx].strip()
        metric_key = header_m[idx].strip()
        normalized_metric = metric_key.lower()
        if not env or normalized_metric not in {"p50", "p99", "delta_best"}:
            continue
        env_cols.setdefault(env, {})[normalized_metric] = idx

    envs = list(env_cols.keys())
    if not envs:
        raise RuntimeError("No environment columns found in CSV header.")

    required_metrics = ["p50", "p99", "delta_best"]
    for env in envs:
        for metric in required_metrics:
            if metric not in env_cols.get(env, {}):
                raise RuntimeError(f"Missing metric '{metric}' for environment '{env}'.")

    query_rows: list[tuple[str, list[tuple[str, dict[str, dict[str, str]]]]]] = []
    current_query = ""
    current_system_rows: list[tuple[str, dict[str, dict[str, str]]]] = []

    for row in rows[2:]:
        if len(row) < len(header_q):
            row = row + [""] * (len(header_q) - len(row))

        query = row[0].strip()
        system = row[1].strip()
        if not system:
            continue

        if query:
            if current_query:
                query_rows.append((current_query, current_system_rows))
            current_query = query
            current_system_rows = []

        if not current_query:
            continue

        metrics_by_env: dict[str, dict[str, str]] = {}
        for env in envs:
            metrics_by_env[env] = {}
            for metric in required_metrics:
                metrics_by_env[env][metric] = row[env_cols[env][metric]].strip()

        current_system_rows.append((system, metrics_by_env))

    if current_query:
        query_rows.append((current_query, current_system_rows))

    if not query_rows:
        raise RuntimeError("No data rows found in CSV.")

    col_count = 2 + len(envs) * len(required_metrics)
    align = "@{} c c " + " ".join(["c c c"] * len(envs)) + " @{}"

    top_header_parts = [r"\multirow{2}{*}{\textbf{Query}}", r"\multirow{2}{*}{\textbf{System}}"]
    for env in envs:
        top_header_parts.append(rf"\multicolumn{{{len(required_metrics)}}}{{c}}{{\textbf{{{env}}}}}")

    env_cmid_parts = []
    start = 3
    for _ in envs:
        end = start + len(required_metrics) - 1
        env_cmid_parts.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1

    sub_header_parts = ["", ""]
    for _ in envs:
        sub_header_parts.extend([
            r"\textbf{median}",
            r"\textbf{p99}",
            r"\textbf{$\Delta$ p99 to best}",
        ])

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append(r"  \toprule")
    lines.append("  " + " & ".join(top_header_parts) + r" \\")
    lines.append("  " + " ".join(env_cmid_parts))
    lines.append("  " + " & ".join(sub_header_parts) + r" \\")
    lines.append(r"  \midrule")

    for query_idx, (query, systems) in enumerate(query_rows):
        winner: dict[tuple[str, str], set[int]] = {}
        excluded_winner_rows = {
            idx for idx, (system, _) in enumerate(systems)
            if _should_ignore_nes(system)
        }
        if systems:
            for env in envs:
                for metric in required_metrics:
                    values = [
                        _parse_float(system_data[env][metric])
                        for _, system_data in systems
                    ]
                    valid = [
                        (idx, value)
                        for idx, value in enumerate(values)
                        if not np.isnan(value) and idx not in excluded_winner_rows
                    ]
                    if not valid:
                        winner[(env, metric)] = set()
                        continue

                    target = min(value for _, value in valid)
                    winner[(env, metric)] = {
                        idx for idx, value in valid if value == target
                    }

        for sys_idx, (system, data) in enumerate(systems):
            query_cell = rf"\multirow{{{len(systems)}}}{{*}}{{{query}}}" if sys_idx == 0 else ""
            ignore_row = _should_ignore_nes(system)
            row_parts = [query_cell, _maybe_dim(system, ignore_row)]

            for env in envs:
                p50_tex = _tex_num(data[env]["p50"])
                p50_is_bold = (not ignore_row) and sys_idx in winner.get((env, "p50"), set())
                row_parts.append(_maybe_dim(_maybe_bold(p50_tex, p50_is_bold), ignore_row))

                p99_tex = _tex_num(data[env]["p99"])
                p99_is_bold = (not ignore_row) and sys_idx in winner.get((env, "p99"), set())
                row_parts.append(_maybe_dim(_maybe_bold(p99_tex, p99_is_bold), ignore_row))

                delta_raw = data[env]["delta_best"].strip()
                delta_sign = "+"
                if delta_raw.startswith("-"):
                    delta_sign = "-"
                delta_tex = _tex_num(delta_raw)
                if delta_tex:
                    delta_content = f"{delta_sign}{delta_tex}\\%"
                else:
                    delta_content = ""
                delta_is_bold = (not ignore_row) and sys_idx in winner.get((env, "delta_best"), set())
                row_parts.append(_maybe_dim(_maybe_bold(delta_content, delta_is_bold), ignore_row))

            add_gap_before_next = (
                sys_idx + 1 < len(systems)
                and _should_ignore_nes(systems[sys_idx + 1][0])
            )
            row_suffix = r" \\[0.15em]" if add_gap_before_next else r" \\"
            lines.append("  " + " & ".join(row_parts) + row_suffix)

        if query_idx < len(query_rows) - 1:
            lines.append(rf"  \cmidrule(lr){{1-{col_count}}}")

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    output_dir = os.path.dirname(output_tex)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output_tex


def gen_tput_latency_table_data(
        tput_df: pd.DataFrame,
        lat_df: pd.DataFrame,
        file_name: str,
        output_suffix: str = "",
        tput_value_col: str = "value",
        split_flinke2c_variants: bool = False,
) -> str:
    """Generate a combined throughput/latency summary table CSV.

    Throughput columns:
      - mean, std, rel_std, min, max

    Latency columns:
      - p50, p99

    The latency input can either be:
      - a summary DataFrame with columns ``exp_name``, ``p50``, ``p99``
      - raw per-event data with columns ``exp_name``, ``repetition``, ``value``
        in which case pooled p50/p99 are computed.
    """
    tput_required = {"exp_name", tput_value_col}
    tput_missing = tput_required - set(tput_df.columns)
    if tput_missing:
        raise ValueError(f"Missing columns for throughput table CSV: {', '.join(sorted(tput_missing))}")

    lat_summary_mode = {"exp_name", "p50", "p99"}.issubset(lat_df.columns)
    if not lat_summary_mode:
        lat_required = {"exp_name", "repetition", "value"}
        lat_missing = lat_required - set(lat_df.columns)
        if lat_missing:
            raise ValueError(f"Missing columns for latency table CSV: {', '.join(sorted(lat_missing))}")

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_dir = f"plot_data{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    def _safe_text(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value)

    def _infer_system(raw_key: str, label: str) -> str | None:
        key = raw_key.lower()
        lbl = label.lower()
        if "flinke2c_bu" in key or "bottom-up" in lbl:
            return "flinke2c_bu" if split_flinke2c_variants else "flinke2c"
        if "flinke2c_td" in key or "top-down" in lbl:
            return "flinke2c_td" if split_flinke2c_variants else "flinke2c"
        if "flinke2c" in key or "flinke2c" in lbl:
            return "flinke2c"
        if "capsys" in key or "capsys" in lbl:
            return "capsys"
        if "flink_det" in key or "flink (DET)" in lbl:
            return "flink_det"
        if key.startswith("nes") or re.search(r"\bnes\b", lbl):
            return "nes"
        if key.startswith("flink") or key.startswith("base_") or "baseline" in lbl:
            return "flink"
        return None

    def _infer_query(raw_key: str, label: str) -> str:
        key = raw_key.lower()
        lbl = label.lower()
        key_match = re.search(r"(?:^|_)q(\d+[a-z0-9]*)", key)
        if key_match:
            return f"Q{key_match.group(1).upper()}"

        label_match = re.search(r"\bq(\d+[a-z0-9]*)\b", lbl)
        if label_match:
            return f"Q{label_match.group(1).upper()}"
        return "Unknown"

    def _infer_env(raw_key: str, label: str) -> str:
        key = raw_key.lower()
        lbl = label.lower()
        if "_wan" in key or "(wan)" in lbl or " wide area" in lbl or "wide-area" in lbl:
            return "WAN"
        if "edge_heavy" in key or "_eh" in key or "edge-heavy" in lbl:
            return "Edge-heavy"
        if "_e2c" in key or "edge_to_cloud" in key or "edge-to-cloud" in lbl or "het_het" in key:
            return "Balanced"
        if "_cloud" in key or "(cloud)" in lbl or "hom_hom" in key:
            return "Cloud"
        if "on_prem" in key or "on-prem" in lbl:
            return "On-prem"
        if "hom_het" in key or "(mixed)" in lbl:
            return "Mixed"
        return "Unknown"

    def _query_sort_key(query: str) -> tuple[int, str]:
        match = re.fullmatch(r"Q(\d+)([A-Z0-9]*)", query.upper())
        if match:
            return (int(match.group(1)), match.group(2))
        return (10_000, query)

    def _format_value(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):,.2f}"

    def _format_percent(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return ""
        return f"{float(value):.2f}%"

    tput_stats: dict[tuple[str, str, str], dict[str, float]] = {}
    parsed_tput_rows: list[dict[str, object]] = []
    for row in tput_df.itertuples(index=False):
        value = pd.to_numeric(getattr(row, tput_value_col), errors="coerce")
        if pd.isna(value):
            continue

        exp_name = _safe_text(getattr(row, "exp_name"))
        exp_key = _safe_text(getattr(row, "exp_key")) if hasattr(row, "exp_key") else ""
        raw_key = exp_key or exp_name
        system = _infer_system(raw_key, exp_name)
        if system is None:
            continue

        parsed_tput_rows.append({
            "env": _infer_env(raw_key, exp_name),
            "query": _infer_query(raw_key, exp_name),
            "system": system,
            "value": float(value),
        })

    if not parsed_tput_rows:
        raise ValueError("No valid throughput values found to build combined table data.")

    parsed_tput_df = pd.DataFrame(parsed_tput_rows)
    for (env, query, system), group in parsed_tput_df.groupby(["env", "query", "system"]):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rel_std_value = (std_value / mean_value) * 100.0 if mean_value != 0 else np.nan
        tput_stats[(str(env), str(query), str(system))] = {
            "mean": round(mean_value, 2),
            "std": round(std_value, 2),
            "rel_std": round(rel_std_value, 2) if not pd.isna(rel_std_value) else np.nan,
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
        }

    lat_stats: dict[tuple[str, str, str], dict[str, float]] = {}
    if lat_summary_mode:
        for row in lat_df.itertuples(index=False):
            exp_name = _safe_text(getattr(row, "exp_name"))
            exp_key = _safe_text(getattr(row, "exp_key")) if hasattr(row, "exp_key") else ""
            raw_key = exp_key or exp_name
            system = _infer_system(raw_key, exp_name)
            if system is None:
                continue

            p50_value = pd.to_numeric(pd.Series([getattr(row, "p50")]), errors="coerce").iloc[0]
            p99_value = pd.to_numeric(pd.Series([getattr(row, "p99")]), errors="coerce").iloc[0]
            if pd.isna(p50_value) or pd.isna(p99_value):
                continue

            lat_stats[(_infer_env(raw_key, exp_name), _infer_query(raw_key, exp_name), system)] = {
                "p50": round(float(p50_value), 2),
                "p99": round(float(p99_value), 2),
            }
    else:
        parsed_lat_rows: list[dict[str, object]] = []
        for row in lat_df.itertuples(index=False):
            value = pd.to_numeric(getattr(row, "value"), errors="coerce")
            if pd.isna(value):
                continue

            exp_name = _safe_text(getattr(row, "exp_name"))
            exp_key = _safe_text(getattr(row, "exp_key")) if hasattr(row, "exp_key") else ""
            raw_key = exp_key or exp_name
            system = _infer_system(raw_key, exp_name)
            if system is None:
                continue

            parsed_lat_rows.append({
                "env": _infer_env(raw_key, exp_name),
                "query": _infer_query(raw_key, exp_name),
                "system": system,
                "value": float(value),
            })

        if parsed_lat_rows:
            parsed_lat_df = pd.DataFrame(parsed_lat_rows)
            for (env, query, system), group in parsed_lat_df.groupby(["env", "query", "system"]):
                values = pd.to_numeric(group["value"], errors="coerce").dropna()
                if values.empty:
                    continue
                lat_stats[(str(env), str(query), str(system))] = {
                    "p50": round(float(np.quantile(values.to_numpy(), 0.5, method="linear")), 2),
                    "p99": round(float(np.quantile(values.to_numpy(), 0.99, method="linear")), 2),
                }

    if not lat_stats:
        raise ValueError("No valid latency values found to build combined table data.")

    env_order = {
        "On-prem": 0,
        "Cloud": 1,
        "Mixed": 2,
        "Balanced": 3,
        "Edge-heavy": 4,
        "WAN": 5,
        "Unknown": 99,
    }

    query_order = sorted(
        {query for _, query, _ in set(tput_stats.keys()) | set(lat_stats.keys())},
        key=_query_sort_key,
    )
    envs = sorted(
        {env for env, _, _ in set(tput_stats.keys()) | set(lat_stats.keys())},
        key=lambda env: (env_order.get(env, 99), env),
    )
    system_order = [("flink", "Flink"), ("flink_det", "Flink (DET)")]
    if split_flinke2c_variants:
        system_order.extend([
            ("flinke2c_td", "FlinkE2C (TD)"),
            ("flinke2c_bu", "FlinkE2C (BU)"),
            ("flinke2c", "FlinkE2C"),
        ])
    else:
        system_order.append(("flinke2c", "FlinkE2C"))
    system_order.append(("capsys", "CapSys"))
    system_order.append(("nes", "NES"))

    metric_order = ["mean", "std", "rel_std", "min", "max", "p50", "p99"]
    columns: list[tuple[str, str]] = [("Query", ""), ("System", "")]
    columns.extend((env, metric) for env in envs for metric in metric_order)

    export_rows: list[dict[tuple[str, str], object]] = []
    for query in query_order:
        first_system_row = True
        for system_key, system_label in system_order:
            has_any_env = any(
                (env, query, system_key) in tput_stats or (env, query, system_key) in lat_stats
                for env in envs
            )
            if not has_any_env:
                continue

            row: dict[tuple[str, str], object] = {
                ("Query", ""): query if first_system_row else "",
                ("System", ""): system_label,
            }

            for env in envs:
                t_stats = tput_stats.get((env, query, system_key))
                l_stats = lat_stats.get((env, query, system_key))

                row[(env, "mean")] = _format_value(t_stats["mean"]) if t_stats else ""
                row[(env, "std")] = _format_value(t_stats["std"]) if t_stats else ""
                row[(env, "rel_std")] = _format_percent(t_stats["rel_std"]) if t_stats else ""
                row[(env, "min")] = _format_value(t_stats["min"]) if t_stats else ""
                row[(env, "max")] = _format_value(t_stats["max"]) if t_stats else ""
                row[(env, "p50")] = _format_value(l_stats["p50"]) if l_stats else ""
                row[(env, "p99")] = _format_value(l_stats["p99"]) if l_stats else ""

            export_rows.append(row)
            first_system_row = False

    if not export_rows:
        raise ValueError("No combined export rows available after normalization.")

    export = pd.DataFrame(export_rows, columns=columns)
    export.columns = pd.MultiIndex.from_tuples(columns)

    output_path = f"{output_dir}/{file_name}_perf_table.csv"
    export.to_csv(output_path, index=False)
    return output_path


def gen_tput_latency_table_latex(input_csv: str, output_tex: str, ignore_nes: bool = False) -> str:
    """Generate LaTeX tabular code from combined throughput/latency table CSV."""
    with open(input_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise RuntimeError("CSV must contain at least 3 rows (2 header rows + data).")

    header_q = rows[0]
    header_m = rows[1]

    def _parse_float(value: str) -> float:
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        if not cleaned:
            return float("nan")
        return float(cleaned)

    def _tex_num(value: str) -> str:
        raw = value.strip().replace(",", "")
        if raw.endswith("%"):
            raw = raw[:-1]
        if not raw:
            return ""
        return f"\\num{{{raw}}}"

    def _maybe_bold(content: str, is_bold: bool) -> str:
        if not content:
            return content
        return "{\\bfseries " + content + "}" if is_bold else content

    def _maybe_dim(content: str, is_dimmed: bool) -> str:
        if not content:
            return content
        return "{\\color[gray]{0.20} " + content + "}" if is_dimmed else content

    def _tex_pm_std(std_value: str, rel_std: str, *, bold_std: bool = False, bold_rel: bool = False) -> str:
        std_tex = _tex_num(std_value)
        rel_clean = rel_std.strip().rstrip("%")
        rel_tex = _tex_num(rel_clean)
        if not std_tex:
            return ""
        std_content = _maybe_bold(std_tex, bold_std)
        if not rel_tex:
            return f"$\\pm$ {std_content}"
        rel_content = _maybe_bold(f"({rel_tex}\\%)", bold_rel)
        return f"$\\pm$ {std_content}~{rel_content}"

    def _is_nes_system(system: str) -> bool:
        normalized = system.strip().lower().replace("-", "").replace(" ", "")
        return normalized in {"nes", "nebulastream"}

    def _should_ignore_nes(system: str) -> bool:
        return ignore_nes and _is_nes_system(system)

    env_cols: dict[str, dict[str, int]] = {}
    for idx in range(2, len(header_q)):
        env = header_q[idx].strip()
        metric = header_m[idx].strip().lower()
        if not env or not metric:
            continue
        env_cols.setdefault(env, {})[metric] = idx

    envs = list(env_cols.keys())
    if not envs:
        raise RuntimeError("No environment columns found in CSV header.")

    required_metrics = ["mean", "std", "rel_std", "min", "max", "p50", "p99"]
    for env in envs:
        for metric in required_metrics:
            if metric not in env_cols.get(env, {}):
                raise RuntimeError(f"Missing metric '{metric}' for environment '{env}'.")

    query_rows: list[tuple[str, list[tuple[str, dict[str, dict[str, str]]]]]] = []
    current_query = ""
    current_system_rows: list[tuple[str, dict[str, dict[str, str]]]] = []

    for row in rows[2:]:
        if len(row) < len(header_q):
            row = row + [""] * (len(header_q) - len(row))

        query = row[0].strip()
        system = row[1].strip()
        if not system:
            continue

        if query:
            if current_query:
                query_rows.append((current_query, current_system_rows))
            current_query = query
            current_system_rows = []

        if not current_query:
            continue

        metrics_by_env: dict[str, dict[str, str]] = {}
        for env in envs:
            metrics_by_env[env] = {}
            for metric in required_metrics:
                metrics_by_env[env][metric] = row[env_cols[env][metric]].strip()

        current_system_rows.append((system, metrics_by_env))

    if current_query:
        query_rows.append((current_query, current_system_rows))

    if not query_rows:
        raise RuntimeError("No data rows found in CSV.")

    visible_metrics = ["mean", "std", "min", "max", "p50", "p99"]
    col_count = 2 + len(envs) * len(visible_metrics)
    align_parts = ["@{}", "c", "c"]
    for _ in envs:
        align_parts.extend(["c", "l", "c", "c", r"@{\hspace{0.7em}}", "c", "c"])
    align_parts.append("@{}")
    align = " ".join(align_parts)

    top_header_parts = [r"\multirow{3}{*}{\textbf{Query}}", r"\multirow{3}{*}{\textbf{System}}"]
    for env in envs:
        top_header_parts.append(rf"\multicolumn{{{len(visible_metrics)}}}{{c}}{{\textbf{{{env}}}}}")

    env_cmid_parts = []
    start = 3
    for _ in envs:
        end = start + len(visible_metrics) - 1
        env_cmid_parts.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1

    group_header_parts = ["", ""]
    for _ in envs:
        group_header_parts.extend([
            r"\multicolumn{4}{c}{\textbf{Throughput}}",
            r"\multicolumn{2}{c}{\textbf{Latency}}",
        ])

    metric_cmid_parts = []
    start = 3
    for _ in envs:
        metric_cmid_parts.append(rf"\cmidrule(lr){{{start}-{start + 3}}}")
        metric_cmid_parts.append(rf"\cmidrule(lr){{{start + 4}-{start + 5}}}")
        start += len(visible_metrics)

    sub_header_parts = ["", ""]
    for _ in envs:
        sub_header_parts.extend([
            r"\textbf{mean}",
            r"\multicolumn{1}{l}{\textbf{$\pm$ std (RSD)}}",
            r"\textbf{min}",
            r"\textbf{max}",
            r"\textbf{median}",
            r"\textbf{p99}",
        ])

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append(r"  \toprule")
    lines.append("  " + " & ".join(top_header_parts) + r" \\")
    lines.append("  " + " ".join(env_cmid_parts))
    lines.append("  " + " & ".join(group_header_parts) + r" \\")
    lines.append("  " + " ".join(metric_cmid_parts))
    lines.append("  " + " & ".join(sub_header_parts) + r" \\")
    lines.append(r"  \midrule")

    for query_idx, (query, systems) in enumerate(query_rows):
        tput_winner: dict[tuple[str, str], set[int]] = {}
        lat_winner: dict[tuple[str, str], set[int]] = {}
        excluded_winner_rows = {
            idx for idx, (system, _) in enumerate(systems)
            if _should_ignore_nes(system)
        }

        for env in envs:
            for metric in ["mean", "std", "rel_std", "min", "max"]:
                values = [_parse_float(system_data[env][metric]) for _, system_data in systems]
                valid = [
                    (idx, value)
                    for idx, value in enumerate(values)
                    if not np.isnan(value) and idx not in excluded_winner_rows
                ]
                if not valid:
                    tput_winner[(env, metric)] = set()
                    continue
                target = min(value for _, value in valid) if metric in {"std", "rel_std"} else max(value for _, value in valid)
                tput_winner[(env, metric)] = {idx for idx, value in valid if value == target}

            for metric in ["p50", "p99"]:
                values = [_parse_float(system_data[env][metric]) for _, system_data in systems]
                valid = [
                    (idx, value)
                    for idx, value in enumerate(values)
                    if not np.isnan(value) and idx not in excluded_winner_rows
                ]
                if not valid:
                    lat_winner[(env, metric)] = set()
                    continue
                target = min(value for _, value in valid)
                lat_winner[(env, metric)] = {idx for idx, value in valid if value == target}

        for sys_idx, (system, data) in enumerate(systems):
            query_cell = rf"\multirow{{{len(systems)}}}{{*}}{{{query}}}" if sys_idx == 0 else ""
            ignore_row = _should_ignore_nes(system)
            row_parts = [query_cell, _maybe_dim(system, ignore_row)]

            for env in envs:
                mean_tex = _tex_num(data[env]["mean"])
                mean_bold = (not ignore_row) and sys_idx in tput_winner.get((env, "mean"), set())
                row_parts.append(_maybe_dim(_maybe_bold(mean_tex, mean_bold), ignore_row))

                std_tex = _tex_pm_std(
                    data[env]["std"],
                    data[env]["rel_std"],
                    bold_std=(not ignore_row) and sys_idx in tput_winner.get((env, "std"), set()),
                    bold_rel=(not ignore_row) and sys_idx in tput_winner.get((env, "rel_std"), set()),
                )
                row_parts.append(_maybe_dim(std_tex, ignore_row))

                min_tex = _tex_num(data[env]["min"])
                min_bold = (not ignore_row) and sys_idx in tput_winner.get((env, "min"), set())
                row_parts.append(_maybe_dim(_maybe_bold(min_tex, min_bold), ignore_row))

                max_tex = _tex_num(data[env]["max"])
                max_bold = (not ignore_row) and sys_idx in tput_winner.get((env, "max"), set())
                row_parts.append(_maybe_dim(_maybe_bold(max_tex, max_bold), ignore_row))

                p50_tex = _tex_num(data[env]["p50"])
                p50_bold = (not ignore_row) and sys_idx in lat_winner.get((env, "p50"), set())
                row_parts.append(_maybe_dim(_maybe_bold(p50_tex, p50_bold), ignore_row))

                p99_tex = _tex_num(data[env]["p99"])
                p99_bold = (not ignore_row) and sys_idx in lat_winner.get((env, "p99"), set())
                row_parts.append(_maybe_dim(_maybe_bold(p99_tex, p99_bold), ignore_row))

            add_gap_before_next = (
                sys_idx + 1 < len(systems)
                and _should_ignore_nes(systems[sys_idx + 1][0])
            )
            row_suffix = r" \\[0.15em]" if add_gap_before_next else r" \\"
            lines.append("  " + " & ".join(row_parts) + row_suffix)

        if query_idx < len(query_rows) - 1:
            lines.append(rf"  \cmidrule(lr){{1-{col_count}}}")

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    output_dir = os.path.dirname(output_tex)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return output_tex


def gen_boxplot_summary_data(
        df: pd.DataFrame,
        file_name: str,
        output_suffix: str = "",
        x_col: str = "exp_name",
        escape_tex: bool = True,
        include_outliers: bool = False,
        outlier_col: str = "outliers",
) -> str | tuple[str, str]:
    """Write boxplot summary stats to CSV for pgfplots.

    Expects columns: x_col, q1, median, q3, lower_whisker, upper_whisker.
    If include_outliers=True, also writes an outlier CSV and returns
    (summary_path, outliers_path).
    """
    required = {x_col, "q1", "median", "q3", "lower_whisker", "upper_whisker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for boxplot CSV: {', '.join(sorted(missing))}")

    suffix = f"_{output_suffix}" if output_suffix else ""
    output_dir = f"plot_data{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    export = df[[x_col, "q1", "median", "q3", "lower_whisker", "upper_whisker"]].copy()
    export["label"] = export[x_col].astype(str).apply(lambda x: exp_name_mapping.get(x, x))
    if escape_tex:
        export["label"] = (
            export["label"]
            .str.replace("\\", r"\\textbackslash{}", regex=False)
            .str.replace("_", r"\\_", regex=False)
            .str.replace("%", r"\\%", regex=False)
            .str.replace("&", r"\\&", regex=False)
            .str.replace("#", r"\\#", regex=False)
            .str.replace("{", r"\\{", regex=False)
            .str.replace("}", r"\\}", regex=False)
            .str.replace("$", r"\\$", regex=False)
            .str.replace("~", r"\\textasciitilde{}", regex=False)
            .str.replace("^", r"\\textasciicircum{}", regex=False)
            .str.replace("|", r"\\textbar{}", regex=False)
            .str.replace("\n", r"\\", regex=False)
        )

    export = export[["label", "q1", "median", "q3", "lower_whisker", "upper_whisker"]]
    output_path = f"{output_dir}/{file_name}_boxplot.csv"
    export.to_csv(output_path, index=False)

    if not include_outliers:
        return output_path

    if outlier_col not in df.columns:
        raise ValueError(f"Missing outlier column '{outlier_col}' for outlier export.")

    def _to_outlier_list(raw_value: object) -> list[float]:
        if raw_value is None:
            return []

        if isinstance(raw_value, (list, tuple, np.ndarray, pd.Series)):
            return [float(v) for v in raw_value if pd.notna(v)]

        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
            if isinstance(parsed, (list, tuple, np.ndarray, pd.Series)):
                return [float(v) for v in parsed if pd.notna(v)]
            return [float(parsed)]

        if pd.isna(raw_value):
            return []

        return [float(raw_value)]

    outlier_records = []
    for x_index, (x_value, raw_outliers) in enumerate(df[[x_col, outlier_col]].itertuples(index=False, name=None)):
        label = exp_name_mapping.get(str(x_value), str(x_value))
        if escape_tex:
            label = (
                label
                .replace("\\", r"\\textbackslash{}")
                .replace("_", r"\\_")
                .replace("%", r"\\%")
                .replace("&", r"\\&")
                .replace("#", r"\\#")
                .replace("{", r"\\{")
                .replace("}", r"\\}")
                .replace("$", r"\\$")
                .replace("~", r"\\textasciitilde{}")
                .replace("^", r"\\textasciicircum{}")
                .replace("|", r"\\textbar{}")
                .replace("\n", r"\\")
            )

        for outlier_index, value in enumerate(_to_outlier_list(raw_outliers)):
            outlier_records.append({
                "label": label,
                "x_index": int(x_index),
                "outlier_index": outlier_index,
                "outlier": value,
            })

    outliers_path = f"{output_dir}/{file_name}_boxplot_outliers.csv"
    pd.DataFrame(outlier_records, columns=["label", "x_index", "outlier_index", "outlier"]).to_csv(
        outliers_path,
        index=False,
    )

    return output_path, outliers_path
