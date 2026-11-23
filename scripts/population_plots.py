#!/usr/bin/env python3
"""
Create population-level plots for a BOOM pipeline run.

Currently this produces population distributions from the per-object metrics
written by plot_lc.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BANDS = ["u", "g", "r", "i", "z", "y"]
DELTA_MAX_BASELINE_DAYS = 20


def save_figure(fig, output_path):
    fig.savefig(output_path)
    pickle_path = os.path.splitext(output_path)[0] + ".pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(fig, f)
    return pickle_path


def load_object_metrics(object_metrics_csv):
    with open(object_metrics_csv, "r", newline="") as f:
        return list(csv.DictReader(f))


def float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def plot_delta_distribution(object_metrics, output_dir):

    n_excluded_long_baseline = 0
    for row in object_metrics:
        baseline_days = float_or_none(row.get("baseline_days"))
        if baseline_days is not None and baseline_days > DELTA_MAX_BASELINE_DAYS:
            n_excluded_long_baseline += 1

    n_singlepoint = 0
    n_both = 0

    for row in object_metrics:
        if row.get("delta_status") == "single_point":
            n_singlepoint += 1
            if float_or_none(row.get("baseline_days")) > DELTA_MAX_BASELINE_DAYS:
                n_both += 1


    used_deltas = [
        float_or_none(row.get("delta"))
        for row in object_metrics
        if row.get("delta_status") == "used"
        and float_or_none(row.get("baseline_days")) < DELTA_MAX_BASELINE_DAYS
    ]

    used_deltas = [value for value in used_deltas if value is not None]

    n_used = len(used_deltas)
    denominator = len(object_metrics)
    used_fraction = n_used / denominator if denominator > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 5))

    if used_deltas:
        bins = np.linspace(0, 1, 21)
        ax.hist(used_deltas, bins=bins, color="steelblue", edgecolor="white")
    else:
        ax.text(
            0.5,
            0.5,
            "No usable multi-point objects",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    if denominator > 0:
        proportion_text = f"Used objects: {n_used}/{denominator} = {used_fraction:.1%}"
    else:
        proportion_text = "Used objects: 0/0"
    ax.text(
        0.98,
        0.95,
        proportion_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    ax.text(
        0.98,
        0.85,
        f"Excluded baseline > {DELTA_MAX_BASELINE_DAYS} d: {n_excluded_long_baseline}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    ax.text(
        0.98,
        0.75,
        f"Excluded single-passing-point objects: {n_singlepoint}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    ax.text(
        0.98,
        0.65,
        f"Objects excluded for both conditions: {n_both}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )


    ax.set_title("Distribution of Relative Time Range of Interest")
    ax.set_xlabel("Relative time range of interest")
    ax.set_ylabel("Number of objects")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    output_path = os.path.join(output_dir, "delta_distribution.png")
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)

    return {
        "delta_distribution_plot": output_path,
        "n_used_deltas": n_used,
        "n_single_point_deltas": n_singlepoint,
        "used_delta_fraction": used_fraction,
        "n_delta_excluded_long_baseline": n_excluded_long_baseline,
    }


def values_from_column(rows, column):
    values = []
    for row in rows:
        value = float_or_none(row.get(column))
        if value is not None:
            values.append(value)
    return values


def integer_bins(values):
    if not values:
        return np.arange(-0.5, 1.5, 1)
    upper = int(max(values))
    return np.arange(-0.5, upper + 1.5, 1)


def logarithmic_bins(values, n_bins=20):
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return None

    lower = min(positive_values)
    upper = max(positive_values)
    if lower == upper:
        lower = lower / np.sqrt(10)
        upper = upper * np.sqrt(10)

    return np.logspace(np.log10(lower), np.log10(upper), n_bins + 1)


def annotate_used_objects(ax, n_used, n_total):
    if n_total > 0:
        text = f"Used objects: {n_used}/{n_total} = {n_used / n_total:.1%}"
    else:
        text = "Used objects: 0/0"

    annotate_box(ax, text)


def plot_fraction_histogram(ax, values, title, n_total):
    if values:
        bins = np.linspace(0, 1, 21)
        ax.hist(values, bins=bins, color="seagreen", edgecolor="white")
    else:
        ax.text(
            0.5,
            0.5,
            "No usable objects",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    annotate_used_objects(ax, len(values), n_total)


def plot_highlighted_fraction_by_band(object_metrics, output_dir):
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 8))
    axes = axes.flatten()
    n_total_objects = len(object_metrics)
    n_used_by_band = {}

    for i, band in enumerate(BANDS):
        values = values_from_column(object_metrics, f"highlighted_fraction_{band}")
        n_used_by_band[band] = len(values)
        plot_fraction_histogram(ax=axes[i], values=values, title=f"{band}-band", n_total=n_total_objects)
        axes[i].set_ylabel("Number of objects")
        if i >= 4:
            axes[i].set_xlabel("Highlighted / total LSST points")

    fig.suptitle("Highlighted-Point Fraction by Band")
    output_path = os.path.join(output_dir, "highlighted_fraction_by_band.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    plt.close(fig)

    return {
        "highlighted_fraction_by_band_plot": output_path,
        "n_used_highlighted_fraction_by_band": n_used_by_band,
    }


def plot_global_highlighted_fraction(object_metrics, output_dir):
    values = values_from_column(object_metrics, "highlighted_fraction")

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_fraction_histogram(
        ax=ax,
        values=values,
        title="Global Highlighted-Point Fraction",
        n_total=len(object_metrics),
    )
    ax.set_xlabel("Highlighted / total LSST points")
    ax.set_ylabel("Number of objects")

    output_path = os.path.join(output_dir, "highlighted_fraction_global.png")
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)

    return {
        "highlighted_fraction_global_plot": output_path,
        "n_used_highlighted_fraction_global": len(values),
    }


def plot_count_histogram(ax, values, title, bins):
    if values:
        ax.hist(values, bins=bins, color="slateblue", edgecolor="white")
    else:
        ax.text(
            0.5,
            0.5,
            "No objects",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_lsst_points_by_band(object_metrics, output_dir):
    values_by_band = {band: values_from_column(object_metrics, f"n_total_points_{band}") for band in BANDS}
    # The same integer bins in every panel, up to the largest count of any band
    bins = integer_bins([value for values in values_by_band.values() for value in values])

    # Each band keeps its own y scale: the u and y panels' tall 0-point bars
    # would otherwise flatten the other bands
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axes = axes.flatten()
    n_used_by_band = {}

    for i, band in enumerate(BANDS):
        values = values_by_band[band]
        n_used_by_band[band] = len(values)
        plot_count_histogram(axes[i], values, f"{band}-band", bins)
        axes[i].set_ylabel("Number of objects")
        if i >= 4:
            axes[i].set_xlabel("Number of LSST points")

    # Set once, after every panel is drawn: the x axis is shared, so a limit set
    # while drawing would freeze it at the first panel's range and crop the others
    axes[0].set_xlim(bins[0], bins[-1])

    fig.suptitle("Number of LSST Points by Band")
    output_path = os.path.join(output_dir, "lsst_points_by_band.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    plt.close(fig)

    return {
        "lsst_points_by_band_plot": output_path,
        "n_used_lsst_points_by_band": n_used_by_band,
    }


def plot_global_lsst_points(object_metrics, output_dir):
    values = values_from_column(object_metrics, "n_total_points")
    positive_values = [value for value in values if value > 0]
    n_zero_or_negative = len(values) - len(positive_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    if positive_values:
        ax.hist(
            positive_values,
            bins=logarithmic_bins(positive_values),
            color="slateblue",
            edgecolor="white",
        )
        ax.set_xscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            "No positive LSST point counts",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )

    if n_zero_or_negative:
        ax.text(
            0.98,
            0.95,
            f"Non-positive counts excluded: {n_zero_or_negative}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )

    ax.set_title("Global Number of LSST Points")
    ax.set_xlabel("Number of LSST points")
    ax.set_ylabel("Number of objects")
    ax.grid(axis="y", alpha=0.25)

    output_path = os.path.join(output_dir, "lsst_points_global.png")
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)

    return {
        "lsst_points_global_plot": output_path,
        "n_used_lsst_points_global": len(values),
    }


# =========================
# FIRST-PASSING ALERT, PEAK MAGNITUDE AND RATE DISTRIBUTIONS
# =========================

BAND_COLORS = {"u": "purple", "g": "green", "r": "red", "i": "orange", "z": "brown", "y": "gray"}

# Rates (mag/day) included in the rate histograms; the others are left out
RISING_RATE_RANGE = (-2.0, 0.0)
FADING_RATE_RANGE = (0.0, 2.0)


def annotate_box(ax, text):
    ax.text(
        0.98,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )


def show_no_data(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=10)


def leave_room_for_box(ax):
    """Raise the top of the y axis so the annotation box doesn't hide the tallest bars."""
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)


def common_bins(values_by_band, n_bins=20):
    """The same bins for every panel of a figure, over all its values."""
    values = [value for values in values_by_band.values() for value in values]
    if not values:
        return n_bins
    low, high = min(values), max(values)
    if low == high:
        low, high = low - 0.5, high + 0.5
    return np.linspace(low, high, n_bins + 1)


def save_band_figure(fig, axes, title, xlabel, output_path):
    for i, ax in enumerate(axes):
        ax.set_title(f"{BANDS[i]}-band")
        ax.set_ylabel("Number of objects")
        ax.grid(axis="y", alpha=0.25)
        if i >= 4:
            ax.set_xlabel(xlabel)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_path)
    plt.close(fig)


def plot_first_passing_delay(object_metrics, output_dir):
    values = values_from_column(object_metrics, "first_passing_delay_days")

    fig, ax = plt.subplots(figsize=(8, 5))
    if values:
        ax.hist(values, bins=120, color="steelblue", edgecolor="white")
        leave_room_for_box(ax)
    else:
        show_no_data(ax, "No objects")
    annotate_used_objects(ax, len(values), len(object_metrics))
    ax.set_title("Time From First Alert To First Filter-Passing Alert")
    ax.set_xlabel("t(first passing alert) - t(first alert) [days]")
    ax.set_ylabel("Number of objects")
    ax.grid(axis="y", alpha=0.25)

    output_path = os.path.join(output_dir, "first_passing_delay.png")
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)

    return {
        "first_passing_delay_plot": output_path,
        "n_used_first_passing_delay": len(values),
    }


def plot_rate_by_band(object_metrics, output_dir, column_prefix, title, filename, value_range):
    """
    One panel per band: the distribution of the rates in column
    <column_prefix>_<band>. Only the rates within value_range (low, high),
    bounds included, are used: shown in the histogram and counted in the legend.
    """
    low, high = value_range
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axes = axes.flatten()
    n_total = len(object_metrics)
    n_used_by_band = {}

    for ax, band in zip(axes, BANDS):
        values = [
            value for value in values_from_column(object_metrics, f"{column_prefix}_{band}")
            if low <= value <= high
        ]
        n_used_by_band[band] = len(values)
        if values:
            ax.hist(values, bins=np.linspace(low, high, 21), color=BAND_COLORS[band], edgecolor="white")
            leave_room_for_box(ax)
        else:
            show_no_data(ax, f"No rates between {low:g} and {high:g} mag/day")
        ax.set_xlim(low, high)
        annotate_used_objects(ax, len(values), n_total)

    output_path = os.path.join(output_dir, filename)
    save_band_figure(fig, axes, title, "Rate [mag/day]", output_path)

    return {
        f"{column_prefix}_by_band_plot": output_path,
        f"n_used_{column_prefix}_by_band": n_used_by_band,
    }


def plot_peak_mag_by_band(object_metrics, output_dir):
    values_by_band = {band: values_from_column(object_metrics, f"peak_mag_{band}") for band in BANDS}
    bins = common_bins(values_by_band)

    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axes = axes.flatten()
    for ax, band in zip(axes, BANDS):
        values = values_by_band[band]
        if values:
            ax.hist(values, bins=bins, color=BAND_COLORS[band], edgecolor="white")
            leave_room_for_box(ax)
        else:
            show_no_data(ax, "No detections")
        annotate_used_objects(ax, len(values), len(object_metrics))

    output_path = os.path.join(output_dir, "peak_mag_by_band.png")
    save_band_figure(fig, axes, "Peak (Brightest) Magnitude by Band", "Peak magpsf", output_path)

    return {
        "peak_mag_by_band_plot": output_path,
        "n_used_peak_mag_by_band": {band: len(values) for band, values in values_by_band.items()},
    }


def plot_first_passing_mag_by_band(object_metrics, output_dir):
    """
    Magnitude of each object's first filter-passing alert, in the panel of that
    alert's band: every object appears in exactly one panel.
    """
    values_by_band = {band: [] for band in BANDS}
    for row in object_metrics:
        band = row.get("first_passing_band")
        mag = float_or_none(row.get("first_passing_mag"))
        if band in values_by_band and mag is not None:
            values_by_band[band].append(mag)
    bins = common_bins(values_by_band)

    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axes = axes.flatten()
    for ax, band in zip(axes, BANDS):
        values = values_by_band[band]
        if values:
            ax.hist(values, bins=bins, color=BAND_COLORS[band], edgecolor="white")
            leave_room_for_box(ax)
        else:
            show_no_data(ax, "No first passing alert in this band")
        annotate_box(ax, f"Objects: {len(values)}/{len(object_metrics)}")

    output_path = os.path.join(output_dir, "first_passing_mag_by_band.png")
    save_band_figure(
        fig, axes, "Magnitude of the First Filter-Passing Alert, by Its Band", "magpsf", output_path
    )

    return {
        "first_passing_mag_by_band_plot": output_path,
        "n_first_passing_mag_by_band": {band: len(values) for band, values in values_by_band.items()},
    }


def plot_first_passing_band(object_metrics, output_dir):
    counts = [sum(1 for row in object_metrics if row.get("first_passing_band") == band) for band in BANDS]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(BANDS, counts, color=[BAND_COLORS[band] for band in BANDS], edgecolor="white")
    ax.bar_label(bars)
    leave_room_for_box(ax)
    annotate_box(ax, f"Objects: {sum(counts)}/{len(object_metrics)}")
    ax.set_title("Band of the First Filter-Passing Alert")
    ax.set_xlabel("Band")
    ax.set_ylabel("Number of objects")
    ax.grid(axis="y", alpha=0.25)

    output_path = os.path.join(output_dir, "first_passing_band.png")
    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)

    return {
        "first_passing_band_plot": output_path,
        "n_first_passing_band": dict(zip(BANDS, counts)),
    }


def main(object_metrics_csv, output_dir, output_subdir="pop_plots"):
    output_dir = os.path.join(output_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    object_metrics = load_object_metrics(object_metrics_csv)
    result = plot_delta_distribution(object_metrics, output_dir)
    result.update(plot_highlighted_fraction_by_band(object_metrics, output_dir))
    result.update(plot_global_highlighted_fraction(object_metrics, output_dir))
    result.update(plot_lsst_points_by_band(object_metrics, output_dir))
    result.update(plot_global_lsst_points(object_metrics, output_dir))
    result.update(plot_first_passing_delay(object_metrics, output_dir))
    result.update(plot_first_passing_band(object_metrics, output_dir))
    result.update(plot_first_passing_mag_by_band(object_metrics, output_dir))
    result.update(plot_peak_mag_by_band(object_metrics, output_dir))
    rate_plots = [
        ("final_rising_rate", "Final Rising Rate by Band (latest alert of each band)", "final_rising_rate_by_band.png", RISING_RATE_RANGE),
        ("final_fading_rate", "Final Fading Rate by Band (latest alert of each band)", "final_fading_rate_by_band.png", FADING_RATE_RANGE),
        ("first_passing_rising_rate", "Rising Rate at the First Filter-Passing Alert, by Band", "first_passing_rising_rate_by_band.png", RISING_RATE_RANGE),
        ("first_passing_fading_rate", "Fading Rate at the First Filter-Passing Alert, by Band", "first_passing_fading_rate_by_band.png", FADING_RATE_RANGE),
    ]
    for column_prefix, title, filename, value_range in rate_plots:
        result.update(plot_rate_by_band(object_metrics, output_dir, column_prefix, title, filename, value_range))
    result["pop_plots_dir"] = output_dir

    print(f"Saved delta distribution to {result['delta_distribution_plot']}")
    print(f"Saved highlighted-point fraction by band to {result['highlighted_fraction_by_band_plot']}")
    print(f"Saved global highlighted-point fraction to {result['highlighted_fraction_global_plot']}")
    print(f"Saved LSST point counts by band to {result['lsst_points_by_band_plot']}")
    print(f"Saved global LSST point counts to {result['lsst_points_global_plot']}")
    print(f"Saved first-passing delay distribution to {result['first_passing_delay_plot']}")
    print(f"Saved first-passing band distribution to {result['first_passing_band_plot']}")
    print(f"Saved first-passing magnitude by band to {result['first_passing_mag_by_band_plot']}")
    print(f"Saved peak magnitude by band to {result['peak_mag_by_band_plot']}")
    for column_prefix, _, _, _ in rate_plots:
        print(f"Saved {column_prefix.replace('_', ' ')} by band to {result[f'{column_prefix}_by_band_plot']}")
    print(
        "Used objects for relative time range of interest: "
        f"{result['n_used_deltas']}/"
        f"{result['n_used_deltas'] + result['n_single_point_deltas']}"
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create population-level BOOM run plots.")
    parser.add_argument("--object_metrics_csv", required=True, help="Path to object_metrics.csv from plot_lc.py")
    parser.add_argument("--output_dir", required=True, help="Run directory where pop_plots is created")
    parser.add_argument("--output_subdir", default="pop_plots", help="Population plot subfolder name")
    args = parser.parse_args()

    main(
        args.object_metrics_csv,
        args.output_dir,
        output_subdir=args.output_subdir,
    )
