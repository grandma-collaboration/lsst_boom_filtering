import json
import os
import argparse
import csv
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from astropy.time import Time


# Color mapping
COLORS = {
    "u": "purple",
    "g": "green",
    "r": "red",
    "i": "orange",
    "z": "brown",
    "y": "gray"
}

BANDS = ["u", "g", "r", "i", "z", "y"]

# =========================
# CONFIG
# =========================

INCLUDE_ZTF = False
INCLUDE_FP = True

# Seconds before a BOOM request times out (boom-client's default is 60 s)
BOOM_TIMEOUT_S = 300

# =========================
# DATA LOADING
# =========================
def save_figure(fig, output_path):
    fig.savefig(output_path)
    pickle_path = os.path.splitext(output_path)[0] + ".pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(fig, f)
    return pickle_path


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def make_exclude_jd(exclude_start_time, exclude_end_time):
    """
    Convert an optional UTC ISO exclusion interval to a (start_jd, end_jd) tuple.

    Here it is only used to shade the interval on plots: the exclusion itself
    is applied by boom_filter_per_run.py, which drops every object with a
    point in it.
    """
    if exclude_start_time is None and exclude_end_time is None:
        return None
    if exclude_start_time is None or exclude_end_time is None:
        raise ValueError("exclude_start_time and exclude_end_time must be provided together")
    return (
        Time(exclude_start_time, scale="utc").jd,
        Time(exclude_end_time, scale="utc").jd,
    )


def in_excluded_interval(point, exclude_jd):
    """
    True if the point falls in the half-open excluded interval [start_jd, end_jd).
    Not used in this file; imported by cleaned_population_plots.py.
    """
    if exclude_jd is None or point.get("jd") is None:
        return False
    return exclude_jd[0] <= point["jd"] < exclude_jd[1]


def empty_object_metric(obj_id, status):
    metric = {
        "objectId": obj_id,
        "delta": None,
        "delta_status": status,
        "n_highlighted_points": 0,
        "n_total_points": 0,
        "highlighted_fraction": None,
        "baseline_days": None,
        "highlighted_window_days": None,
        "highlighted_start_day": None,
        "highlighted_stop_day": None,
    }
    for band in BANDS:
        metric[f"n_highlighted_points_{band}"] = 0
        metric[f"n_total_points_{band}"] = 0
        metric[f"highlighted_fraction_{band}"] = None
    return metric


def write_object_metrics(metrics, output_csv):
    fieldnames = [
        "objectId",
        "delta",
        "delta_status",
        "n_highlighted_points",
        "n_total_points",
        "highlighted_fraction",
        "baseline_days",
        "highlighted_window_days",
        "highlighted_start_day",
        "highlighted_stop_day",
    ]
    for band in BANDS:
        fieldnames.extend([
            f"n_highlighted_points_{band}",
            f"n_total_points_{band}",
            f"highlighted_fraction_{band}",
        ])

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def first_valid_coordinates(points):
    for point in points:
        ra = point.get("ra")
        dec = point.get("dec")
        if ra is not None and dec is not None:
            return ra, dec
    return None, None

# =========================
# ZTF CROSS-MATCH
# =========================

def match_ztf_object(client, ra, dec, radius_arcsec=1.0, limit=100):
    # The ZTF objects within radius_arcsec, nearest first, each with its
    # distance_arcsec (computed from coordinates.radec_geojson)
    data = client.near_search(
        "ZTF_alerts_aux",
        ra,
        dec,
        radius_arcsec=radius_arcsec,
        projection={
                "_id": 1,                       # ZTF ID of the object
                "coordinates.radec_geojson": 1, # Position of the object, for the distance

                "prv_candidates.jd": 1,         # Photometry for the 5-sigma detections
                "prv_candidates.magpsf": 1,
                "prv_candidates.sigmapsf": 1,
                "prv_candidates.band": 1,
                "prv_candidates.snr_psf": 1,
                "prv_candidates.ra": 1,
                "prv_candidates.dec": 1,

                "prv_nondetections.jd": 1,     # Photometry for the non-detections (upper limits)
                "prv_nondetections.diffmaglim": 1,
                "prv_nondetections.band": 1,

        },
        limit=limit,
    )

    if len(data) == 0:
        print("No ZTF match found")
        return None, [], []

    # -----------------------------------
    # Find closest object with detections
    # -----------------------------------
    best_obj = None

    for obj in data:
        print(f"Checking ZTF object {obj.get('_id')}: separation {obj.get('distance_arcsec', float('nan')):.2f} arcsec")

        if obj.get("prv_candidates"):
            best_obj = obj
            break

    if best_obj is None:
        print("No ZTF object with detections")
        return None, [], []

    # -----------------------------------
    # Extract detections
    # -----------------------------------
    detections = []
    for p in best_obj.get("prv_candidates", []):
        if is_valid_point(p):
            detections.append(p)

    # -----------------------------------
    # Extract upper limits
    # -----------------------------------
    limits = []
    for p in best_obj.get("prv_nondetections", []):
        if is_valid_ulim(p):
            limits.append(p)

    print(f"Matched ZTF object: {best_obj.get('_id')} (dist ≈ {best_obj.get('distance_arcsec', float('nan')):.2f} arcsec)")

    return best_obj, detections, limits

# =========================
# ALERT HIGHLIGHTING
# =========================
def get_alert_points_passed(alerts_data):
    """
    Build a mapping from objectId -> set of (band, jd) that passed the filter.
    """
    alert_points_passed = defaultdict(set)
    for alert in alerts_data:
        obj_id = alert["objectId"]
        band = alert["candidate"]["band"]
        jd = alert["candidate"]["jd"]
        
        if band is None or jd is None:
            continue

        alert_points_passed[obj_id].add((band, jd))
    return alert_points_passed

def is_highlight(band, jd, passed_points, tol=1e-6):
    for b, j in passed_points:
        if b == band and abs(jd - j) < tol:
            return True
    return False


# ====================================
# PHOTOMETRY QUALITY CHECK
# ====================================

def is_valid_point(p):
    bool_magpsf = "magpsf" in p and p["magpsf"] is not None
    bool_sigmapsf = "sigmapsf" in p and p["sigmapsf"] is not None
    bool_jd = "jd" in p and p["jd"] is not None
    bool_band = "band" in p and p["band"] is not None
    bool_snr = "snr_psf" in p and p["snr_psf"] is not None and p["snr_psf"] >= 2

    return (bool_magpsf and bool_sigmapsf and bool_jd and bool_band and bool_snr)

def is_valid_ulim(p):
    bool_jd = "jd" in p and p["jd"] is not None
    bool_lim = "diffmaglim" in p and p["diffmaglim"] is not None
    bool_band = "band" in p and p["band"] is not None

    return (bool_jd and bool_lim and bool_band)


def compute_object_metrics(obj, alert_points_passed):
    obj_id = obj.get("_id", obj.get("objectId"))
    lc_points = obj.get("prv_candidates", [])

    points_by_band = defaultdict(list)
    for p in lc_points:
        if p.get("band") is not None and p.get("jd") is not None and p.get("magpsf") is not None:
            points_by_band[p["band"]].append(p)

    all_lsst_jds = [p["jd"] for points in points_by_band.values() for p in points]
    if not all_lsst_jds:
        return empty_object_metric(obj_id, "no_valid_lsst_photometry")

    jd_min = min(all_lsst_jds)
    baseline = max(all_lsst_jds) - jd_min

    n_total_points_by_band = {band: 0 for band in BANDS}
    n_highlighted_points_by_band = {band: 0 for band in BANDS}
    all_highlighted_x = []

    for band in BANDS:
        band_points = points_by_band.get(band, [])
        band_points.sort(key=lambda point: point["jd"])

        highlights = [
            is_highlight(band, p["jd"], alert_points_passed[obj_id])
            for p in band_points
        ]
        x_high = [
            p["jd"] - jd_min
            for p, highlight in zip(band_points, highlights)
            if highlight
        ]

        n_total_points_by_band[band] = len(band_points)
        n_highlighted_points_by_band[band] = sum(highlights)
        all_highlighted_x.extend(x_high)

    highlighted_window = None
    highlighted_window_days = None
    delta_status = "no_highlights"
    object_delta = 0
    if len(all_highlighted_x) >= 2 and baseline > 0:
        highlighted_window = (min(all_highlighted_x), max(all_highlighted_x))
        highlighted_window_days = highlighted_window[1] - highlighted_window[0]
        object_delta = highlighted_window_days / baseline
        delta_status = "used"
    elif len(all_highlighted_x) == 1:
        delta_status = "single_point"
    elif len(all_highlighted_x) >= 2:
        delta_status = "zero_baseline"

    n_total_points = sum(n_total_points_by_band.values())
    highlighted_count = sum(n_highlighted_points_by_band.values())
    highlighted_fraction = highlighted_count / n_total_points if n_total_points > 0 else None

    metric = {
        "objectId": obj_id,
        "delta": object_delta if delta_status == "used" else None,
        "delta_status": delta_status,
        "n_highlighted_points": highlighted_count,
        "n_total_points": n_total_points,
        "highlighted_fraction": highlighted_fraction,
        "baseline_days": baseline,
        "highlighted_window_days": highlighted_window_days,
        "highlighted_start_day": highlighted_window[0] if highlighted_window is not None else None,
        "highlighted_stop_day": highlighted_window[1] if highlighted_window is not None else None,
    }
    for band in BANDS:
        n_total_band = n_total_points_by_band[band]
        n_highlighted_band = n_highlighted_points_by_band[band]
        metric[f"n_highlighted_points_{band}"] = n_highlighted_band
        metric[f"n_total_points_{band}"] = n_total_band
        metric[f"highlighted_fraction_{band}"] = (
            n_highlighted_band / n_total_band if n_total_band > 0 else None
        )

    return metric

# =========================
# LIGHT CURVE PLOTTING
# =========================
def plot_object_lc(client, obj, alert_points_passed, output_dir, iter, exclude_jd=None, band_rates=None):
    obj_id = obj.get("_id", obj.get("objectId"))
    metric = compute_object_metrics(obj, alert_points_passed)
    band_rates = band_rates or {}

    print(f"\n\n\n      ========================== Object {obj_id} ({iter+1}) ==========================\n")

    lc_points = obj.get("prv_candidates", [])
    if INCLUDE_FP:
        lc_point_fp = obj.get("fp_hists", [])
    else:
        lc_point_fp = []

    obj_ra, obj_dec = first_valid_coordinates(lc_points)
    if obj_ra is not None and obj_dec is not None:
        print(f"Object coordinates: RA={obj_ra:.4f}, Dec={obj_dec:.4f}")
    else:
        print("Object coordinates unavailable; skipping ZTF crossmatch for this object")

    lc_points_ztf = []
    ztf_ulims = []
    if INCLUDE_ZTF and obj_ra is not None and obj_dec is not None:
        best_match, lc_points_ztf, ztf_ulims = match_ztf_object(client, ra=obj_ra, dec=obj_dec)

    if not lc_points and not lc_point_fp and not lc_points_ztf:
        print("No plottable photometry for this object")
        return empty_object_metric(obj_id, "no_photometry")

    # Organize points by band
    points_by_band = defaultdict(list)
    for p in lc_points:
        if p.get("band") is not None and p.get("jd") is not None and p.get("magpsf") is not None:
            points_by_band[p["band"]].append(p)

    points_by_band_fp = defaultdict(list)
    if INCLUDE_FP:
        for p in lc_point_fp:
            if is_valid_point(p):
                points_by_band_fp[p["band"]].append(p)

    points_by_band_ztf = defaultdict(list)
    if INCLUDE_ZTF:
        for p in lc_points_ztf:
            if is_valid_point(p):
                points_by_band_ztf[p["band"]].append(p)

    # Compute earliest JD across all bands
    all_plot_jds = (
        [p["jd"] for points in points_by_band.values() for p in points]
        + [p["jd"] for points in points_by_band_fp.values() for p in points]
        + [p["jd"] for points in points_by_band_ztf.values() for p in points]
    )
    if not all_plot_jds:
        print("No valid photometry survived quality cuts for this object")
        return empty_object_metric(obj_id, "no_valid_photometry")

    jd_min = min(all_plot_jds)
    jd_min_iso = Time(jd_min, format='jd', scale='utc').to_datetime()

    # Setup subplots
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axes = axes.flatten()

    x_min, x_max = 1e30, 0

    highlighted_count = 0
    all_highlighted_x = []
    n_total_points_by_band = {band: 0 for band in BANDS}
    n_highlighted_points_by_band = {band: 0 for band in BANDS}

    for i, band in enumerate(BANDS):
        ax = axes[i]
        band_points = points_by_band.get(band, [])
        band_points_fp = points_by_band_fp.get(band, [])
        band_points_ztf = points_by_band_ztf.get(band, [])
        band_color = COLORS.get(band, "black")

        ax.set_ylabel(f"{band}-band")
        if i >= 4:  # bottom row shows x-axis
            ax.set_xlabel("Age (days)")

        # BOOM's latest rising/fading rate for this band, as invisible proxy
        # artists so they show up in the legend even on an otherwise-empty panel
        band_rate_info = band_rates.get(band, {})
        rising_rate = band_rate_info.get("rising_rate")
        fading_rate = band_rate_info.get("fading_rate")
        if rising_rate is not None:
            ax.plot([], [], ' ', label=f"Rising {rising_rate:.2f} mag/day")
        if fading_rate is not None:
            ax.plot([], [], ' ', label=f"Fading {fading_rate:.2f} mag/day")

        if not band_points and not band_points_fp and not band_points_ztf:
            continue
        
        if INCLUDE_FP:
            if not band_points_fp:
                print(f"No forced photometry points in band {band}")

        # ========================================
        #           Sort points by JD 
        # ========================================

        # LSST points
        band_points.sort(key=lambda x: x["jd"])
        x = [p["jd"] - jd_min for p in band_points]
        y = [p["magpsf"] for p in band_points]
        yerr = [p.get("sigmapsf", 0.0) for p in band_points]

        # Forced photometry points
        if INCLUDE_FP:
            band_points_fp.sort(key=lambda x: x["jd"])
            x_fp = [p["jd"] - jd_min for p in band_points_fp]
            y_fp = [p["magpsf"] for p in band_points_fp]
            yerr_fp = [p.get("sigmapsf", 0.0) for p in band_points_fp]
        else:
            x_fp, y_fp, yerr_fp = [], [], []

        # ZTF points
        if INCLUDE_ZTF:
            band_points_ztf.sort(key=lambda x: x["jd"])
            x_ztf = [p["jd"] - jd_min for p in band_points_ztf]
            y_ztf = [p["magpsf"] for p in band_points_ztf]
            yerr_ztf = [p.get("sigmapsf", 0.0) for p in band_points_ztf]
        else:
            x_ztf, y_ztf, yerr_ztf = [], [], []

        # Computing the maximum time range
        x_all = x + x_fp + x_ztf
        if x_all:
            x_min = min(x_min, min(x_all))
            x_max = max(x_max, max(x_all))

        # ========================================
        #      Highlighting relevant points
        # ========================================

        # Identify which points to highlight ( <=> the LSST points that passed the filter)
        highlights = [is_highlight(band, p["jd"], alert_points_passed[obj_id]) for p in band_points]

        # Updating the count of highlighted points
        n_highlighted_band = sum(highlights)
        n_total_points_by_band[band] = len(band_points)
        n_highlighted_points_by_band[band] = n_highlighted_band
        highlighted_count += n_highlighted_band

        # ========================================
        #      Plotting the light curves
        # ========================================

        # Plot forced photometry points (if available)
        if INCLUDE_FP and x_fp:
            if all(yerr_fp):
                ax.errorbar(x_fp, y_fp, yerr=yerr_fp, fmt='x', color=band_color, alpha=0.2, label='fp')
            else:
                ax.scatter(x_fp, y_fp, color=band_color, alpha=0.2, label='fp')

        # Plot ZTF points (if available)
        if INCLUDE_ZTF and x_ztf:
            if all(yerr_ztf):
                ax.errorbar(x_ztf, y_ztf, yerr=yerr_ztf, fmt='s', color='cyan', alpha=0.7, label='ZTF')
            else:
                ax.scatter(x_ztf, y_ztf, color='cyan', alpha=0.7, label='ZTF')

        # Plot normal LSST points
        x_norm = [xx for xx, h in zip(x, highlights) if not h]
        y_norm = [yy for yy, h in zip(y, highlights) if not h]
        yerr_norm = [ee for ee, h in zip(yerr, highlights) if not h]
        if x_norm:
            if all(yerr_norm):
                ax.errorbar(x_norm, y_norm, yerr=yerr_norm, fmt='o', color=band_color, alpha=0.7)             # label='Other points'
            else:
                ax.scatter(x_norm, y_norm, color=band_color, alpha=0.7)                                       # label='Other points'

        # Plot highlighted LSST points
        x_high = [xx for xx, h in zip(x, highlights) if h]
        y_high = [yy for yy, h in zip(y, highlights) if h]
        yerr_high = [ee for ee, h in zip(yerr, highlights) if h]

        if x_high:
            all_highlighted_x.extend(x_high)
            ax.errorbar(x_high, y_high, yerr=yerr_high, fmt='o', color='black', markersize=8, markeredgecolor='yellow', markeredgewidth=1.5, label='Passed filter')

        ax.invert_yaxis()
    
    # Compute the overall period of interest ratio across all bands
    baseline = x_max - x_min
    highlighted_window = None
    highlighted_window_days = None
    delta_status = "no_highlights"
    if len(all_highlighted_x) >= 2 and baseline > 0:
        highlighted_window = (min(all_highlighted_x), max(all_highlighted_x))
        highlighted_window_days = highlighted_window[1] - highlighted_window[0]
        object_delta = highlighted_window_days / baseline
        delta_status = "used"
    else:
        object_delta = 0
        if len(all_highlighted_x) == 1:
            delta_status = "single_point"
        elif len(all_highlighted_x) >= 2:
            delta_status = "zero_baseline"

    if highlighted_window is not None:
        for i, ax in enumerate(axes):
            label = "Time range of interest" if i == 0 else "_nolegend_"
            ax.axvspan(highlighted_window[0], highlighted_window[1], color="gray", alpha=0.5, label=label)

    # Shade the excluded interval where it overlaps the plotted range
    if exclude_jd is not None:
        shade_lo = max(exclude_jd[0] - jd_min, x_min)
        shade_hi = min(exclude_jd[1] - jd_min, x_max)
        if shade_lo < shade_hi:
            for i, ax in enumerate(axes):
                label = "Excluded time range" if i == 0 else "_nolegend_"
                ax.axvspan(shade_lo, shade_hi, color="red", alpha=0.1, label=label)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right')

    metric_delta = metric["delta"] if metric["delta"] is not None else 0
    if metric_delta > 0.1:
        print(f"Significant relative time range of interest: {metric_delta:.2f}")

    title_time_range = metric_delta
    if metric["delta_status"] == "single_point":
        title_time_range = "single point"

    n_total_points = metric["n_total_points"]
    highlighted_count = metric["n_highlighted_points"]
    highlighted_fraction = metric["highlighted_fraction"] or 0
    if obj_ra is not None and obj_dec is not None:
        coord_text = f"RA, Dec = {obj_ra:.4f}, {obj_dec:.4f}"
    else:
        coord_text = "RA, Dec = N/A"
    plt.suptitle(
        f"Light Curve for Object {obj_id} ({iter+1}), T0 = {jd_min_iso.strftime('%Y-%m-%d %H:%M:%S')} UTC, "
        f"Rel. time range of interest = {title_time_range}\n"
        f"Filter-passing LSST points: {highlighted_count}/{n_total_points} ({highlighted_fraction:.1%}); "
        f"{coord_text}"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = os.path.join(output_dir, f"lc_{iter+1}_object_{obj_id}.png")
    pickle_path = save_figure(fig, output_path)
    plt.close(fig)
    print(f"\nSaved to {output_path}")
    print(f"Saved figure pickle to {pickle_path}")

    return metric


# =========================
# MAIN
# =========================
def main(
    objects_json_file,
    alerts_json_file,
    output_dir,
    include_ztf=None,
    include_fp=None,
    output_subdir=None,
    max_objects=None,
    make_plots=True,
    exclude_start_time=None,
    exclude_end_time=None,
    band_rates_json=None,
    client=None,
):
    global INCLUDE_ZTF, INCLUDE_FP

    if include_ztf is not None:
        INCLUDE_ZTF = include_ztf
    if include_fp is not None:
        INCLUDE_FP = include_fp

    os.makedirs(output_dir, exist_ok=True)
    # A BOOM client is only needed for the ZTF cross-match. Run on its own,
    # this script makes its own (credentials from BOOM_API_USER and
    # BOOM_API_PASSWORD); boom_pipeline.py passes its own. boom is imported
    # here so the scripts that import helpers from this file don't need it.
    if make_plots and INCLUDE_ZTF and client is None:
        from boom import BoomClient
        client = BoomClient(timeout=BOOM_TIMEOUT_S)

    highlight_count = 0
    exclude_jd = make_exclude_jd(exclude_start_time, exclude_end_time)

    # Output subfolder setup
    if output_subdir is None:
        objs_file = os.path.basename(objects_json_file)
        stem = os.path.splitext(objs_file)[0]
        output_subdir = stem[5:] if stem.startswith("objs_") else stem
    plots_dir = os.path.join(output_dir, output_subdir)
    os.makedirs(plots_dir, exist_ok=True)

    # Load local data
    objects_data = load_json(objects_json_file)
    alerts_data = load_json(alerts_json_file)

    # Alerts that passed the filter
    alert_points_passed = get_alert_points_passed(alerts_data)

    # Per-object, per-band rising/fading rates, as of each band's most recent
    # LSST detection (not just its most recent filter-passing one). Written by
    # boom_filter_per_run.py alongside objs.json/cands.json, since the rate
    # only exists on LSST_alerts documents and has to be queried separately.
    rates_by_object = load_json(band_rates_json) if band_rates_json else {}

    # Plotting loop
    if max_objects is not None:
        objects_data = objects_data[:max_objects]

    object_metrics = []
    for i, obj in enumerate(objects_data):
        obj_id = obj.get("_id", obj.get("objectId"))
        if make_plots:
            metric = plot_object_lc(
                client, obj, alert_points_passed, plots_dir, i,
                exclude_jd=exclude_jd, band_rates=rates_by_object.get(obj_id, {}),
            )
        else:
            metric = compute_object_metrics(obj, alert_points_passed)
        object_metrics.append(metric)
        highlight_count += metric["n_highlighted_points"]

    # Next to the run's other JSON and CSV files, not in the plot folder
    object_metrics_csv = os.path.join(output_dir, "object_metrics.csv")
    write_object_metrics(object_metrics, object_metrics_csv)
    print(f"Saved object metrics to {object_metrics_csv}")

    print(f"Total highlighted points: {highlight_count}")
    return {
        "plots_dir": plots_dir,
        "object_metrics_csv": object_metrics_csv,
        "n_objects_plotted": len(objects_data) if make_plots else 0,
        "n_objects_with_metrics": len(objects_data),
        "n_highlighted_points": highlight_count,
        "include_ztf": INCLUDE_ZTF,
        "include_fp": INCLUDE_FP,
        "make_plots": make_plots,
        "exclude_start_time": exclude_start_time,
        "exclude_end_time": exclude_end_time,
        "band_rates_json": band_rates_json,
    }

"""
example command to run:
python3 plot_lc.py --objects_json path/to/run_dir/objs.json --alerts_json path/to/run_dir/cands.json --output_dir path/to/run_dir --output_subdir plots --include_ztf --band_rates_json path/to/run_dir/band_rates.json
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot LSST light curves with highlights")
    parser.add_argument("--objects_json", required=True, help="Path to objects JSON file")
    parser.add_argument("--alerts_json", required=True, help="Path to alerts JSON file")
    parser.add_argument("--output_dir", default="./lc_plots/", help="Directory where object_metrics.csv and the plot subfolder are written")
    parser.add_argument("--include_ztf", action="store_true", help="Include ZTF crossmatch photometry when available")
    parser.add_argument("--no_fp", action="store_true", help="Do not include LSST forced photometry")
    parser.add_argument("--output_subdir", default=None, help="Optional plot subfolder name")
    parser.add_argument("--max_objects", type=int, default=None, help="Optional maximum number of objects to plot")
    parser.add_argument("--skip_individual_plots", action="store_true", help="Only write LSST-only object metrics; do not plot individual light curves")
    parser.add_argument("--exclude_start_time", default=None, help="Optional UTC ISO start of the run's excluded interval, shaded on plots for reference (objects with points in it are dropped by boom_filter_per_run.py, not here)")
    parser.add_argument("--exclude_end_time", default=None, help="Optional UTC ISO end of that interval")
    parser.add_argument("--band_rates_json", default=None, help="Optional path to band_rates.json from boom_filter_per_run.py, for the rising/fading rate legend entries")
    args = parser.parse_args()

    main(
        args.objects_json,
        args.alerts_json,
        args.output_dir,
        include_ztf=args.include_ztf,
        include_fp=not args.no_fp,
        output_subdir=args.output_subdir,
        max_objects=args.max_objects,
        make_plots=not args.skip_individual_plots,
        exclude_start_time=args.exclude_start_time,
        exclude_end_time=args.exclude_end_time,
        band_rates_json=args.band_rates_json,
    )
