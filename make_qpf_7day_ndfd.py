# make_qpf_7day_ndfd.py
# ---------------------------------------------------------
# NWS NDFD DAILY QPF (inches) – Local Calendar Day Totals
#
# Fixes / Style:
# - Smooth radar-style gradient
# - LogNorm so low-end ticks (0.1/0.25/0.5/1) are not crammed together
# - HARD CAP at 5.0": anything >= 5" is the same top color (no extra scaling)
# - VP.004-007 may 404; handled gracefully
# ---------------------------------------------------------

import os
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =========================
# CONFIG
# =========================
VAR_NAME = "qpf"
TITLE = "Daily Total Precipitation (QPF)"
UNITS_LABEL = "in"

NDFD_URLS = [
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.qpf.bin",
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.qpf.bin",  # may 404
]

OUTDIR = "maps"
TMPFILES = [
    os.path.join(OUTDIR, "_ndfd_qpf_001_003.grib2"),
    os.path.join(OUTDIR, "_ndfd_qpf_004_007.grib2"),
]

STATEFILE = os.path.join(OUTDIR, "_last_ndfd_qpf_daily_7day.json")
MANIFEST = os.path.join(OUTDIR, "manifest_qpf.json")
LATEST_OUTFILE = os.path.join(OUTDIR, "latest_qpf_ndfd.png")
PNG_PATTERN = os.path.join(OUTDIR, "ndfd_qpf_f{idx:03d}.png")

EXTENT = (-125, -66.5, 24, 50.5)

DPI = 150
WATERMARK_TEXT = "WxLaunch"

MAX_DAYS = 7
FORCE_RENDER = False

LOCAL_TZ = ZoneInfo("America/New_York")

# Threshold / range
QPF_MIN_IN = 0.01
QPF_VMAX_IN = 5.0   # <-- HARD CAP at 5"

# Ticks (stop at 5")
COLORBAR_TICKS = [0.1, 0.25, 0.5, 1, 2, 3, 5]
COLORBAR_TICKLABELS = ["0.1", "0.25", "0.5", "1", "2", "3", "5+"]


# =========================
# UTILITIES
# =========================
def ensure_dir(path_or_dir: str) -> None:
    d = path_or_dir
    if os.path.splitext(path_or_dir)[1]:
        d = os.path.dirname(path_or_dir)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def download_optional(url: str, dest: str) -> bool:
    try:
        r = requests.get(url, timeout=180)
        if r.status_code == 404:
            print(f"Missing on tgftp (404), skipping: {url}")
            return False
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"Download failed: {url}\n  -> {e}")
        return False


def mm_to_in(mm):
    return mm / 25.4


def format_day_label(day_date):
    return day_date.strftime("%a %b %d").replace(" 0", " ")


# =========================
# COLOR
# =========================
def qpf_cmap():
    # Smooth radar-ish gradient:
    # light -> green -> yellow -> orange -> red -> purple (top)
    return mcolors.LinearSegmentedColormap.from_list(
        "wxlaunch_qpf",
        [
            "#e8f4ff",
            "#b7e3ff",
            "#7fe6a2",
            "#2ecc71",
            "#f1c40f",
            "#f39c12",
            "#e67e22",
            "#e74c3c",
            "#9b59b6",
            "#6f2dbd",
        ],
        N=256
    )


def qpf_norm():
    # Slightly compress low-end values so 0.5" isn't dramatic
    return mcolors.PowerNorm(
        gamma=1.6,   # >1 compresses low values
        vmin=QPF_MIN_IN,
        vmax=QPF_VMAX_IN,
        clip=True
    )


# =========================
# GRIB READ
# =========================
def read_qpf_messages(grib_path: str):
    import pygrib
    grbs = pygrib.open(grib_path)

    out = []

    for i in range(1, grbs.messages + 1):
        m = grbs.message(i)

        disc = getattr(m, "discipline", None)
        cat = getattr(m, "parameterCategory", None)
        num = getattr(m, "parameterNumber", None)

        # GRIB2: Total precipitation = discipline 0, category 1, number 8
        if not (disc == 0 and cat == 1 and num == 8):
            continue

        data = m.values
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        data = np.where(data < 0, 0, data)
        data = np.where(data > 500, np.nan, data)

        lats, lons = m.latlons()
        end_utc = m.validDate.replace(tzinfo=timezone.utc)

        start_step = getattr(m, "startStep", None)
        end_step = getattr(m, "endStep", None)

        if isinstance(start_step, int) and isinstance(end_step, int):
            dur_hours = end_step - start_step
        else:
            dur_hours = 6

        start_utc = end_utc - timedelta(hours=dur_hours)

        out.append((data.astype("float32"), lats, lons, start_utc, end_utc))

    grbs.close()
    return out


# =========================
# DAILY AGGREGATION
# =========================
def distribute_to_local_days(grid, start_utc, end_utc):
    start_local = start_utc.astimezone(LOCAL_TZ)
    end_local = end_utc.astimezone(LOCAL_TZ)

    total = (end_local - start_local).total_seconds()
    if total <= 0:
        return {}

    results = {}
    cur = start_local

    while cur < end_local:
        day = cur.date()
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=LOCAL_TZ)
        day_end = day_start + timedelta(days=1)

        seg_start = max(cur, day_start)
        seg_end = min(end_local, day_end)

        overlap = (seg_end - seg_start).total_seconds()
        if overlap > 0:
            frac = overlap / total
            results[day] = grid * frac

        cur = day_end

    return results


def build_daily(all_msgs):
    if not all_msgs:
        return None, None, []

    _, lats0, lons0, _, _ = all_msgs[0]

    daily_sum = {}

    for grid, _, _, start_utc, end_utc in all_msgs:
        pieces = distribute_to_local_days(grid, start_utc, end_utc)
        for day, part in pieces.items():
            if day not in daily_sum:
                daily_sum[day] = np.zeros_like(part)
            daily_sum[day] += part

    daily = []
    for day in sorted(daily_sum.keys()):
        grid_in = mm_to_in(daily_sum[day])

        # mask below threshold
        grid_in = np.where(grid_in >= QPF_MIN_IN, grid_in, np.nan)

        # HARD CAP at 5": anything above becomes 5 so it uses the top color
        grid_in = np.where(grid_in > QPF_VMAX_IN, QPF_VMAX_IN, grid_in)

        if not np.isfinite(grid_in).any():
            continue

        daily.append({"date": day, "data": grid_in})

    return lats0, lons0, daily


# =========================
# RENDER
# =========================
def render(data, lats, lons, day, outfile):
    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax = plt.axes(projection=ccrs.LambertConformal(-96, 38))
    ax.set_extent(EXTENT, ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lons, lats, data,
        cmap=qpf_cmap(),
        norm=qpf_norm(),
        transform=ccrs.PlateCarree(),
        shading="auto"
    )

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6)

    ax.set_title(
        f"{TITLE}\n{format_day_label(day)} (Local Day Total)",
        fontsize=14, pad=12
    )

    fig.text(
        0.015, 0.02, WATERMARK_TEXT,
        ha="left", va="bottom",
        fontsize=20,
        fontweight="bold",
        color="white",
        alpha=0.7,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
)

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        fraction=0.045,
        aspect=40
    )
    cbar.set_label(UNITS_LABEL)
    cbar.set_ticks(COLORBAR_TICKS)
    cbar.set_ticklabels(COLORBAR_TICKLABELS)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUTDIR)

    downloaded = []
    for url, tmp in zip(NDFD_URLS, TMPFILES):
        if download_optional(url, tmp):
            downloaded.append(tmp)

    if not downloaded:
        raise RuntimeError("No QPF files downloaded.")

    all_msgs = []
    for f in downloaded:
        all_msgs.extend(read_qpf_messages(f))

    if not all_msgs:
        raise RuntimeError("No QPF messages found in GRIB.")

    lats, lons, daily = build_daily(all_msgs)
    daily = daily[:MAX_DAYS]

    for idx, d in enumerate(daily):
        outfile = PNG_PATTERN.format(idx=idx)
        render(d["data"], lats, lons, d["date"], outfile)
        print(f"Wrote {outfile}")

    first = PNG_PATTERN.format(idx=0)
    with open(first, "rb") as src, open(LATEST_OUTFILE, "wb") as dst:
        dst.write(src.read())

    print("Done.")


if __name__ == "__main__":
    main()
