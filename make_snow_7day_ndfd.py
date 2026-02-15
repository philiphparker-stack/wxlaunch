# make_snow_7day_ndfd.py
# ---------------------------------------------------------
# NWS NDFD DAILY SNOW (Liquid-Equivalent, inches) – Local Day Totals
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
from matplotlib.cm import ScalarMappable

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =========================
# CONFIG
# =========================
VAR_NAME = "snow"
TITLE = "Daily Total Snow (Liquid-Equivalent)"
UNITS_LABEL = "in (LE)"

NDFD_URLS = [
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.snow.bin",
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.snow.bin",
]

OUTDIR = "maps"
TMPFILES = [
    os.path.join(OUTDIR, "_ndfd_snow_001_003.grib2"),
    os.path.join(OUTDIR, "_ndfd_snow_004_007.grib2"),
]

STATEFILE = os.path.join(OUTDIR, "_last_ndfd_snow_daily_7day.json")
MANIFEST = os.path.join(OUTDIR, "manifest_snow.json")
LATEST_OUTFILE = os.path.join(OUTDIR, "latest_snow_ndfd.png")
PNG_PATTERN = os.path.join(OUTDIR, "ndfd_snow_f{idx:03d}.png")

EXTENT = (-125, -66.5, 24, 50.5)

DPI = 150
WATERMARK_TEXT = "WxLaunch"

MAX_DAYS = 7
LOCAL_TZ = ZoneInfo("America/New_York")

SNOW_MIN_IN = 0.01
SNOW_VMAX_IN = 2.0
GAMMA = 1.6

COLORBAR_TICKS = [0.1, 0.25, 0.5, 1.0, 2.0]
COLORBAR_TICKLABELS = ["0.1", "0.25", "0.5", "1", "2+"]


# =========================
# UTILITIES
# =========================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def download_optional(url, dest):
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


def format_day_label(d):
    return d.strftime("%a %b %d").replace(" 0", " ")


def mm_to_in(mm):
    return mm / 25.4


# =========================
# COLOR
# =========================
def snow_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "wxlaunch_snow",
        [
            "#e8f4ff", "#b7e3ff", "#7fe6a2",
            "#2ecc71", "#f1c40f", "#f39c12",
            "#e67e22", "#e74c3c", "#9b59b6", "#6f2dbd",
        ],
        N=256
    )


def snow_norm():
    return mcolors.PowerNorm(
        gamma=GAMMA,
        vmin=SNOW_MIN_IN,
        vmax=SNOW_VMAX_IN,
        clip=True
    )


# =========================
# GRIB READ
# =========================
def read_snow_messages(grib_path):
    import pygrib
    grbs = pygrib.open(grib_path)
    out = []

    for i in range(1, grbs.messages + 1):
        m = grbs.message(i)

        if not hasattr(m, "validDate"):
            continue

        ss = getattr(m, "startStep", None)
        es = getattr(m, "endStep", None)
        if not isinstance(ss, int) or not isinstance(es, int) or es <= ss:
            continue

        data = m.values
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        data = np.where(data < 0, 0, data)
        data = mm_to_in(data)

        lats, lons = m.latlons()

        end_utc = m.validDate.replace(tzinfo=timezone.utc)
        start_utc = end_utc - timedelta(hours=(es - ss))

        out.append((data.astype("float32"), lats, lons, start_utc, end_utc))

    grbs.close()
    return out


# =========================
# DAILY AGGREGATION
# =========================
def distribute_to_local_days(grid, start_utc, end_utc):
    if end_utc <= start_utc:
        return {}

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
        grid = daily_sum[day]

        has_measurable = bool(np.nanmax(grid) >= SNOW_MIN_IN) if np.isfinite(grid).any() else False
        data = np.where(grid >= SNOW_MIN_IN, grid, np.nan)
        data = np.where(data > SNOW_VMAX_IN, SNOW_VMAX_IN, data)

        daily.append({
            "date": day,
            "data": data,
            "has_measurable": has_measurable
        })

    return lats0, lons0, daily


# =========================
# RENDER
# =========================
def render(data, lats, lons, day, outfile, has_measurable):
    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax = plt.axes(projection=ccrs.LambertConformal(-96, 38))
    ax.set_extent(EXTENT, ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lons, lats, data,
        cmap=snow_cmap(),
        norm=snow_norm(),
        transform=ccrs.PlateCarree(),
        shading="auto"
    )

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6)

    ax.set_title(f"{TITLE}\n{format_day_label(day)} (Local Day Total)", fontsize=14)

    if not has_measurable:
        ax.text(
            0.5, 0.05,
            f"No measurable snow (< {SNOW_MIN_IN:.2f}\")",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

    sm = ScalarMappable(norm=snow_norm(), cmap=snow_cmap())
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        fraction=0.045,
        aspect=40
    )
    cbar.set_label(UNITS_LABEL)
    cbar.set_ticks(COLORBAR_TICKS)
    cbar.set_ticklabels(COLORBAR_TICKLABELS)

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
        raise RuntimeError("No snow files downloaded.")

    all_msgs = []
    for f in downloaded:
        all_msgs.extend(read_snow_messages(f))

    lats, lons, daily = build_daily(all_msgs)
    daily = daily[:MAX_DAYS]

    if not daily:
        raise RuntimeError("No daily snow frames produced.")

    # Render frames
    for idx, d in enumerate(daily):
        outfile = PNG_PATTERN.format(idx=idx)
        render(d["data"], lats, lons, d["date"], outfile, d["has_measurable"])
        print(f"Wrote {outfile}")

    # Build manifest
    frames = []
    for idx, d in enumerate(daily):
        frames.append({
            "file": f"ndfd_snow_f{idx:03d}.png",
            "valid_local": format_day_label(d["date"])
        })

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    manifest_obj = {
        "title": TITLE,
        "generated_at_utc": stamp,
        "frames": frames
    }

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)

    print(f"Wrote {MANIFEST}")

    # Latest image
    first = PNG_PATTERN.format(idx=0)
    with open(first, "rb") as src, open(LATEST_OUTFILE, "wb") as dst:
        dst.write(src.read())

    print(f"Wrote {LATEST_OUTFILE}")
    print("Done.")


if __name__ == "__main__":
    main()
