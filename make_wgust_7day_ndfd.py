# make_wgust_7day_ndfd.py
# ---------------------------------------------------------
# NWS NDFD 7-Day DAILY MAX WIND GUST (mph) – Local Calendar Day Max
#
# - Downloads Wind Gust GRIB2 from NDFD:
#     VP.001-003 (short range) + VP.004-007 (extended, if available)
# - Reads all gust valid times, converts to LOCAL (America/New_York)
# - For each LOCAL DATE, computes DAILY MAX gust (mph)
# - Renders up to 7 daily frames:
#     maps/ndfd_wgust_f000.png ... f006.png
# - Writes:
#     maps/latest_wgust_ndfd.png
#     maps/manifest_wgust.json
# - Skip logic via STATEFILE unless FORCE_RENDER=True
#
# Notes:
# - VP.004-007 may 404 for some variables on tgftp; handled gracefully.
# - Uses a WIND-style colormap (not rain-looking).
# ---------------------------------------------------------

import os
import json
import warnings
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =========================
# CONFIG
# =========================
VAR_NAME = "wgust"
TITLE = "Daily Max Wind Gust (mph)"
UNITS_LABEL = "mph"

NDFD_URLS = [
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.wgust.bin",
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.wgust.bin",  # may 404
]

OUTDIR = "maps"
TMPFILES = [
    os.path.join(OUTDIR, "_ndfd_wgust_001_003.grib2"),
    os.path.join(OUTDIR, "_ndfd_wgust_004_007.grib2"),
]

STATEFILE = os.path.join(OUTDIR, "_last_ndfd_wgust_daily_7day.json")
MANIFEST = os.path.join(OUTDIR, "manifest_wgust.json")
LATEST_OUTFILE = os.path.join(OUTDIR, "latest_wgust_ndfd.png")
PNG_PATTERN = os.path.join(OUTDIR, "ndfd_wgust_f{idx:03d}.png")

EXTENT = (-125, -66.5, 24, 50.5)

DPI = 150
WATERMARK_TEXT = "WxLaunch"

MAX_DAYS = 7
FORCE_RENDER = False

LOCAL_TZ = ZoneInfo("America/New_York")

# Gust display / scale
GUST_MIN_MPH = 10.0      # mask below this (keeps map clean)
GUST_VMAX_MPH = 70.0     # cap; 70+ is one color

# Colorbar ticks
COLORBAR_TICKS = [10, 20, 30, 40, 50, 60, 70]


# =========================
# Utilities
# =========================
def ensure_dir(path_or_dir: str) -> None:
    d = path_or_dir
    if os.path.splitext(path_or_dir)[1]:
        d = os.path.dirname(path_or_dir)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def download_optional(url: str, dest: str) -> bool:
    """
    Download URL to dest.
    - Returns True if downloaded
    - Returns False if missing (404) or failed; does NOT raise
    """
    try:
        r = requests.get(url, timeout=180)
        if r.status_code == 404:
            print(f"Missing on tgftp (404), skipping: {url}")
            # remove stale file if present
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except Exception:
                    pass
            return False
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"Download failed, skipping: {url}\n  -> {e}")
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except Exception:
                pass
        return False


def format_day_label(day_date):
    # Windows-safe: remove leading zero day
    return day_date.strftime("%a %b %d").replace(" 0", " ")


def ms_to_mph(x):
    return x * 2.2369362920544


# =========================
# Color
# =========================
def gust_cmap_wind():
    """
    Wind-style ramp (NOT rain):
    teal -> cyan -> blue -> indigo -> purple -> magenta -> red
    with a single cap color for 70+.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "wxlaunch_wgust_wind",
        [
            "#eafcff",  # very light cyan
            "#bff3ff",  # pale cyan
            "#7fe3ff",  # cyan
            "#39c6ff",  # bright cyan-blue
            "#1e88ff",  # blue
            "#1b4fff",  # deep blue
            "#3b2dbd",  # indigo
            "#6a00a8",  # purple
            "#b300b3",  # magenta
            "#ff2d55",  # hot pink-red
        ],
        N=256
    )
    cmap.set_over("#7a0000")   # 70+ mph cap = dark maroon
    cmap.set_under("#ffffff")  # masked low gusts = white
    return cmap


def gust_norm():
    return mcolors.Normalize(vmin=GUST_MIN_MPH, vmax=GUST_VMAX_MPH)


# =========================
# GRIB Read
# =========================
def read_wgust_messages(grib_path: str):
    """
    Return list of tuples:
      (grid_mph, lats, lons, valid_utc)
    """
    import pygrib
    grbs = pygrib.open(grib_path)

    msgs = []
    try:
        msgs = grbs.select(name="Wind speed (gust)")
    except Exception:
        msgs = []

    if not msgs:
        try:
            msgs = grbs.select(shortName="gust")
        except Exception:
            msgs = []

    # Fallback: all messages
    if not msgs:
        msgs = [grbs.message(i) for i in range(1, grbs.messages + 1)]

    # Sort by valid time
    msgs = sorted(msgs, key=lambda m: getattr(m, "validDate", datetime.min))

    out = []
    for m in msgs:
        data = m.values
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        # Gust sanity: m/s typical. Filter garbage.
        data = np.where((data < 0) | (data > 90), np.nan, data)  # 90 m/s ~ 200 mph
        grid_mph = ms_to_mph(data).astype("float32")

        lats, lons = m.latlons()
        valid_utc = m.validDate.replace(tzinfo=timezone.utc)

        out.append((grid_mph, lats, lons, valid_utc))

    grbs.close()
    return out


# =========================
# Daily aggregation (local day max)
# =========================
def build_daily_local_max(all_msgs):
    """
    all_msgs: list of (grid_mph, lats, lons, valid_utc)
    Returns: lats, lons, daily_list
      daily_list: [{"date": date, "data_mph": array, "source_valids": [...]}, ...]
    """
    if not all_msgs:
        return None, None, []

    _, lats0, lons0, _ = all_msgs[0]

    buckets = {}  # date -> list[(grid_mph, valid_iso)]
    for grid_mph, _lats, _lons, valid_utc in all_msgs:
        dt_local = valid_utc.astimezone(LOCAL_TZ)
        day = dt_local.date()

        # mask low gusts early so they don't influence max
        grid = np.where(grid_mph >= GUST_MIN_MPH, grid_mph, np.nan)

        valid_iso = valid_utc.strftime("%Y-%m-%dT%H:%MZ")
        buckets.setdefault(day, []).append((grid, valid_iso))

    daily = []
    for day in sorted(buckets.keys()):
        pairs = buckets[day]
        if not pairs:
            continue

        stack = np.stack([p[0] for p in pairs], axis=0)

        # Suppress "All-NaN slice" warnings at pixels with no data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            daymax = np.nanmax(stack, axis=0).astype("float32")

        # If nothing finite, skip this day
        if not np.isfinite(daymax).any():
            continue

        # Cap at vmax so 70+ becomes the same color bin
        daymax = np.where(daymax > GUST_VMAX_MPH, GUST_VMAX_MPH + 0.01, daymax)

        daily.append({
            "date": day,
            "data_mph": daymax,
            "source_valids": sorted({p[1] for p in pairs}),
        })

    return lats0, lons0, daily


# =========================
# Render
# =========================
def render_daily_png(data_mph, lats, lons, day_date, outfile: str):
    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax = plt.axes(projection=ccrs.LambertConformal(-96, 38))
    ax.set_facecolor("white")
    ax.set_extent(EXTENT, ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lons, lats, data_mph,
        cmap=gust_cmap_wind(),
        norm=gust_norm(),
        transform=ccrs.PlateCarree(),
        shading="auto"
    )

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6)

    ax.set_title(
        f"{TITLE}\n{format_day_label(day_date)} (Local Day Max)",
        fontsize=14, pad=12
    )

    # Watermark bottom-left so it never blocks right-side ticks
    fig.text(
        0.015, 0.02, WATERMARK_TEXT,
        ha="left", va="bottom",
        fontsize=20, fontweight="bold",
        color="white", alpha=0.7,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )

    cbar = plt.colorbar(
        mesh, ax=ax, orientation="horizontal",
        pad=0.08, fraction=0.045, aspect=40, extend="max"
    )
    cbar.set_label(UNITS_LABEL)
    cbar.set_ticks(COLORBAR_TICKS)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    ensure_dir(OUTDIR)

    # Download buckets that exist
    downloaded = []
    for url, tmp in zip(NDFD_URLS, TMPFILES):
        if download_optional(url, tmp):
            downloaded.append(tmp)

    if not downloaded:
        raise RuntimeError("No Wind Gust GRIB2 files could be downloaded (both buckets missing or failed).")

    # Read all messages
    all_msgs = []
    for tmp in downloaded:
        all_msgs.extend(read_wgust_messages(tmp))

    if not all_msgs:
        raise RuntimeError("No wind gust messages found in GRIB.")

    # De-dup by valid time
    all_msgs = sorted(all_msgs, key=lambda t: t[3])
    seen = set()
    deduped = []
    for grid_mph, lats, lons, valid_utc in all_msgs:
        iso = valid_utc.strftime("%Y-%m-%dT%H:%MZ")
        if iso in seen:
            continue
        seen.add(iso)
        deduped.append((grid_mph, lats, lons, valid_utc))

    lats, lons, daily = build_daily_local_max(deduped)
    daily = daily[:MAX_DAYS]

    if not daily:
        raise RuntimeError("No daily gust frames found after aggregation.")

    # Skip logic state
    new_state = {
        "days": [d["date"].isoformat() for d in daily],
        "source_valids_flat": [v for d in daily for v in d["source_valids"]],
    }

    if (not FORCE_RENDER) and os.path.exists(STATEFILE):
        try:
            old = json.load(open(STATEFILE))
            if old.get("days") == new_state["days"] and old.get("source_valids_flat") == new_state["source_valids_flat"]:
                print("Same daily forecast set; skipping render.")
                return
        except Exception:
            pass

    frames = []
    for idx, d in enumerate(daily):
        outfile = PNG_PATTERN.format(idx=idx)
        render_daily_png(d["data_mph"], lats, lons, d["date"], outfile)
        print(f"Wrote day {idx}: {outfile}")

        frames.append({
            "idx": idx,
            "valid_utc": d["date"].isoformat(),  # day-based product
            "valid_local": f"{format_day_label(d['date'])} (Daily Max Gust)",
            "file": os.path.basename(outfile),
        })

    # Latest = day 0
    first = os.path.join(OUTDIR, frames[0]["file"])
    with open(first, "rb") as s, open(LATEST_OUTFILE, "wb") as out:
        out.write(s.read())
    print(f"Wrote: {LATEST_OUTFILE}")

    manifest = {
        "var": VAR_NAME,
        "title": TITLE,
        "units": UNITS_LABEL,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "frame_count": len(frames),
        "frames": frames,
        "aggregation": {
            "type": "local_day_max",
            "tz": "America/New_York",
            "note": "Daily max computed over all valid times within each local calendar day."
        },
        "scale": {
            "vmin_mph": GUST_MIN_MPH,
            "vmax_mph": GUST_VMAX_MPH,
            "cap_note": f"{GUST_VMAX_MPH:.0f}+ mph shown as one color (capped)."
        }
    }

    json.dump(manifest, open(MANIFEST, "w"))
    json.dump(new_state, open(STATEFILE, "w"))
    print(f"Wrote: {MANIFEST}")


if __name__ == "__main__":
    main()
