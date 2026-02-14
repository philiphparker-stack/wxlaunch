# make_snow_7day_ndfd.py
# ---------------------------------------------------------
# NWS NDFD DAILY SNOW (Liquid-Equivalent, inches) – Local Day Totals
#
# Robust for ds.snow.bin where naming/metadata varies.
#
# Fix vs your error:
# - If NO day has measurable snow >= SNOW_MIN_IN, we STILL produce frames.
#   (Blank map + "No measurable snow" label instead of crashing.)
#
# Outputs:
#   maps/ndfd_snow_f000.png ... f006.png
#   maps/latest_snow_ndfd.png
#   maps/manifest_snow.json
#   maps/_last_ndfd_snow_daily_7day.json
#
# Notes:
# - This is LIQUID-EQUIVALENT (LES/LE), not snow depth.
# - VP.004-007 may 404; handled gracefully.
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
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.snow.bin",  # may 404
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
FORCE_RENDER = False

LOCAL_TZ = ZoneInfo("America/New_York")

# Display / scaling (inches liquid-equivalent)
SNOW_MIN_IN = 0.01

# cap for color scale; >= this uses top color
SNOW_VMAX_IN = 2.0

# "less dramatic" scaling like your QPF (gamma > 1 compresses low end)
GAMMA = 1.6

COLORBAR_TICKS = [0.1, 0.25, 0.5, 1.0, 2.0]
COLORBAR_TICKLABELS = ["0.1", "0.25", "0.5", "1", "2+"]

# If you ever need to diagnose:
AUTO_DEBUG_IF_EMPTY = False  # set True to print first 25 message metadata


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


def format_day_label(day_date):
    return day_date.strftime("%a %b %d").replace(" 0", " ")


def mm_to_in(mm):
    return mm / 25.4


def m_to_in(m):
    return m * 39.3700787402


# =========================
# COLOR
# =========================
def snow_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "wxlaunch_snow_le",
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


def snow_norm():
    return mcolors.PowerNorm(gamma=GAMMA, vmin=SNOW_MIN_IN, vmax=SNOW_VMAX_IN, clip=True)


# =========================
# GRIB READ (STRUCTURAL)
# =========================
def message_to_inches_le(m, data):
    units = (getattr(m, "units", "") or "").strip().lower()

    if units in {"mm", "kg m-2", "kg/m^2", "kg m**-2"}:
        return mm_to_in(data)

    if units == "m":
        return m_to_in(data)

    # heuristic fallback
    return mm_to_in(data)


def is_depth_like(m):
    name = (getattr(m, "name", "") or "").lower()
    short = (getattr(m, "shortName", "") or "").lower()
    if "depth" in name:
        return True
    if short in {"sd", "snowd", "snod"}:
        return True
    return False


def looks_like_accum_interval(m):
    ss = getattr(m, "startStep", None)
    es = getattr(m, "endStep", None)
    return isinstance(ss, int) and isinstance(es, int) and es > ss


def read_snow_messages(grib_path: str):
    """
    Return list of tuples:
      (grid_in, lats, lons, start_utc, end_utc, end_iso)
    """
    import pygrib
    grbs = pygrib.open(grib_path)

    out = []

    if AUTO_DEBUG_IF_EMPTY:
        print(f"\nOpened {grib_path} with {grbs.messages} messages")

    for i in range(1, grbs.messages + 1):
        m = grbs.message(i)

        if not hasattr(m, "validDate"):
            continue
        if is_depth_like(m):
            continue
        if not looks_like_accum_interval(m):
            continue

        data = m.values
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        data = np.where(data < 0, 0.0, data)
        data = np.where(data > 10000, np.nan, data)

        if not np.isfinite(data).any():
            continue

        grid_in = message_to_inches_le(m, data).astype("float32")

        lats, lons = m.latlons()
        end_utc = m.validDate.replace(tzinfo=timezone.utc)

        ss = getattr(m, "startStep", None)
        es = getattr(m, "endStep", None)
        dur_hours = es - ss

        start_utc = end_utc - timedelta(hours=dur_hours)
        end_iso = end_utc.strftime("%Y-%m-%dT%H:%MZ")

        out.append((grid_in, lats, lons, start_utc, end_utc, end_iso))

        if AUTO_DEBUG_IF_EMPTY and len(out) <= 5:
            print(
                f"match: name={getattr(m,'name',None)} short={getattr(m,'shortName',None)} "
                f"units={getattr(m,'units',None)} startStep={ss} endStep={es}"
            )

    if AUTO_DEBUG_IF_EMPTY and not out:
        print("\n--- AUTO DEBUG: first 25 messages ---")
        for i in range(1, min(grbs.messages, 25) + 1):
            m = grbs.message(i)
            print(
                f"{i:02d}: name={getattr(m,'name',None)} | short={getattr(m,'shortName',None)} | "
                f"units={getattr(m,'units',None)} | startStep={getattr(m,'startStep',None)} | "
                f"endStep={getattr(m,'endStep',None)}"
            )
        print("--- END DEBUG ---\n")

    grbs.close()
    return out


# =========================
# DAILY AGGREGATION
# =========================
def distribute_to_local_days(grid_in, start_utc, end_utc):
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
            results[day] = grid_in * frac

        cur = day_end

    return results


def build_daily(all_msgs):
    if not all_msgs:
        return None, None, []

    _, lats0, lons0, _, _, _ = all_msgs[0]

    daily_sum = {}
    daily_sources = {}

    for grid_in, _, _, start_utc, end_utc, end_iso in all_msgs:
        pieces = distribute_to_local_days(grid_in, start_utc, end_utc)
        for day, part in pieces.items():
            if day not in daily_sum:
                daily_sum[day] = np.zeros_like(part, dtype="float32")
                daily_sources[day] = []
            daily_sum[day] += part
            daily_sources[day].append(end_iso)

    daily = []
    for day in sorted(daily_sum.keys()):
        grid = daily_sum[day]

        # determine if anything is >= measurable threshold (before masking)
        max_val = np.nanmax(grid) if np.isfinite(grid).any() else 0.0
        has_measurable = bool(max_val >= SNOW_MIN_IN)

        # mask tiny
        data = np.where(grid >= SNOW_MIN_IN, grid, np.nan)

        # cap for display
        data = np.where(data > SNOW_VMAX_IN, SNOW_VMAX_IN, data)

        # IMPORTANT FIX: DO NOT SKIP if all-NaN; still produce a frame.
        daily.append({
            "date": day,
            "data": data,
            "source_valids": sorted(set(daily_sources.get(day, []))),
            "has_measurable": has_measurable,
        })

    return lats0, lons0, daily


# =========================
# RENDER
# =========================
def render(data_in, lats, lons, day, outfile, has_measurable: bool):
    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax = plt.axes(projection=ccrs.LambertConformal(-96, 38))
    ax.set_extent(EXTENT, ccrs.PlateCarree())

    # draw the field (even if all-NaN; will just appear blank)
    mesh = ax.pcolormesh(
        lons, lats, data_in,
        cmap=snow_cmap(),
        norm=snow_norm(),
        transform=ccrs.PlateCarree(),
        shading="auto"
    )

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6)

    ax.set_title(
        f"{TITLE}\n{format_day_label(day)} (Local Day Total) — Liquid-Equivalent",
        fontsize=14, pad=12
    )

    # Optional callout if nothing meets threshold
    if not has_measurable:
        ax.text(
            0.5, 0.06,
            f"No measurable snow (LE < {SNOW_MIN_IN:.2f}\")",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12,
            color="black",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

    # watermark bottom-left
    fig.text(
        0.015, 0.02, WATERMARK_TEXT,
        ha="left", va="bottom",
        fontsize=20, fontweight="bold",
        color="white", alpha=0.7,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )

    # Colorbar: use a ScalarMappable so it ALWAYS draws cleanly
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
        raise RuntimeError("No SNOW files downloaded (both buckets missing or failed).")

    all_msgs = []
    for f in downloaded:
        all_msgs.extend(read_snow_messages(f))

    if not all_msgs:
        raise RuntimeError("No accumulation-interval messages found in ds.snow.bin.")

    # De-dup by end time + duration
    deduped = []
    seen = set()
    for grid_in, lats, lons, start_utc, end_utc, end_iso in sorted(all_msgs, key=lambda t: t[4]):
        key = (end_iso, int((end_utc - start_utc).total_seconds()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((grid_in, lats, lons, start_utc, end_utc, end_iso))

    lats, lons, daily = build_daily(deduped)

    # Keep up to 7 days, but do NOT require measurable snow
    daily = daily[:MAX_DAYS]
    if not daily:
        raise RuntimeError("No daily frames produced (unexpected).")

    # Skip logic
    new_state = {
        "days": [d["date"].isoformat() for d in daily],
        "source_valids_flat": [v for d in daily for v in d["source_valids"]],
    }

    if (not FORCE_RENDER) and os.path.exists(STATEFILE):
        try:
            old = json.load(open(STATEFILE, "r"))
            if old.get("days") == new_state["days"] and old.get("source_valids_flat") == new_state["source_valids_flat"]:
                print("Same daily forecast set; skipping render.")
                return
        except Exception:
            pass

    frames = []
    for idx, d in enumerate(daily):
        outfile = PNG_PATTERN.format(idx=idx)
        render(d["data"], lats, lons, d["date"], outfile, d["has_measurable"])
        print(f"Wrote {outfile}")

        frames.append({
            "idx": idx,
            "valid_utc": d["date"].isoformat(),
            "valid_local": f"{format_day_label(d['date'])} (Daily Total LE)",
            "file": os.path.basename(outfile),
        })

    # Latest = day 0
    first = os.path.join(OUTDIR, frames[0]["file"])
    with open(first, "rb") as src, open(LATEST_OUTFILE, "wb") as dst:
        dst.write(src.read())
    print(f"Wrote {LATEST_OUTFILE}")

    manifest = {
        "var": VAR_NAME,
        "title": TITLE,
        "units": UNITS_LABEL,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "frame_count": len(frames),
        "frames": frames,
        "aggregation": {
            "type": "local_day_total",
            "tz": "America/New_York",
            "note": "Snow shown is LIQUID-EQUIVALENT (LES/LE). Accumulation intervals are split across local calendar days by overlap."
        },
        "cap_note": f"Color scale capped at {SNOW_VMAX_IN:.2f}\" LE (values above shown as top color).",
        "style_note": f"PowerNorm gamma={GAMMA} to reduce low-end drama.",
        "threshold_note": f"Values below {SNOW_MIN_IN:.2f}\" LE are treated as 'no measurable snow' and rendered blank."
    }

    json.dump(manifest, open(MANIFEST, "w"))
    json.dump(new_state, open(STATEFILE, "w"))
    print(f"Wrote {MANIFEST}")
    print("Done.")


if __name__ == "__main__":
    main()
