# make_dewpoint_7day_ndfd.py
# ---------------------------------------------------------
# NWS NDFD 7-Day Daytime MAX Dewpoint (°F) – Humidity Focus
#
# - Downloads dewpoint GRIB2 from TWO NDFD periods:
#   VP.001-003 (short range) + VP.004-007 (extended)
# - Reads all dewpoint valid times
# - Converts times to LOCAL (America/New_York)
# - For each LOCAL DATE, computes MAX dewpoint during a daytime window
# - Renders up to 7 daily frames for your slider:
#     maps/ndfd_dew_f000.png ... f006.png
# - Writes:
#     maps/latest_dew_ndfd.png
#     maps/manifest_dew.json
# - Skip logic: if daily set unchanged, skip (unless FORCE_RENDER)
# ---------------------------------------------------------

import os
import json
from datetime import datetime, timezone, time
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
VAR_NAME = "dew"
TITLE = "Daytime Max Dewpoint (°F)"
UNITS_LABEL = "°F"

# Dewpoint GRIB2 buckets (CONUS)
NDFD_URLS = [
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.td.bin",
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.td.bin",
]

OUTDIR = "maps"
TMPFILES = [
    os.path.join(OUTDIR, "_ndfd_dew_001_003.grib2"),
    os.path.join(OUTDIR, "_ndfd_dew_004_007.grib2"),
]

STATEFILE = os.path.join(OUTDIR, "_last_ndfd_dew_daymax_7day.json")
MANIFEST = os.path.join(OUTDIR, "manifest_dew.json")
LATEST_OUTFILE = os.path.join(OUTDIR, "latest_dew_ndfd.png")
PNG_PATTERN = os.path.join(OUTDIR, "ndfd_dew_f{idx:03d}.png")

# Map extent (CONUS)
EXTENT = (-125, -66.5, 24, 50.5)

# Visual style
DPI = 150
WATERMARK_TEXT = "WxLaunch"

# Dewpoint display rules (same as your working map)
DEW_MIN_F = 35          # mask below -> white
DEW_MAX_F = 70          # 70+ -> green
STEP_F = 1

MAX_DAYS = 7
FORCE_RENDER = False

# Local TZ
LOCAL_TZ = ZoneInfo("America/New_York")

# Daytime window (LOCAL). Tune if you want.
DAY_START_LOCAL = time(10, 0)   # 10:00 AM
DAY_END_LOCAL   = time(19, 0)   # 7:00 PM (inclusive)


# =========================
# Helpers
# =========================
def ensure_dir(path_or_dir: str) -> None:
    d = path_or_dir
    if os.path.splitext(path_or_dir)[1]:
        d = os.path.dirname(path_or_dir)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def download(url: str, dest: str) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


def k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


def format_day_label(day_date):
    # e.g. Tue Feb 11
    # Windows-safe: %a %b %d works; remove leading zero in day
    return day_date.strftime("%a %b %d").replace(" 0", " ")


def dewpoint_cmap():
    """
    Your exact humidity-focus scale:
    35F  -> pale yellow
    40F  -> yellow
    50F  -> orange
    60F  -> reddish-orange
    65F  -> light green
    70F+ -> GREEN
    """
    anchors = [
        (35, "#fff7bc"),
        (40, "#fee391"),
        (45, "#fec44f"),
        (50, "#fe9929"),
        (55, "#ec7014"),
        (60, "#cc4c02"),
        (65, "#99d594"),
        (70, "#2ca25f"),
    ]

    t0, t1 = DEW_MIN_F, DEW_MAX_F
    xs = [(t - t0) / (t1 - t0) for t, _ in anchors]
    cs = [c for _, c in anchors]

    return mcolors.LinearSegmentedColormap.from_list(
        "dewpoint_humidity_focus",
        list(zip(xs, cs))
    )


def in_daytime_window(dt_local):
    # inclusive end
    t = dt_local.timetz().replace(tzinfo=None)
    return (t >= DAY_START_LOCAL) and (t <= DAY_END_LOCAL)


def read_dew_messages(grib_path: str):
    """
    Return list of tuples:
      (data_k, lats, lons, valid_utc)
    """
    import pygrib
    grbs = pygrib.open(grib_path)

    try:
        msgs = grbs.select(name="Dew point temperature")
    except Exception:
        msgs = []

    if not msgs:
        try:
            msgs = grbs.select(shortName="2d")
        except Exception:
            msgs = []

    # Fallback: read all messages
    if not msgs:
        msgs = [grbs.message(i) for i in range(1, grbs.messages + 1)]

    msgs = sorted(msgs, key=lambda m: m.validDate)

    out = []
    for m in msgs:
        data = m.values
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        # Kelvin sanity (prevents fill/garbage)
        data = np.where((data < 150) | (data > 330), np.nan, data)

        lats, lons = m.latlons()
        valid_utc = m.validDate.replace(tzinfo=timezone.utc)

        out.append((data, lats, lons, valid_utc))

    grbs.close()
    return out


def build_daily_daytime_max(all_msgs):
    """
    all_msgs: list of (data_k, lats, lons, valid_utc)
    Returns:
      lats, lons, daily_list
    where daily_list is list of dicts:
      {"date": date, "data_f": array, "source_valids": [utc_iso, ...]}
    """
    if not all_msgs:
        return None, None, []

    # Assume consistent grid across messages
    _, lats0, lons0, _ = all_msgs[0]

    buckets = {}  # date -> list[(data_f, valid_utc)]
    for data_k, _lats, _lons, valid_utc in all_msgs:
        dt_local = valid_utc.astimezone(LOCAL_TZ)

        if not in_daytime_window(dt_local):
            continue

        day = dt_local.date()

        data_f = k_to_f(data_k)

        # extra sanity in F
        data_f = np.where((data_f < -80) | (data_f > 120), np.nan, data_f)

        buckets.setdefault(day, []).append((data_f, valid_utc))

    daily = []
    for day in sorted(buckets.keys()):
        pairs = buckets[day]
        if not pairs:
            continue

        stack = np.stack([p[0] for p in pairs], axis=0)
        daymax = np.nanmax(stack, axis=0)

        # mask < 35F (your rule)
        daymax = np.where(daymax >= DEW_MIN_F, daymax, np.nan)

        # If nothing finite, skip
        if not np.isfinite(daymax).any():
            continue

        source_valids = [p[1].strftime("%Y-%m-%dT%H:%MZ") for p in pairs]

        daily.append({
            "date": day,
            "data_f": daymax,
            "source_valids": sorted(source_valids),
        })

    return lats0, lons0, daily


def render_daily_png(data_f, lats, lons, day_date, outfile: str):
    levels = np.arange(DEW_MIN_F, DEW_MAX_F + STEP_F, STEP_F)

    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax = plt.axes(projection=ccrs.LambertConformal(-96, 38))
    ax.set_facecolor("white")
    ax.set_extent(EXTENT, ccrs.PlateCarree())

    cf = ax.contourf(
        lons, lats, data_f,
        levels=levels,
        cmap=dewpoint_cmap(),
        transform=ccrs.PlateCarree(),
        antialiased=True,
        extend="max"
    )

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6)

    day_label = format_day_label(day_date)
    window_label = f"{DAY_START_LOCAL.strftime('%-I%p') if False else '10AM'}–{DAY_END_LOCAL.strftime('%-I%p') if False else '7PM'} Local"
    # (above "False" avoids %-I on Windows; we hardcode the label for now)

    ax.set_title(f"{TITLE}\n{day_label} (Daytime Max {DAY_START_LOCAL.strftime('%I:%M %p').replace(' 0',' ')}–{DAY_END_LOCAL.strftime('%I:%M %p').replace(' 0',' ')})",
                 fontsize=14, pad=12)

    fig.text(
        0.985, 0.02, WATERMARK_TEXT,
        ha="right", va="bottom",
        fontsize=20, fontweight="bold",
        color="white", alpha=0.7,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )

    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046, extend="max")
    cbar.set_label("°F")
    cbar.set_ticks([35, 40, 45, 50, 55, 60, 65, 70])

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    ensure_dir(OUTDIR)

    # Download both buckets
    for url, tmp in zip(NDFD_URLS, TMPFILES):
        download(url, tmp)

    # Read all messages from both
    all_msgs = []
    for tmp in TMPFILES:
        all_msgs.extend(read_dew_messages(tmp))

    # Sort and de-dup by valid time
    all_msgs = sorted(all_msgs, key=lambda t: t[3])
    seen = set()
    deduped = []
    for data_k, lats, lons, valid_utc in all_msgs:
        iso = valid_utc.strftime("%Y-%m-%dT%H:%MZ")
        if iso in seen:
            continue
        seen.add(iso)
        deduped.append((data_k, lats, lons, valid_utc))

    lats, lons, daily = build_daily_daytime_max(deduped)

    # keep up to 7 days
    daily = daily[:MAX_DAYS]

    if not daily:
        raise RuntimeError("No daytime dewpoint frames found (check daytime window / data availability).")

    # State for skip logic
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
        render_daily_png(d["data_f"], lats, lons, d["date"], outfile)
        print(f"Wrote day {idx}: {outfile}")

        frames.append({
            "idx": idx,
            "valid_utc": d["date"].isoformat(),  # day-based product
            "valid_local": f"{format_day_label(d['date'])} (Daytime Max)",
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
        "daytime_window_local": {
            "start": DAY_START_LOCAL.strftime("%H:%M"),
            "end": DAY_END_LOCAL.strftime("%H:%M"),
            "tz": "America/New_York",
        }
    }

    json.dump(manifest, open(MANIFEST, "w"))
    json.dump(new_state, open(STATEFILE, "w"))
    print(f"Wrote: {MANIFEST}")


if __name__ == "__main__":
    main()
