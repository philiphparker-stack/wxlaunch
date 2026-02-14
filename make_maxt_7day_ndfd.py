# make_maxt_7day_ndfd.py
# ---------------------------------------------------------
# NWS NDFD 7-Day (or max available) Maximum Temperature (°F)
# - Downloads NDFD ds.maxt.bin (CONUS)
# - Reads ALL MaxT messages, sorted by valid time
# - Renders up to 7 frames:
#     maps/ndfd_maxt_f000.png, f001.png, ...
# - Writes:
#     maps/latest_maxt_ndfd.png
#     maps/manifest_maxt.json  (for slider + labels)
# - Skip logic: if valid times unchanged, skip render (unless FORCE_RENDER)
# ---------------------------------------------------------

import os
import json
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
VAR_NAME = "maxt"
TITLE = "Max Temperature (°F)"
UNITS_LABEL = "°F"

# NDFD MaxT (CONUS)
NDFD_URL = (
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/"
    "AR.conus/VP.001-003/ds.maxt.bin"
)

OUTDIR = "maps"
TMPFILE = os.path.join(OUTDIR, "_ndfd_maxt.grib2")
STATEFILE = os.path.join(OUTDIR, "_last_ndfd_maxt_7day.json")
MANIFEST = os.path.join(OUTDIR, "manifest_maxt.json")
LATEST_OUTFILE = os.path.join(OUTDIR, "latest_maxt_ndfd.png")
PNG_PATTERN = os.path.join(OUTDIR, "ndfd_maxt_f{idx:03d}.png")

# Map extent (CONUS)
EXTENT = (-125, -66.5, 24, 50.5)

# Visual style
DPI = 150
WATERMARK_TEXT = "WxLaunch"

# Tight gradients
TEMP_MIN_F = -10
TEMP_MAX_F = 110
STEP_F = 1

# Numbers overlay
DRAW_NUMBERS = True
LABEL_DENSITY_ACROSS = 14  # smaller = fewer numbers

# How many frames to render (max). If NDFD provides less, we render less.
MAX_FRAMES = 7

# Set True to force re-render even if valid times unchanged
FORCE_RENDER = False

# Local timezone for display (auto handles EST/EDT)
LOCAL_TZ = ZoneInfo("America/New_York")


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


def format_local_time(dt_utc):
    dt_local = dt_utc.astimezone(LOCAL_TZ)
    s = dt_local.strftime("%a %b %d, %I:%M %p %Z")
    return s.replace(" 0", " ")


def pivotalish_temp_cmap():
    anchors = [
        (-10, "#5b2c83"),
        (0, "#6f3fb1"),
        (10, "#3b5bd6"),
        (20, "#2f84e0"),
        (30, "#23b2c8"),
        (40, "#33c27a"),
        (50, "#a7d96a"),
        (60, "#f2e85c"),
        (70, "#f7b84a"),
        (80, "#f07a3a"),
        (90, "#d83b2d"),
        (100, "#b2182b"),
        (110, "#7a0177"),
    ]

    t0, t1 = TEMP_MIN_F, TEMP_MAX_F
    xs = [max(0, min(1, (t - t0) / (t1 - t0))) for t, _ in anchors]
    cs = [c for _, c in anchors]
    return mcolors.LinearSegmentedColormap.from_list("pivotalish_temp", list(zip(xs, cs)))


def read_maxt_messages(grib_path: str):
    """
    Returns a list of extracted tuples:
      (data_k, lats, lons, valid_utc)
    Sorted by valid time, unique valid times only.
    """
    import pygrib

    grbs = pygrib.open(grib_path)

    # Prefer full name; fallback to shortName
    try:
        msgs = grbs.select(name="Maximum temperature")
    except Exception:
        msgs = []

    if not msgs:
        try:
            msgs = grbs.select(shortName="mx2t")
        except Exception:
            msgs = []

    if not msgs:
        grbs.close()
        raise RuntimeError("No Maximum temperature (maxt) messages found in GRIB")

    msgs = sorted(msgs, key=lambda m: m.validDate)

    extracted = []
    seen = set()
    for m in msgs:
        vd = m.validDate
        if vd in seen:
            continue
        seen.add(vd)

        data_k = m.values
        if isinstance(data_k, np.ma.MaskedArray):
            data_k = data_k.filled(np.nan)

        lats, lons = m.latlons()
        valid_utc = m.validDate.replace(tzinfo=timezone.utc)
        extracted.append((data_k, lats, lons, valid_utc))

    grbs.close()
    return extracted


def render_maxt_png(data_k, lats, lons, valid_utc, outfile: str):
    data_f = k_to_f(data_k)

    # Sanity filter (optional but safe)
    data_f = np.where((data_f < -80) | (data_f > 140), np.nan, data_f)

    levels = np.arange(TEMP_MIN_F, TEMP_MAX_F + STEP_F, STEP_F)
    cmap = pivotalish_temp_cmap()
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor("white")

    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96, central_latitude=38))
    ax.set_facecolor("white")
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Fill
    cf = ax.contourf(
        lons, lats, data_f,
        levels=levels,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        antialiased=True,
    )

    # Boundaries
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6, edgecolor="black")

    # Isotherms every 10F
    iso_levels = np.arange(TEMP_MIN_F, TEMP_MAX_F + 10, 10)
    ax.contour(
        lons, lats, data_f,
        levels=iso_levels,
        colors="k",
        linewidths=0.35,
        alpha=0.35,
        transform=ccrs.PlateCarree(),
    )

    # Numbers overlay
    if DRAW_NUMBERS:
        ny, nx = data_f.shape
        step_x = max(1, nx // LABEL_DENSITY_ACROSS)
        step_y = max(1, int(step_x * (ny / nx)))

        for j in range(0, ny, step_y):
            for i in range(0, nx, step_x):
                v = data_f[j, i]
                if np.isnan(v):
                    continue

                lat = lats[j, i]
                lon = lons[j, i]
                if not (EXTENT[0] <= lon <= EXTENT[1] and EXTENT[2] <= lat <= EXTENT[3]):
                    continue

                txt = ax.text(
                    lon, lat, f"{int(round(float(v)))}",
                    transform=ccrs.PlateCarree(),
                    fontsize=9, fontweight="bold",
                    color="black",
                    ha="center", va="center",
                    zorder=5,
                )
                txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white", alpha=0.85)])

    # Local time display
    valid_local = format_local_time(valid_utc)
    ax.set_title(f"{TITLE}\nValid: {valid_local}", fontsize=14, pad=12)

    # Watermark
    fig.text(
        0.985, 0.02, WATERMARK_TEXT,
        ha="right", va="bottom",
        fontsize=20, fontweight="bold",
        color="white", alpha=0.7,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046)
    cbar.set_label(UNITS_LABEL)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(OUTDIR)

    # Download latest file
    download(NDFD_URL, TMPFILE)

    all_msgs = read_maxt_messages(TMPFILE)
    msgs = all_msgs[:MAX_FRAMES]  # render up to 7, or fewer if unavailable

    if not msgs:
        raise RuntimeError("No MaxT frames found after filtering")

    # Build new state based on valid times (UTC stored for consistency)
    frames = []
    for idx, (_data_k, _lats, _lons, valid_utc) in enumerate(msgs):
        frames.append({
            "idx": idx,
            "valid_utc": valid_utc.strftime("%Y-%m-%dT%H:%MZ"),
            "valid_local": format_local_time(valid_utc),
            "file": os.path.basename(PNG_PATTERN.format(idx=idx))
        })

    new_state = {"valids": [f["valid_utc"] for f in frames]}

    # Skip logic
    if (not FORCE_RENDER) and os.path.exists(STATEFILE):
        try:
            old = json.load(open(STATEFILE))
            if old.get("valids") == new_state["valids"]:
                print("Same forecast set; skipping render.")
                return
        except Exception:
            pass

    # Render all frames
    for idx, (data_k, lats, lons, valid_utc) in enumerate(msgs):
        outfile = PNG_PATTERN.format(idx=idx)
        render_maxt_png(data_k, lats, lons, valid_utc, outfile)
        print(f"Wrote frame {idx}: {outfile}")

    # Write "latest" as frame 0
    first = os.path.join(OUTDIR, frames[0]["file"])
    with open(first, "rb") as s, open(LATEST_OUTFILE, "wb") as d:
        d.write(s.read())
    print(f"Wrote: {LATEST_OUTFILE}")

    # Manifest for slider
    manifest = {
        "var": VAR_NAME,
        "title": TITLE,
        "units": UNITS_LABEL,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "frame_count": len(frames),
        "frames": frames
    }
    json.dump(manifest, open(MANIFEST, "w"))
    json.dump(new_state, open(STATEFILE, "w"))
    print(f"Wrote: {MANIFEST}")


if __name__ == "__main__":
    main()
