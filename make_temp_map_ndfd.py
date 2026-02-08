# make_temp_map_ndfd.py
import os
import json
from datetime import datetime, timezone

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ----------------------------
# SETTINGS YOU CARE ABOUT
# ----------------------------
OUTFILE = os.path.join("maps", "latest_maxt_ndfd.png")
STATEFILE = os.path.join("maps", "_last_ndfd_maxt.json")

# NDFD maxt (CONUS) day 1-3
NDFD_URL = (
    "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/"
    "AR.conus/VP.001-003/ds.maxt.bin"
)

# Map extent (CONUS)
EXTENT = (-125, -66.5, 24, 50.5)

# Visual style
DPI = 150
TITLE = "Daytime Max Temp (°F)"
WATERMARK_TEXT = "WxLaunch"

# "Tight gradients" => smaller step = tighter
TEMP_MIN_F = -10
TEMP_MAX_F = 110
STEP_F = 1  # 1 = extra smooth, 2 = a bit faster/lighter

# Overlay numbers
DRAW_NUMBERS = True
LABEL_DENSITY_ACROSS = 14  # smaller = fewer numbers (14 is nice)


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def download(url: str, dest: str) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


def k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


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


def load_grib_with_pygrib(grib_path: str):
    import pygrib

    grbs = pygrib.open(grib_path)

    msgs = grbs.select(name="Maximum temperature")
    if not msgs:
        # fallback
        try:
            msgs = grbs.select(shortName="mx2t")
        except Exception:
            msgs = []

    if not msgs:
        grbs.close()
        raise RuntimeError("No 'Maximum temperature' fields found in GRIB.")

    now = datetime.now(timezone.utc)
    future = [m for m in msgs if m.validDate.replace(tzinfo=timezone.utc) >= now]
    chosen = min(future, key=lambda m: m.validDate) if future else msgs[0]

    data_k = chosen.values
    lats, lons = chosen.latlons()
    valid = chosen.validDate.replace(tzinfo=timezone.utc)
    grbs.close()
    return data_k, lats, lons, valid


def read_ndfd_maxt(grib_path: str):
    # We're using pygrib in your environment (works great on conda-forge)
    return load_grib_with_pygrib(grib_path)


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(OUTFILE)
    tmp_grib = os.path.join("maps", "_ndfd_maxt.grib2")

    # Download latest file
    download(NDFD_URL, tmp_grib)

    # Read grib + choose next valid
    data_k, lats, lons, valid = read_ndfd_maxt(tmp_grib)

    # Skip render if valid time unchanged
    last = None
    if os.path.exists(STATEFILE):
        try:
            with open(STATEFILE, "r") as f:
                last = json.load(f).get("valid_utc")
        except Exception:
            last = None

    valid_iso = valid.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    if last == valid_iso and os.path.exists(OUTFILE):
        print(f"Same valid time ({valid_iso}); skipping render.")
        return

    data_f = k_to_f(data_k)

    # Tight gradient levels
    levels = np.arange(TEMP_MIN_F, TEMP_MAX_F + STEP_F, STEP_F)

    cmap = pivotalish_temp_cmap()
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    ax = plt.axes(
        projection=ccrs.LambertConformal(central_longitude=-96, central_latitude=38)
    )
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Fill
    cf = ax.contourf(
        lons,
        lats,
        data_f,
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

    # Isotherms (every 10F)
    iso_levels = np.arange(TEMP_MIN_F, TEMP_MAX_F + 10, 10)
    ax.contour(
        lons,
        lats,
        data_f,
        levels=iso_levels,
        colors="k",
        linewidths=0.35,
        alpha=0.35,
        transform=ccrs.PlateCarree(),
    )

    # Numbers overlay (subsample grid)
    if DRAW_NUMBERS:
        ny, nx = data_f.shape
        step_x = max(1, nx // LABEL_DENSITY_ACROSS)
        step_y = max(1, int(step_x * (ny / nx)))

        for j in range(0, ny, step_y):
            for i in range(0, nx, step_x):
                v = data_f[j, i]

                # skip masked / invalid points
                if np.ma.is_masked(v) or np.isnan(v):
                    continue

                lat = lats[j, i]
                lon = lons[j, i]

                if not (
                    EXTENT[0] <= lon <= EXTENT[1]
                    and EXTENT[2] <= lat <= EXTENT[3]
                ):
                    continue

                txt = ax.text(
                    lon,
                    lat,
                    f"{int(round(float(v)))}",
                    transform=ccrs.PlateCarree(),
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    ha="center",
                    va="center",
                    zorder=5,
                )
                txt.set_path_effects(
                    [pe.withStroke(linewidth=2.2, foreground="white", alpha=0.85)]
                )

    # Title + timestamp
    ax.set_title(f"{TITLE}\nValid: {valid_iso}", fontsize=14, pad=12)

    # WxLaunch watermark (figure-level so it always shows)
    fig.text(
        0.985,
        0.02,
        WATERMARK_TEXT,
        ha="right",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        color="white",
        alpha=0.7,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046)
    cbar.set_label("°F")

    plt.tight_layout()
    plt.savefig(OUTFILE, bbox_inches="tight")
    plt.close(fig)

    with open(STATEFILE, "w") as f:
        json.dump({"valid_utc": valid_iso}, f)

    print(f"Wrote: {OUTFILE}")


if __name__ == "__main__":
    main()
