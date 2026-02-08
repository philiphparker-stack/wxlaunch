# make_dewpoint_map_ndfd.py
import os, json
from datetime import datetime, timezone

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


OUTFILE = os.path.join("maps", "latest_dewpoint_ndfd.png")
STATEFILE = os.path.join("maps", "_last_ndfd_dew.json")

NDFD_URL = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.td.bin"


EXTENT = (-125, -66.5, 24, 50.5)

DPI = 150
TITLE = "Daytime Dewpoint (°F)"
WATERMARK_TEXT = "WxLaunch"

DEW_MIN_F = -10
DEW_MAX_F = 80
STEP_F = 1


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def download(url, dest):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)


def k_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


def dewpoint_cmap():
    anchors = [
        (-10, "#6e40aa"),
        (0, "#4c6edb"),
        (20, "#38a6db"),
        (30, "#4cc3a1"),
        (40, "#7fd34e"),
        (50, "#c9e75f"),
        (60, "#f6d743"),
        (65, "#f7a73c"),
        (70, "#e95b2a"),
        (75, "#c91d13"),
        (80, "#7f0000"),
    ]
    t0, t1 = DEW_MIN_F, DEW_MAX_F
    xs = [(t - t0) / (t1 - t0) for t, _ in anchors]
    cs = [c for _, c in anchors]
    return mcolors.LinearSegmentedColormap.from_list("dewpoint", list(zip(xs, cs)))


def read_grib(grib_path):
    import pygrib

    grbs = pygrib.open(grib_path)

    # NDFD dewpoint files contain dewpoint only — take first message
    m = grbs.message(1)

    data = m.values
    lats, lons = m.latlons()
    valid = m.validDate.replace(tzinfo=timezone.utc)

    grbs.close()
    return data, lats, lons, valid


def main():
    ensure_dir(OUTFILE)
    tmp = os.path.join("maps", "_ndfd_dew.grib2")

    download(NDFD_URL, tmp)
    data_k, lats, lons, valid = read_grib(tmp)

    valid_iso = valid.strftime("%Y-%m-%dT%H:%MZ")
    if os.path.exists(STATEFILE):
        if json.load(open(STATEFILE)).get("valid") == valid_iso:
            print("Same valid time; skipping render.")
            return

    data_f = k_to_f(data_k)
    levels = np.arange(DEW_MIN_F, DEW_MAX_F + STEP_F, STEP_F)

    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    ax = plt.axes(projection=ccrs.LambertConformal(-96, 38))
    ax.set_extent(EXTENT, ccrs.PlateCarree())

    cf = ax.contourf(
        lons, lats, data_f,
        levels=levels,
        cmap=dewpoint_cmap(),
        transform=ccrs.PlateCarree(),
        antialiased=True
    )

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6)

    ax.set_title(f"{TITLE}\nValid: {valid_iso}", fontsize=14, pad=12)

    fig.text(
        0.985, 0.02, WATERMARK_TEXT,
        ha="right", va="bottom",
        fontsize=20, fontweight="bold",
        color="white", alpha=0.7,
        path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    )

    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, fraction=0.046)
    cbar.set_label("°F")

    plt.tight_layout()
    plt.savefig(OUTFILE, bbox_inches="tight")
    plt.close(fig)

    json.dump({"valid": valid_iso}, open(STATEFILE, "w"))
    print(f"Wrote: {OUTFILE}")


if __name__ == "__main__":
    main()
