# make_wind_map_ndfd.py
import os, json
from datetime import datetime, timezone

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

OUTFILE   = os.path.join("maps", "latest_wind_ndfd.png")
STATEFILE = os.path.join("maps", "_last_ndfd_wind.json")

# NDFD wind speed + direction (CONUS) day 1-3
BASE = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/"
WSPD_URL = BASE + "ds.wspd.bin"
WDIR_URL = BASE + "ds.wdir.bin"

EXTENT = (-125, -66.5, 24, 50.5)

DPI = 150
TITLE = "Wind (Speed + Direction)"
WATERMARK_TEXT = "WxLaunch"

# Speed shading (mph)
SPD_MIN = 0
SPD_MAX = 60
SPD_STEP = 1

# Arrow density
ARROWS_ACROSS = 30  # smaller = fewer arrows

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def download(url, dest):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)

def wind_speed_cmap():
    # dark -> bright (clean and readable)
    anchors = [
        (0,  "#102a43"),
        (5,  "#1c4e80"),
        (10, "#2a9d8f"),
        (20, "#e9c46a"),
        (30, "#f4a261"),
        (40, "#e76f51"),
        (50, "#c1121f"),
        (60, "#780000"),
    ]
    xs = [(v - SPD_MIN) / (SPD_MAX - SPD_MIN) for v, _ in anchors]
    cs = [c for _, c in anchors]
    return mcolors.LinearSegmentedColormap.from_list("windspd", list(zip(xs, cs)))

def read_first_message(grib_path):
    import pygrib
    grbs = pygrib.open(grib_path)
    m = grbs.message(1)
    data = m.values
    lats, lons = m.latlons()
    valid = m.validDate.replace(tzinfo=timezone.utc)
    units = getattr(m, "units", "")
    grbs.close()
    return data, lats, lons, valid, units

def convert_speed_to_mph(speed, units):
    u = (units or "").lower().strip()
    # common possibilities
    if "m s**-1" in u or "m/s" in u or "m s-1" in u:
        return speed * 2.2369362920544  # m/s -> mph
    if "knot" in u or "kt" == u:
        return speed * 1.1507794480235  # kt -> mph
    # if unknown, assume m/s (most common for grib wind)
    return speed * 2.2369362920544

def dir_speed_to_uv(dir_deg, spd):
    # Meteorological direction: degrees FROM which wind blows
    # Convert to math u/v (east/north components) where wind blows TOWARD
    rad = np.deg2rad(dir_deg)
    u = -spd * np.sin(rad)
    v = -spd * np.cos(rad)
    return u, v

def main():
    ensure_dir(OUTFILE)

    wspd_path = os.path.join("maps", "_ndfd_wspd.grib2")
    wdir_path = os.path.join("maps", "_ndfd_wdir.grib2")

    download(WSPD_URL, wspd_path)
    download(WDIR_URL, wdir_path)

    wspd_raw, lats, lons, valid1, units1 = read_first_message(wspd_path)
    wdir_deg, _, _, valid2, _ = read_first_message(wdir_path)

    # Use the later valid as "current"
    valid = valid1 if valid1 >= valid2 else valid2
    valid_iso = valid.strftime("%Y-%m-%dT%H:%MZ")

    # Skip if unchanged
    if os.path.exists(STATEFILE) and os.path.exists(OUTFILE):
        try:
            if json.load(open(STATEFILE, "r")).get("valid") == valid_iso:
                print(f"Same valid time ({valid_iso}); skipping render.")
                return
        except Exception:
            pass

    wspd_mph = convert_speed_to_mph(wspd_raw, units1)

    # Levels
    levels = np.arange(SPD_MIN, SPD_MAX + SPD_STEP, SPD_STEP)
    cmap = wind_speed_cmap()
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(14, 8), dpi=DPI)
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-96, central_latitude=38))
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Speed shading
    cf = ax.contourf(
        lons, lats, wspd_mph,
        levels=levels, cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        antialiased=True
    )

    # Boundaries
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.6, edgecolor="black")

    # Wind arrows (subsample)
    ny, nx = wspd_mph.shape
    step_x = max(1, nx // ARROWS_ACROSS)
    step_y = max(1, int(step_x * (ny / nx)))

    # Convert to u/v for arrows (mph components)
    u, v = dir_speed_to_uv(wdir_deg, wspd_mph)

    # Mask invalids
    mask = np.ma.getmaskarray(wspd_mph) | np.ma.getmaskarray(wdir_deg) | np.isnan(wspd_mph) | np.isnan(wdir_deg)
    u = np.ma.array(u, mask=mask)
    v = np.ma.array(v, mask=mask)

    ax.quiver(
        lons[::step_y, ::step_x],
        lats[::step_y, ::step_x],
        u[::step_y, ::step_x],
        v[::step_y, ::step_x],
        transform=ccrs.PlateCarree(),
        scale=700,   # arrow size tuning
        width=0.0022,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
        zorder=6
    )

    ax.set_title(f"{TITLE}\nValid: {valid_iso}", fontsize=14, pad=12)

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
    cbar.set_label("Wind speed (mph)")

    plt.tight_layout()
    plt.savefig(OUTFILE, bbox_inches="tight")
    plt.close(fig)

    json.dump({"valid": valid_iso}, open(STATEFILE, "w"))
    print(f"Wrote: {OUTFILE}")

if __name__ == "__main__":
    main()
