import os, json
from datetime import datetime, timezone

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ----------------------------
# SETTINGS
# ----------------------------
OUTFILE   = os.path.join("maps", "latest_temp.png")
STATEFILE = os.path.join("maps", "_last_run.json")

LABEL_MODE = "real"   # "real" or "random"
RUN_INTERVAL_MINUTES = 10

# CONUS extent like TV maps
EXTENT = (-125, -66.5, 24, 50)  # (west, east, south, north)

# smoother background
GRID_NX = 520
GRID_NY = 300

# Weather Central legend shown (-20 to 80)
# (You can expand later; I’m matching the “TV vibe”)
TMIN_F = -20
TMAX_F = 80

MAX_STATIONS = 220

# IDW tuning
IDW_POWER = 1.7
IDW_RADIUS_DEG = 7

UA = {"User-Agent": "WXLaunch (temp-map script) - contact: you@example.com"}

STATION_IDS = [
    "KSEA","KPDX","KSFO","KOAK","KLAX","KSAN","KPHX","KTUS","KLAS","KSLC","KDEN",
    "KABQ","KELP","KDFW","KDAL","KHOU","KIAH","KOKC","KTUL","KMCI","KSTL","KMSP",
    "KORD","KMDW","KDTW","KCLE","KPIT","KBUF","KROC","KBOS","KBDL","KJFK","KLGA",
    "KEWR","KPHL","KBWI","KDCA","KIAD","KRIC","KCLT","KATL","KBNA","KMEM","KMSY",
    "KMIA","KFLL","KTPA","KJAX","KORL","KMCO","KCHS","KSAV","KAVL","KRDU","KGSO",
    "KIND","KSDF","KCVG","KCMH","KDAY","KDSM","KOMA","KICT","KAMA","KBOI","KGEG",
    "KALB","KSYR","KACK","KPVD","KPWM","PHNL"
]

# ----------------------------
# WEATHER-CENTRAL-LIKE COLORS
# (purple -> blue -> cyan -> green -> yellow -> orange -> red)
# ----------------------------
COLOR_STOPS = [
    (-20, "#f5a3ff"),  # very cold pink/purple
    (-10, "#c06cff"),
    (0,   "#6a52ff"),
    (10,  "#2b78ff"),
    (20,  "#00b6ff"),
    (30,  "#00e7ff"),
    (40,  "#00ff9a"),
    (50,  "#b7ff00"),
    (60,  "#ffe100"),
    (70,  "#ff8a00"),
    (80,  "#ff2a00"),
]

def ensure_dirs():
    os.makedirs("maps", exist_ok=True)

def safe_get(url, timeout=18):
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.json()

def load_last_state():
    if os.path.exists(STATEFILE):
        try:
            with open(STATEFILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_last_state(state):
    with open(STATEFILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def f_from_c(c):
    return c * 9/5 + 32

def fetch_station_meta(station_id):
    j = safe_get(f"https://api.weather.gov/stations/{station_id}")
    lon, lat = j["geometry"]["coordinates"]
    return float(lat), float(lon)

def fetch_latest_temp_f(station_id):
    j = safe_get(f"https://api.weather.gov/stations/{station_id}/observations/latest")
    props = j["properties"]
    temp_c = props.get("temperature", {}).get("value", None)
    ts = props.get("timestamp", None)
    if temp_c is None or ts is None:
        return None, None
    return float(f_from_c(float(temp_c))), ts

def build_colormap(stops, tmin, tmax):
    xs, cs = [], []
    for t, hexcol in stops:
        x = (t - tmin) / (tmax - tmin)
        xs.append(float(np.clip(x, 0, 1)))
        cs.append(hexcol)
    return mcolors.LinearSegmentedColormap.from_list("wc_like_temp", list(zip(xs, cs)))

def idw_grid(lons, lats, vals, grid_lons, grid_lats, power=2.0, radius_deg=None, taper=0.55):
    """
    IDW with OPTIONAL SOFT RADIUS TAPER (no hard ring edges).
    taper controls how quickly weights fade near the radius (0.4–0.7 is a good range).
    """
    dx = grid_lons[None, :, :] - lons[:, None, None]
    dy = grid_lats[None, :, :] - lats[:, None, None]
    dist2 = dx*dx + dy*dy
    dist2 = np.where(dist2 < 1e-8, 1e-8, dist2)

    # classic IDW base weights
    w = 1.0 / (dist2 ** (power / 2.0))

    if radius_deg is not None:
        r2 = radius_deg * radius_deg

        # SOFT taper: multiply by exp(-(d^2)/(taper*r)^2)
        # - inside radius: still strong
        # - outside radius: smoothly fades, no hard ring
        sigma2 = (taper * radius_deg) ** 2
        w *= np.exp(-dist2 / sigma2)

        # optional: kill truly tiny weights to speed things up / reduce haze
        w = np.where(dist2 <= (3.0 * r2), w, 0.0)

    num = np.sum(w * vals[:, None, None], axis=0)
    den = np.sum(w, axis=0)

    out = np.full_like(num, np.nan, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def soften_field(z, passes=2):
    """
    Cheap smoothing to reduce 'bullseyes' without SciPy.
    (3x3 weighted blur, repeated)
    """
    if z is None:
        return z
    z2 = z.copy()
    for _ in range(passes):
        a = z2
        # pad edges
        p = np.pad(a, ((1,1),(1,1)), mode="edge")
        # weighted kernel-ish blur
        z2 = (
            4*p[1:-1,1:-1] +
            2*(p[ :-2,1:-1] + p[2:  ,1:-1] + p[1:-1, :-2] + p[1:-1,2:  ]) +
            1*(p[ :-2, :-2] + p[ :-2,2:  ] + p[2:  , :-2] + p[2:  ,2:  ])
        ) / 16.0
    return z2

def main():
    ensure_dirs()

    coord_cache_path = os.path.join("maps", "_station_coords.json")
    if os.path.exists(coord_cache_path):
        with open(coord_cache_path, "r", encoding="utf-8") as f:
            coord_cache = json.load(f)
    else:
        coord_cache = {}

    station_points = []
    newest_ts = None

    ids = STATION_IDS[:MAX_STATIONS]
    for sid in ids:
        try:
            if sid not in coord_cache:
                lat, lon = fetch_station_meta(sid)
                coord_cache[sid] = {"lat": lat, "lon": lon}
            else:
                lat = coord_cache[sid]["lat"]
                lon = coord_cache[sid]["lon"]

            tf, ts = fetch_latest_temp_f(sid)
            if tf is None or ts is None:
                continue

            # clamp to legend range so labels/field match TV graphic
            tf = float(np.clip(tf, TMIN_F, TMAX_F))

            if newest_ts is None or ts > newest_ts:
                newest_ts = ts

            station_points.append((sid, lon, lat, tf, ts))
        except Exception:
            continue

    with open(coord_cache_path, "w", encoding="utf-8") as f:
        json.dump(coord_cache, f, indent=2)

    if not station_points:
        print("No station data returned. Nothing to plot.")
        return

    last = load_last_state()
    if newest_ts and last.get("newest_ts") == newest_ts:
        print("No newer data since last render. Skipping.")
        return

    west, east, south, north = EXTENT

    # grid
    gx = np.linspace(west, east, GRID_NX)
    gy = np.linspace(south, north, GRID_NY)
    grid_lons, grid_lats = np.meshgrid(gx, gy)

    lons = np.array([p[1] for p in station_points], dtype=float)
    lats = np.array([p[2] for p in station_points], dtype=float)
    vals = np.array([p[3] for p in station_points], dtype=float)

    # IDW with radius to avoid far-away influence
    bg = idw_grid(lons, lats, vals, 
    grid_lons, grid_lats,
              power=IDW_POWER, radius_deg=IDW_RADIUS_DEG, taper=0.65
    )

    holes = np.isnan(bg)
    if np.any(holes):
        bg2 = idw_grid(
            lons, lats, vals, 
            grid_lons, grid_lats,
            power=2.2, 
            radius_deg=None
    )
        bg[holes] = bg2[holes]

    # fill holes with a broader pass
    holes = np.isnan(bg)
    if np.any(holes):
        bg2 = idw_grid(lons, lats, vals, grid_lons, grid_lats,
                       power=2.2, radius_deg=None)
        bg[holes] = bg2[holes]

    bg = np.clip(bg, TMIN_F, TMAX_F)

    # FINAL smoothing pass for broadcast-style gradients
    bg = soften_field(bg, passes=1)


    # soften bullseyes (big visual difference)
    bg = soften_field(bg, passes=2)

    # colormap + norm (nice smooth gradient like TV)
    cmap = build_colormap(COLOR_STOPS, TMIN_F, TMAX_F)

    # Weather Central uses a top bar with labeled ticks:
    ticks = np.arange(TMIN_F, TMAX_F + 1, 10)
    norm = mcolors.Normalize(vmin=TMIN_F, vmax=TMAX_F)

    # ----------------------------
    # FIGURE LAYOUT (TV STYLE)
    # ----------------------------
    # 1536x768-ish
    fig = plt.figure(figsize=(12, 6), dpi=160, facecolor="#1f232a")

    # main map axes (leave room at top for header + legend)
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.80], projection=ccrs.PlateCarree())
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

    # Basemap: stock_img gives you a satellite/terrain-ish look
    # (Cartopy may download Natural Earth imagery the first time)
    ax.stock_img()

    # Add water a touch darker (TV feel)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#0b2236", alpha=0.25)

    # Temp overlay (semi-transparent is KEY)
    # Use pcolormesh for smooth fill like Weather Central
    mesh = ax.pcolormesh(
        grid_lons, grid_lats, bg,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        shading="gouraud",
        alpha=0.78,
        zorder=3
    )

    # Borders: thin + clean
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="black", alpha=0.65, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, edgecolor="black", alpha=0.55, zorder=5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35, edgecolor="black", alpha=0.45, zorder=5)

    # Station number labels: bold white with thick black outline
    stroke = [pe.withStroke(linewidth=2.8, foreground="black")]

    if LABEL_MODE == "random":
        rng = np.random.default_rng(42)
        for _ in range(95):
            lon = float(rng.uniform(west + 2, east - 2))
            lat = float(rng.uniform(south + 2, north - 2))
            t = int(rng.uniform(TMIN_F, TMAX_F))
            ax.text(lon, lat, str(t),
                    transform=ccrs.PlateCarree(),
                    fontsize=8.0, fontweight="bold",
                    ha="center", va="center",
                    color="white", zorder=10,
                    path_effects=stroke)
    else:
        LABEL_EVERY_N = 2 if lon > -90 else 4  # smaller = more labels
        for i, (sid, lon, lat, tf, ts) in enumerate(station_points):
            skip = 2 if lon > -90 else 4
            if i % skip != 0:
                continue
            ax.text(lon, lat, f"{int(round(tf))}",
                    transform=ccrs.PlateCarree(),
                    fontsize=8.5, fontweight="bold",
                    ha="center", va="center",
                    color="white", zorder=10,
                    path_effects=stroke)

    # ----------------------------
    # HEADER BAR + TOP LEGEND
    # ----------------------------
    # Header band
    hdr = fig.add_axes([0.02, 0.895, 0.96, 0.085])
    hdr.set_facecolor("#2b2f36")
    hdr.set_xticks([]); hdr.set_yticks([])
    for s in hdr.spines.values():
        s.set_visible(False)

    if newest_ts:
        dt = datetime.fromisoformat(newest_ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        stamp = dt.strftime("%Y-%m-%d %H:%M UTC")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    hdr.text(0.012, 0.62, "US Current Temperatures (°F)",
             color="white", fontsize=14, fontweight="bold", ha="left", va="center")

    # small update text right side
    hdr.text(0.988, 0.62, f"Updated: {stamp}",
             color="#cfd6df", fontsize=10.5, ha="right", va="center")

    # Top colorbar axis inside header (like Weather Central)
    cax = fig.add_axes([0.20, 0.915, 0.60, 0.022])
    cb = plt.colorbar(mesh, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors="white", length=0)
    cb.ax.set_facecolor("#2b2f36")

    # make tick labels white
    for t in cb.ax.get_xticklabels():
        t.set_color("white")

    # WXLaunch mark (subtle) bottom right
    ax.text(0.985, 0.02, "WXLaunch",
            transform=ax.transAxes,
            fontsize=11, fontweight="bold",
            color="white", alpha=0.35,
            ha="right", va="bottom",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="black")],
            zorder=20)

    # Save (tight but keep header)
    plt.savefig(OUTFILE, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    save_last_state({"newest_ts": newest_ts, "rendered_at_utc": stamp})
    print(f"Rendered: {OUTFILE}")

if __name__ == "__main__":
    main()
