"""
LULC Change Analysis Tool  (v4 BETA)
------------------------------------
- 10 land use / land cover classes with fixed colours
- Random Forest (ESA WorldCover labels) + JRC GHSL settlements + JRC Global
  Surface Water + OpenStreetMap roads/canals
- Quarterly / whole-year image seasons, radius 5-30 km
- Permanent report history and result cache (Supabase), PDF with map,
  north arrow, scale bar and 10-class legend
"""
import base64
import datetime
import hashlib
import io
import json
import math
import textwrap

import ee
import numpy as np
import requests
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

try:
    import contextily as ctx
except Exception:  # basemap is optional
    ctx = None

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
SITE_NAME = "www.tanmaysanyal.in"
APP_VERSION = "4.0-beta"      # part of the cache key: bump to invalidate cache
BOTTLE = "#0B5D45"
BEIGE = "#F5F0E1"

# The 10 classes (code, name, HEX, kind). Codes 1-2 are drawn as lines.
CLASSES = {
    1: ("Main Road", "#E31A1C", "line"),
    2: ("Canal", "#0033CC", "line"),
    3: ("Rural Area", "#FDAE61", "area"),
    4: ("Vegetation", "#1B7837", "area"),
    5: ("Pond / Lake", "#4FC3F7", "area"),
    6: ("Urban Area", "#A63603", "area"),
    7: ("Scrub Land", "#A6D96A", "area"),
    8: ("Crop Land", "#FFE500", "area"),
    9: ("Aquaculture", "#1DE9B6", "area"),
    10: ("Open Land", "#E7298A", "area"),
}
assert len({v[1] for v in CLASSES.values()}) == 10  # all colours unique
AREA_CODES = [k for k, v in CLASSES.items() if v[2] == "area"]
PALETTE = [CLASSES[k][1].lstrip("#") for k in sorted(CLASSES)]

NAMES = ["blue", "green", "red", "nir", "swir1", "swir2"]
SENSORS = {
    "L5": ("LANDSAT/LT05/C02/T1_L2",
           ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]),
    "L7": ("LANDSAT/LE07/C02/T1_L2",
           ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]),
    "L8": ("LANDSAT/LC08/C02/T1_L2",
           ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]),
    "L9": ("LANDSAT/LC09/C02/T1_L2",
           ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]),
}
PERIODS = {
    "1990 to 2000": [1990, 2000],
    "2000 to 2010": [2000, 2010],
    "2010 to 2020": [2010, 2020],
    "1990 to 2020 (4 maps)": [1990, 2000, 2010, 2020],
}
# Calendar quarters (month range); None = whole year
SEASONS = {
    "Quarter 1 (Jan-Mar)": (1, 3),
    "Quarter 2 (Apr-Jun)": (4, 6),
    "Quarter 3 (Jul-Sep)": (7, 9),
    "Quarter 4 (Oct-Dec)": (10, 12),
    "Whole Year": None,
}
RADII_KM = [5, 10, 15, 20, 25, 30]
R_EARTH = 6378137.0
SCALE = 30  # analysis resolution in metres

# OLI (Landsat 8/9) -> ETM+ (Landsat 5/7) harmonization (Roy et al. 2016)
OLI_SLOPE = [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]
OLI_INTERCEPT = [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]

# ESA WorldCover 2020 -> training classes (vegetation, scrub, crop, open land)
WC_FROM = [10, 95, 20, 30, 40, 60]
WC_TO = [4, 4, 7, 7, 8, 10]
TRAIN_CLASSES = [4, 7, 8, 10]
FEATURES = NAMES + ["ndvi", "mndwi", "ndbi"]

BUILT_MIN = 1000      # GHSL built_surface (m2 per 100 m cell) => settlement
URBAN_DENS = 0.30     # share built within 500 m: >= urban, otherwise rural
AQUA_MAX_PX = 60      # water patch <= 60 px (~5 ha) ...
AQUA_DENS = 0.10      # ... and >= 10% water within 500 m => aquaculture


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def merc(lon, lat):
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def hav_km(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def circle_lonlat(lat, lon, r_km, n=96):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        dlat = (r_km / 111.32) * math.sin(a)
        dlon = (r_km / (111.32 * math.cos(math.radians(lat)))) * math.cos(a)
        pts.append((lon + dlon, lat + dlat))
    pts.append(pts[0])
    return pts


def circle_wkt(lat, lon, r_km):
    pts = circle_lonlat(lat, lon, r_km, 48)
    return "POLYGON((" + ",".join(f"{x:.5f} {y:.5f}" for x, y in pts) + "))"


def to_jpeg(png_bytes, quality=90):
    im = PILImage.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Earth Engine
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_ee():
    info = dict(st.secrets["gee_service_account"])
    creds = ee.ServiceAccountCredentials(info["client_email"],
                                         key_data=json.dumps(info))
    ee.Initialize(creds, project=info["project_id"])
    return True


def sensors_for(year):
    s = []
    if year <= 2011:
        s.append("L5")
    if 1999 <= year <= 2022:
        s.append("L7")
    if year >= 2013:
        s.append("L8")
    if year >= 2022:
        s.append("L9")
    return s


def _prep(key):
    _, src = SENSORS[key]

    def f(img):
        qa = img.select("QA_PIXEL")  # bit mask: never resample this band
        clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        sr = (img.select(src, NAMES).resample("bilinear")
              .multiply(0.0000275).add(-0.2))
        if key in ("L8", "L9"):  # make OLI comparable with TM/ETM+
            slope = ee.Image.constant(OLI_SLOPE).rename(NAMES)
            icpt = ee.Image.constant(OLI_INTERCEPT).rename(NAMES)
            sr = sr.multiply(slope).add(icpt)
        sr = sr.updateMask(clear)
        return sr.copyProperties(img, ["system:time_start"])

    return f


def composite(year, region, season, win=1):
    """Cloud-free median composite of (year-win ... year+win) in a season."""
    keys = sorted({k for y in range(year - win, year + win + 1)
                   for k in sensors_for(y)})
    merged = None
    for k in keys:
        col = (ee.ImageCollection(SENSORS[k][0])
               .filterBounds(region)
               .filterDate(f"{year - win}-01-01", f"{year + win + 1}-01-01"))
        if season:
            col = col.filter(ee.Filter.calendarRange(season[0], season[1],
                                                     "month"))
        col = col.map(_prep(k))
        merged = col if merged is None else merged.merge(col)
    n = merged.size().getInfo()
    if n == 0:
        raise RuntimeError(
            f"No usable Landsat images for {year} in this image season. "
            "Try another quarter or 'Whole Year'.")
    return merged.median().clip(region), n


def features(img):
    ndvi = img.normalizedDifference(["nir", "red"]).rename("ndvi")
    mndwi = img.normalizedDifference(["green", "swir1"]).rename("mndwi")
    ndbi = img.normalizedDifference(["swir1", "nir"]).rename("ndbi")
    return img.addBands([ndvi, mndwi, ndbi])


def train_model(train_img, region):
    """Random Forest trained on ESA WorldCover 2020 labels (30% held out)."""
    wc = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map")
    label = wc.remap(WC_FROM, WC_TO).rename("label")
    stack = features(train_img).select(FEATURES).addBands(label)
    samples = stack.stratifiedSample(
        numPoints=0, classBand="label", region=region, scale=SCALE,
        classValues=TRAIN_CLASSES, classPoints=[500] * len(TRAIN_CLASSES),
        seed=42, tileScale=4, dropNulls=True)
    n = samples.size().getInfo()
    if n < 100:
        raise RuntimeError("Too few training samples for this location "
                           "(try a larger radius or another image season).")
    samples = samples.randomColumn("rnd", 7)
    tr = samples.filter(ee.Filter.lt("rnd", 0.7))
    te = samples.filter(ee.Filter.gte("rnd", 0.7))
    clf = ee.Classifier.smileRandomForest(numberOfTrees=100, seed=7).train(
        tr, "label", FEATURES)
    em = te.classify(clf).errorMatrix("label", "classification")
    return clf, {"overall": em.accuracy().getInfo(),
                 "kappa": em.kappa().getInfo(), "n": n}


def built_image(year):
    epoch = min(max(int(round(year / 5.0)) * 5, 1975), 2030)
    return ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}").select(
        "built_surface")


def built_surface_km2(year, region):
    try:
        v = built_image(year).reduceRegion(
            ee.Reducer.sum(), region, 100, maxPixels=1e10,
            bestEffort=True).get("built_surface").getInfo()
        return v / 1e6 if v is not None else None
    except Exception:
        return None


def gsw_available():
    try:
        ee.ImageCollection("JRC/GSW1_4/YearlyHistory").size().getInfo()
        return True
    except Exception:
        return False


def water_mask(img, year, use_gsw):
    """Permanent water of that year (JRC Global Surface Water), else MNDWI."""
    if use_gsw:
        y = min(max(year, 1984), 2021)
        gsw = (ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
               .filterDate(f"{y}-01-01", f"{y + 1}-01-01").first()
               .select("waterClass"))
        return gsw.eq(3).unmask(0)
    f = features(img)
    return (f.select("mndwi").gt(0.1).And(f.select("ndvi").lt(0.3))
            .unmask(0))


def classify_year(img, clf, year, use_gsw):
    """Raster classes 3-10 (roads and canals are drawn from OSM lines)."""
    proj = ee.Projection("EPSG:3857").atScale(SCALE)
    cover = img.select("red").mask()
    rf = (features(img).select(FEATURES).classify(clf)
          .updateMask(cover).reproject(proj))      # 4, 7, 8, 10
    # settlements: GHSL built cells, urban vs rural by local built density
    b = built_image(year).unmask(0)
    built = b.gt(BUILT_MIN)
    dens = b.divide(10000).focalMean(500, "circle", "meters")
    urban = built.And(dens.gte(URBAN_DENS))
    rural = built.And(dens.lt(URBAN_DENS))
    c = rf.where(rural, 3).where(urban, 6)
    # water: permanent water; small water clusters = aquaculture (experimental)
    water = water_mask(img, year, use_gsw).reproject(proj)
    patch = water.selfMask().connectedPixelCount(maxSize=100,
                                                 eightConnected=False)
    small = patch.lte(AQUA_MAX_PX).unmask(0)
    wdens = water.focalMean(500, "circle", "meters")
    aqua = water.And(small).And(wdens.gte(AQUA_DENS))
    pond = water.And(aqua.Not())
    c = c.where(pond, 5).where(aqua, 9)
    return c.rename("lulc").toInt().updateMask(rf.mask())


def area_by(code_img, region):
    r = (ee.Image.pixelArea().divide(1e6).addBands(code_img)
         .reduceRegion(reducer=ee.Reducer.sum().group(1, "c"),
                       geometry=region, scale=SCALE,
                       maxPixels=1e10, bestEffort=True).getInfo())
    return {int(g["c"]): g["sum"] for g in r.get("groups", [])}


def bbox(region):
    co = region.bounds().coordinates().getInfo()[0]
    xs = [p[0] for p in co]
    ys = [p[1] for p in co]
    return min(xs), min(ys), max(xs), max(ys)


def png(img, rect, vis, dim=1400):
    url = img.visualize(**vis).getThumbURL(
        {"region": rect, "dimensions": dim, "crs": "EPSG:3857",
         "format": "png"})
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return resp.content


def population(year, region):
    try:
        epoch = min(max(int(round(year / 5.0)) * 5, 1975), 2030)
        img = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{epoch}").select(
            "population_count")
        v = img.reduceRegion(ee.Reducer.sum(), region, 100, maxPixels=1e10,
                             bestEffort=True).get("population_count").getInfo()
        return {"epoch": epoch, "value": v}
    except Exception:
        return None


# --------------------------------------------------------------------------
# External data: GBIF (wildlife) and OpenStreetMap (roads, canals)
# --------------------------------------------------------------------------
UA = {"User-Agent": f"LULC-Change-Analyzer/4.0 (+https://{SITE_NAME})"}
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.private.coffee/api/interpreter"]


def gbif(lat, lon, r_km, year):
    try:
        params = {"geometry": circle_wkt(lat, lon, r_km),
                  "year": f"{year - 2},{year + 2}", "kingdomKey": 1,
                  "hasCoordinate": "true", "limit": 0,
                  "facet": "SPECIES_KEY", "facetLimit": 1000}
        j = requests.get("https://api.gbif.org/v1/occurrence/search",
                         params=params, headers=UA, timeout=60).json()
        sp = len(j["facets"][0]["counts"]) if j.get("facets") else 0
        return {"records": j.get("count", 0), "species": sp}
    except Exception:
        return None


def osm_lines(lat, lon, r_km):
    """Main roads and canals (current OpenStreetMap) + lengths inside radius."""
    out = {"road": [], "canal": [], "ok": False,
           "road_km": None, "canal_km": None}
    r = int(r_km * 1000)
    q = (f'[out:json][timeout:40];('
         f'way["highway"~"^(motorway|trunk|primary|secondary)$"]'
         f'(around:{r},{lat},{lon});'
         f'way["waterway"="canal"](around:{r},{lat},{lon});'
         f');out geom;')
    data = None
    for url in OVERPASS:
        try:
            resp = requests.post(url, data={"data": q}, headers=UA, timeout=70)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception:
            continue
    if data is None:
        return out
    out["ok"] = True
    road_km = canal_km = 0.0
    for el in data.get("elements", []):
        g = el.get("geometry")
        if not g:
            continue
        pts = [(p["lon"], p["lat"]) for p in g]
        length = 0.0
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if hav_km(lon, lat, mx, my) <= r_km:
                length += hav_km(x1, y1, x2, y2)
        xy = [merc(x, y) for x, y in pts]
        if el.get("tags", {}).get("waterway") == "canal":
            out["canal"].append(xy)
            canal_km += length
        else:
            out["road"].append(xy)
            road_km += length
    out["road_km"], out["canal_km"] = road_km, canal_km
    return out


# --------------------------------------------------------------------------
# Map drawing (publication style)
# --------------------------------------------------------------------------
def class_legend():
    """One entry per class (1-10): colour swatch + name."""
    handles = []
    for k in sorted(CLASSES):
        name, col, kind = CLASSES[k]
        if kind == "line":
            handles.append(Line2D([0], [0], color=col, lw=4, label=name))
        else:
            handles.append(Patch(facecolor=col, edgecolor="black", label=name))
    return handles


def draw_map(panels, ext, lat, lon, r_km, osm, legend, note, ncols=2):
    """panels: list of (title, png_bytes). ext = (xmin, ymin, xmax, ymax)
    in Web-Mercator metres. Scale bar uses the true ground distance."""
    n = len(panels)
    ncols = 1 if n == 1 else ncols
    nrows = math.ceil(n / ncols)
    xmin, ymin, xmax, ymax = ext
    aspect = (ymax - ymin) / (xmax - xmin)
    pw, side = 5.2, 2.7
    figw, figh = pw * ncols + side, pw * aspect * nrows + 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw, figh), dpi=150,
                             squeeze=False)
    cx, cy = zip(*[merc(x, y) for x, y in circle_lonlat(lat, lon, r_km)])
    cos_lat = math.cos(math.radians(lat))
    width_km = (xmax - xmin) * cos_lat / 1000.0      # ground width of the map
    nice = [1, 2, 2.5, 5, 10, 15, 20]
    step = max([v for v in nice if 3 * v <= 0.42 * width_km] or [nice[0]])
    seg = step * 1000.0 / cos_lat / (xmax - xmin)    # one segment, axes frac

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        title, data = panels[i]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_facecolor("#f4f4f4")
        if ctx is not None:
            zoom = int(min(max(round(math.log2(
                156543.03 * 1000.0 / (xmax - xmin))), 1), 15))
            for provider in (ctx.providers.Esri.WorldGrayCanvas,
                             ctx.providers.OpenStreetMap.Mapnik):
                try:
                    ctx.add_basemap(ax, crs="EPSG:3857", source=provider,
                                    zoom=zoom, attribution=False)
                    break
                except Exception:
                    continue
        arr = np.array(PILImage.open(io.BytesIO(data)).convert("RGBA"),
                       dtype=float)
        arr[..., 3] *= 0.88
        ax.imshow(arr.astype("uint8"), extent=(xmin, xmax, ymin, ymax),
                  origin="upper", interpolation="nearest", zorder=3)
        for key, code, lw in (("canal", 2, 1.5), ("road", 1, 1.7)):
            if osm and osm.get(key):
                ax.add_collection(LineCollection(
                    osm[key], colors=CLASSES[code][1], linewidths=lw,
                    zorder=4))
        ax.plot(cx, cy, color="black", lw=1.1, ls="--", zorder=5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.8)
        ax.text(0.5, 0.5, SITE_NAME, transform=ax.transAxes, rotation=30,
                ha="center", va="center", fontsize=15, fontweight="bold",
                color=BOTTLE, alpha=0.20, zorder=9)
        ax.text(0.04, 0.04, title, transform=ax.transAxes, zorder=10,
                fontsize=17 if len(title) <= 5 else 11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black", pad=4))
        # north arrow (top right, inside the map)
        ax.add_patch(FancyBboxPatch(
            (0.885, 0.80), 0.07, 0.165, boxstyle="round,pad=0.004",
            transform=ax.transAxes, facecolor="white", edgecolor="black",
            lw=1.2, alpha=0.92, zorder=9))
        ax.annotate("", xy=(0.92, 0.925), xytext=(0.92, 0.82),
                    xycoords="axes fraction", zorder=10,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2))
        ax.text(0.92, 0.937, "N", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11, fontweight="bold", zorder=10)
        # scale bar (bottom right): 0 - s - 2s - 3s KM, true ground distance
        x0, y0, h = 0.90 - 3 * seg, 0.045, 0.014
        ax.add_patch(Rectangle((x0 - 0.02, y0 - 0.012), 3 * seg + 0.115,
                               h + 0.062, transform=ax.transAxes,
                               facecolor="white", edgecolor="none",
                               alpha=0.85, zorder=9))
        for j in range(3):
            ax.add_patch(Rectangle((x0 + j * seg, y0), seg, h,
                                   transform=ax.transAxes,
                                   facecolor="black" if j % 2 == 0 else "white",
                                   edgecolor="black", lw=0.9, zorder=10))
        for j in range(4):
            lab = f"{j * step:g}" + (" KM" if j == 3 else "")
            ax.text(x0 + j * seg, y0 + h + 0.008, lab,
                    transform=ax.transAxes, ha="center" if j < 3 else "left",
                    fontsize=7.5, fontweight="bold", zorder=10)

    bottom = 0.8 / figh
    fig.subplots_adjust(left=0.01, right=1 - side / figw, bottom=bottom + 0.01,
                        top=0.99, wspace=0.06, hspace=0.05)
    fig.legend(handles=legend, loc="center right",
               bbox_to_anchor=(0.995, 0.5 + bottom / 2), title="Legend",
               title_fontsize=9, fontsize=8, frameon=True, edgecolor="black",
               fancybox=False, borderpad=0.8, labelspacing=0.7)
    fig.text(0.5, 0.05 / figh, textwrap.fill(note, int(figw * 19)),
             ha="center", va="bottom", fontsize=6, color="#555555")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


MAP_NOTE = (f"{SITE_NAME} | Land cover: Landsat (USGS/NASA) Random Forest "
            "trained on ESA WorldCover; settlements: JRC GHSL; water: JRC "
            "Global Surface Water; Main Road and Canal: OpenStreetMap "
            "contributors (current, reference only). Basemap: Esri, HERE, "
            "Garmin, OpenStreetMap contributors. Dashed line = study radius.")


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
WATER = (5, 9)


def run_analysis(lat, lon, r_km, years, period_label, season_label, progress):
    season = SEASONS[season_label]
    region = ee.Geometry.Point([lon, lat]).buffer(r_km * 1000)
    w, s, e, n = bbox(region)
    rect = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)
    x0, y0 = merc(w, s)
    x1, y1 = merc(e, n)
    ext = (x0, y0, x1, y1)
    vis = {"min": 1, "max": 10, "palette": PALETTE}

    use_gsw = gsw_available()
    progress("Building cloud-free composites...")
    imgs, scenes = {}, {}
    for y in years:
        imgs[y], scenes[y] = composite(y, region, season, 1)

    classes, areas, ndvi_mean, pngs, built, acc = {}, {}, {}, {}, {}, {}
    for y in years:
        progress(f"Training Random Forest for {y}...")
        clf, acc[y] = train_model(imgs[y], region)
        progress(f"Classifying and measuring {y}...")
        c = classify_year(imgs[y], clf, y, use_gsw)
        classes[y] = c
        a = area_by(c, region)
        for code in AREA_CODES:
            a.setdefault(code, 0.0)
        areas[y] = a
        ndvi_mean[y] = features(imgs[y]).select("ndvi").reduceRegion(
            ee.Reducer.mean(), region, SCALE, maxPixels=1e10,
            bestEffort=True).get("ndvi").getInfo()
        built[y] = built_surface_km2(y, region)
        pngs[y] = png(c, rect, vis)

    first, last = years[0], years[-1]
    progress("Measuring transitions and river/water change...")
    trans = area_by(classes[first].multiply(100).add(classes[last]), region)
    water = dict(
        lost=sum(v for k, v in trans.items()
                 if k // 100 in WATER and k % 100 not in WATER),
        gain=sum(v for k, v in trans.items()
                 if k // 100 not in WATER and k % 100 in WATER),
        kept=sum(v for k, v in trans.items()
                 if k // 100 in WATER and k % 100 in WATER))
    w1 = classes[first].eq(5).Or(classes[first].eq(9))
    w2 = classes[last].eq(5).Or(classes[last].eq(9))
    wchg = (ee.Image(0).where(w1.And(w2.Not()), 1)
            .where(w1.Not().And(w2), 2).where(w1.And(w2), 3))
    wchg = wchg.updateMask(wchg.gt(0)).clip(region)
    water_png = png(wchg, rect, {"min": 1, "max": 3,
                                 "palette": ["d7191c", "33a02c", "1f78b4"]})

    progress("Fetching roads, canals, population and wildlife data...")
    osm = osm_lines(lat, lon, r_km)
    pop1, pop2 = population(first, region), population(last, region)
    g1, g2 = gbif(lat, lon, r_km, first), gbif(lat, lon, r_km, last)

    progress("Drawing maps...")
    lulc_map = draw_map([(str(y), pngs[y]) for y in years], ext, lat, lon,
                        r_km, osm, class_legend(), MAP_NOTE, ncols=2)
    wlegend = [Patch(facecolor="#1f78b4", edgecolor="black", label="Unchanged water"),
               Patch(facecolor="#d7191c", edgecolor="black", label="Water lost"),
               Patch(facecolor="#33a02c", edgecolor="black", label="New water")]
    water_map = draw_map([(f"{first}-{last}", water_png)], ext, lat, lon,
                         r_km, osm, wlegend, MAP_NOTE)

    now = datetime.datetime.utcnow()
    return dict(lat=lat, lon=lon, r_km=r_km, years=years, dec=2,
                period=period_label, season=season_label,
                scenes=scenes, a=areas, trans=trans, ndvi=ndvi_mean,
                water=water, built=built, acc=acc, use_gsw=use_gsw,
                osm_ok=osm["ok"], road_km=osm["road_km"],
                canal_km=osm["canal_km"],
                lulc_map=to_jpeg(lulc_map), water_map=to_jpeg(water_map),
                p1=pop1, p2=pop2, g1=g1, g2=g2,
                created=now.strftime("%Y-%m-%d %H:%M UTC"))


def pct(a, b):
    return (b - a) / a * 100.0 if a else 0.0


def reliability_flags(r):
    """Plausibility checks: flag results that look like classification artefacts."""
    y1, y2 = r["years"][0], r["years"][-1]
    a1, a2 = r["a"][y1], r["a"][y2]
    area = math.pi * r["r_km"] ** 2
    flags = []
    acc = r.get("acc") or {}
    if acc:
        worst = min(v["overall"] for v in acc.values())
        if worst < 0.75:
            flags.append(f"Agreement with the reference labels is low "
                         f"({worst * 100:.0f}%); treat class areas as rough "
                         "estimates.")
    p1, p2 = r["p1"], r["p2"]
    pop_up = bool(p1 and p2 and p1["value"] and p2["value"]
                  and p2["value"] >= p1["value"])
    if a1[6] > 0 and pct(a1[6], a2[6]) < -20 and pop_up:
        flags.append("Urban area decreased while population did not; check "
                     "the input data for this location.")
    for name, code in (("Vegetation", 4), ("Crop Land", 8)):
        ch = pct(a1[code], a2[code])
        if max(a1[code], a2[code]) > 0.03 * area and (ch > 100 or ch < -60):
            flags.append(f"{name} changed by {ch:+.0f}%, which is unusually "
                         "large and may reflect different image dates or "
                         "sensors rather than real land change.")
    w = r["water"]
    moved = w["lost"] + w["gain"]
    if moved >= 0.005 * area and moved > w["kept"]:
        flags.append("Water extent fluctuates strongly between the two "
                     "years (more change than stable water); river and pond "
                     "changes are unreliable.")
    return flags


def insights(r):
    d = r["dec"]
    y1, y2 = r["years"][0], r["years"][-1]
    a1, a2 = r["a"][y1], r["a"][y2]
    area = math.pi * r["r_km"] ** 2
    out = ["RELIABILITY WARNING: " + f for f in reliability_flags(r)]
    out.append(f"Vegetation changed from {a1[4]:.{d}f} to {a2[4]:.{d}f} km2 "
               f"({pct(a1[4], a2[4]):+.1f}%); Scrub Land from {a1[7]:.{d}f} "
               f"to {a2[7]:.{d}f} km2; Crop Land from {a1[8]:.{d}f} to "
               f"{a2[8]:.{d}f} km2 ({pct(a1[8], a2[8]):+.1f}%); Open Land "
               f"from {a1[10]:.{d}f} to {a2[10]:.{d}f} km2.")
    out.append(f"Urban Area changed from {a1[6]:.{d}f} to {a2[6]:.{d}f} km2 "
               f"({pct(a1[6], a2[6]):+.1f}%); Rural Area (rural settlement) "
               f"from {a1[3]:.{d}f} to {a2[3]:.{d}f} km2.")
    b1, b2 = r["built"].get(y1), r["built"].get(y2)
    if b1 is not None and b2 is not None:
        out.append(f"Actual built surface (JRC GHSL) changed from "
                   f"{b1:.{d}f} to {b2:.{d}f} km2 ({pct(b1, b2):+.0f}%).")
    out.append(f"Pond / Lake (including river water) changed from "
               f"{a1[5]:.{d}f} to {a2[5]:.{d}f} km2; Aquaculture "
               f"(experimental) from {a1[9]:.{d}f} to {a2[9]:.{d}f} km2.")
    w = r["water"]
    if w["lost"] + w["gain"] >= 0.005 * area:
        out.append(f"About {w['lost']:.{d}f} km2 of former water is now "
                   f"land and {w['gain']:.{d}f} km2 of land is now water, "
                   "indicating channel, pond or shoreline shift.")
    else:
        out.append("Water bodies show no significant change in this period.")
    if r.get("osm_ok"):
        out.append(f"Main Road length {r['road_km']:.1f} km and Canal length "
                   f"{r['canal_km']:.1f} km inside the radius (current "
                   "OpenStreetMap, not time-specific).")
    else:
        out.append("Main Road and Canal could not be loaded from "
                   "OpenStreetMap this time (server did not respond); they "
                   "are missing from the map and statistics.")
    p1, p2 = r["p1"], r["p2"]
    if p1 and p2 and p1["value"] and p2["value"]:
        out.append(f"Modelled population estimate changed from "
                   f"{p1['value']:,.0f} ({p1['epoch']}) to {p2['value']:,.0f} "
                   f"({p2['epoch']}), {pct(p1['value'], p2['value']):+.1f}%.")
    g1, g2 = r["g1"], r["g2"]
    if g1 and g2:
        if g1["records"] == 0 and g2["records"] == 0:
            out.append("No GBIF animal records exist for these time windows, "
                       "so wildlife trends cannot be assessed (this does not "
                       "mean there is no wildlife).")
        else:
            out.append(f"Recorded animal species (GBIF): {g1['species']} "
                       f"around {y1} vs {g2['species']} around {y2}.")
            if g1["records"] and g2["records"] / g1["records"] > 3:
                out.append("The rise in animal records mainly reflects "
                           "growing citizen-science observation effort "
                           "(e.g. eBird, iNaturalist), not necessarily more "
                           "wildlife.")
    if a1[4] > 0 and pct(a1[4], a2[4]) < -5 and a2[6] > a1[6]:
        out.append("Vegetation loss together with urban growth suggests "
                   "habitat pressure on wildlife.")
    acc = r.get("acc")
    if acc:
        parts = ", ".join(f"{y}: {v['overall'] * 100:.0f}%"
                          for y, v in acc.items())
        out.append(f"Classification accuracy on held-out points ({parts}). "
                   "Each year's model learns from that year's own imagery "
                   "with WorldCover 2020 labels, so change figures are "
                   "conservative.")
    return out


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def chart_bytes(r):
    years = r["years"]
    shades = plt.cm.Greens(np.linspace(0.35, 0.95, len(years)))
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    bw = 0.8 / len(years)
    for j, y in enumerate(years):
        xs = [i - 0.4 + bw * (j + 0.5) for i in range(len(AREA_CODES))]
        ax.bar(xs, [r["a"][y][k] for k in AREA_CODES], bw, label=str(y),
               color=shades[j])
    ax.set_xticks(range(len(AREA_CODES)))
    ax.set_xticklabels([CLASSES[k][0] for k in AREA_CODES], rotation=20,
                       ha="right", fontsize=8)
    ax.set_ylabel("Area (km2)")
    ax.set_title("Land cover area by class")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def img_flow(data, width_cm):
    w, h = PILImage.open(io.BytesIO(data)).size
    return Image(io.BytesIO(data), width=width_cm * cm,
                 height=width_cm * cm * h / w)


def _page_mark(canvas, doc):
    """Diagonal watermark (behind content) + footer on every PDF page."""
    w, h = A4
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(BOTTLE))
    canvas.setFillAlpha(0.09)
    canvas.setFont("Helvetica-Bold", 34)
    for fy in (0.22, 0.50, 0.78):
        canvas.saveState()
        canvas.translate(w / 2, h * fy)
        canvas.rotate(40)
        canvas.drawCentredString(0, 0, SITE_NAME)
        canvas.restoreState()
    canvas.restoreState()
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(2 * cm, 1 * cm, f"{SITE_NAME}  |  LULC Change Report (Beta)")
    canvas.drawRightString(w - 2 * cm, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(r):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm,
                            rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="LULC Change Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=19, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"],
                        textColor=colors.HexColor(BOTTLE))
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)
    d = r["dec"]
    years = r["years"]
    y1, y2 = years[0], years[-1]

    def tbl(data, widths=None, swatches=None):
        t = Table(data, colWidths=widths, repeatRows=1)
        st_ = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BOTTLE)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#EFE8D4")]),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]
        if swatches:
            st_.append(("ALIGN", (1, 1), (1, -1), "LEFT"))
        for row, col in (swatches or {}).items():
            st_.append(("BACKGROUND", (0, row), (0, row), colors.HexColor(col)))
        t.setStyle(TableStyle(st_))
        return t

    title = (f"LULC Change Analysis: {r['lat']:.5f}, {r['lon']:.5f} | "
             f"{r['r_km']} km | {r['period']}")
    s = [Paragraph("Land Use / Land Cover Change Report (Beta)", h1),
         Paragraph(title, body),
         Paragraph(f"{SITE_NAME} | Report generated {r['created']} | Beta "
                   "version: indicative results, under validation", small),
         Spacer(1, 8),
         tbl([["Parameter", "Value"],
              ["Longitude / Latitude", f"{r['lon']:.5f} / {r['lat']:.5f}"],
              ["Selected radius", f"{r['r_km']} km"],
              ["Selected period", r["period"]],
              ["Selected image season", r["season"]],
              ["Years analysed", ", ".join(str(y) for y in years)],
              ["Landsat scenes used", " / ".join(str(r["scenes"][y]) for y in years)],
              ["Analysis resolution", f"{SCALE} m"],
              ["Accuracy (held-out points)",
               ", ".join(f"{y}: {v['overall'] * 100:.1f}% (kappa {v['kappa']:.2f})"
                         for y, v in r["acc"].items())],
              ["Report generated", r["created"]]],
             [6 * cm, 11 * cm])]

    s.append(Paragraph("1. Summary of findings", h2))
    for line in insights(r):
        s.append(Paragraph("&bull; " + line, body))

    s.append(KeepTogether([Paragraph("2. LULC classification maps", h2),
                           img_flow(r["lulc_map"], 17)]))

    s.append(Paragraph("3. LULC area statistics and change "
                       "(previous vs current)", h2))
    head = ["", "Class"] + [f"{y} km2" for y in years] + ["Change km2", "Change %"]
    rows, sw = [head], {}
    for k in AREA_CODES:
        a, b = r["a"][y1][k], r["a"][y2][k]
        rows.append(["", CLASSES[k][0]]
                    + [f"{r['a'][y][k]:.{d}f}" for y in years]
                    + [f"{b - a:+.{d}f}", f"{pct(a, b):+.1f}"])
        sw[len(rows) - 1] = CLASSES[k][1]
    for code, key in ((1, "road_km"), (2, "canal_km")):
        val = r.get(key)
        txt = f"{val:.1f}" if val is not None else "n/a"
        rows.append(["", CLASSES[code][0] + " (km)*"] + [txt] * len(years)
                    + ["n/a", "n/a"])
        sw[len(rows) - 1] = CLASSES[code][1]
    rest = (17 * cm - 0.5 * cm - 4.2 * cm) / (len(head) - 2)
    s.append(tbl(rows, [0.5 * cm, 4.2 * cm] + [rest] * (len(head) - 2), sw))
    s.append(Paragraph(
        "* Length in km from current OpenStreetMap data; roads and canals are "
        "not time-specific, so no change is computed.", small))
    bl = [f"{y}: {r['built'][y]:.{d}f} km2" for y in years
          if r["built"].get(y) is not None]
    if bl:
        s.append(Paragraph("Actual built surface (JRC GHSL): " + ", ".join(bl)
                           + ". Rural Area and Urban Area count whole 100 m "
                           "cells with at least 10% built surface, so they "
                           "are larger.", small))
    s.append(Spacer(1, 6))
    s.append(Image(chart_bytes(r), width=15 * cm, height=7.2 * cm))
    s.append(Paragraph("Mean NDVI: " + ", ".join(
        f"{r['ndvi'][y]:.3f} ({y})" for y in years) + ".", body))

    s.append(Paragraph("4. River path and water bodies", h2))
    w = r["water"]
    s.append(Paragraph(
        f"{y1} to {y2}: unchanged water {w['kept']:.{d}f} km2, water lost "
        f"(now land) {w['lost']:.{d}f} km2, new water {w['gain']:.{d}f} km2 "
        "(Pond / Lake and Aquaculture combined).", body))
    s.append(KeepTogether([img_flow(r["water_map"], 9.5)]))

    s.append(Paragraph("5. Population", h2))
    if r["p1"] and r["p2"]:
        s.append(tbl([["Epoch", "Estimated population"],
                      [str(r["p1"]["epoch"]), f"{r['p1']['value']:,.0f}"],
                      [str(r["p2"]["epoch"]), f"{r['p2']['value']:,.0f}"]],
                     [6 * cm, 6 * cm]))
        s.append(Paragraph(
            "Source: JRC Global Human Settlement Layer (GHS-POP), nearest "
            "5-year epoch. Values are modelled estimates.", small))
    else:
        s.append(Paragraph("Population data unavailable for this request.", body))

    s.append(Paragraph("6. Wildlife (indicative)", h2))
    if r["g1"] and r["g2"]:
        s.append(tbl([["Window", "Animal records", "Species"],
                      [f"{y1 - 2}-{y1 + 2}", str(r["g1"]["records"]), str(r["g1"]["species"])],
                      [f"{y2 - 2}-{y2 + 2}", str(r["g2"]["records"]), str(r["g2"]["species"])]],
                     [6 * cm, 5 * cm, 5 * cm]))
        extra = ""
        if r["g1"]["records"] == 0 and r["g2"]["records"] == 0:
            extra = ("No records were found in GBIF for these windows; this "
                     "does not mean there is no wildlife. ")
        s.append(Paragraph(
            extra + "Source: GBIF occurrence records. Counts reflect "
            "observation effort as much as real animal numbers, so read them "
            "together with the habitat change.", small))
    else:
        s.append(Paragraph("Wildlife records unavailable for this request.", body))

    s.append(Paragraph("7. How reliable is each class?", h2))
    s.append(Paragraph(
        "<b>Better supported:</b> Vegetation, Crop Land (Random Forest on "
        "Landsat), Pond / Lake (JRC Global Surface Water, permanent water of "
        "each year, independent of the chosen season), Urban Area (JRC "
        "GHSL).<br/><b>Moderate:</b> Scrub Land, Open Land, Rural Area (Rural "
        "vs Urban uses a density rule: Urban if at least 30% of the area "
        "within 500 m is built).<br/><b>Experimental:</b> Aquaculture (small "
        "permanent water clusters, not validated against field data).<br/>"
        "<b>Reference only:</b> Main Road and Canal come from current "
        "OpenStreetMap and are not time-specific.", body))

    s.append(Paragraph("8. Method and limitations", h2))
    s.append(Paragraph(
        "Land cover is classified with a Random Forest model (100 trees) "
        "trained on ESA WorldCover 2020 labels using cloud-masked Landsat "
        "5/7/8/9 Collection 2 Level-2 reflectance (median composite of the "
        "year before, the year, and the year after, restricted to the chosen "
        "image season; Landsat 8/9 adjusted to Landsat 5/7 equivalent values, "
        "Roy et al. 2016) plus NDVI, MNDWI and NDBI. A separate model is "
        "trained for each year on that year's own imagery, using WorldCover "
        "2020 labels as a proxy for stable land cover. This removes sensor "
        "and season bias between decades but assumes most land did not "
        "change, so real change is damped and change figures are "
        "conservative. Accuracy is measured on 30% held-out points of each "
        "year (a random split, so it is optimistic). The 30 m resolution "
        "cannot resolve narrow canals or small ponds. This is a beta, "
        "screening-level analysis, not field-verified.", body))

    doc.build(s, onFirstPage=_page_mark, onLaterPages=_page_mark)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Database (Supabase REST) with in-memory fallback
# --------------------------------------------------------------------------
def _sb():
    try:
        s = st.secrets["supabase"]
        return s["url"].rstrip("/"), s["key"]
    except Exception:
        return None


def db_enabled():
    return _sb() is not None


def _hdr(extra=None):
    _, key = _sb()
    h = {"apikey": key}
    if not key.startswith("sb_"):  # legacy JWT keys also go in Authorization
        h["Authorization"] = f"Bearer {key}"
    if extra:
        h.update(extra)
    return h


@st.cache_resource
def mem_store():
    return {}


def make_key(lat, lon, period, radius, season):
    """One unique record per (lat, lon, period, radius, season)."""
    raw = f"{APP_VERSION}|{lat:.5f}|{lon:.5f}|{period}|{radius}|{season}"
    return hashlib.sha256(raw.encode()).hexdigest()[:40]


def _fmt_time(iso):
    return iso[:16].replace("T", " ") + " UTC"


def load_record(key):
    """Saved report (as a view dict) or None."""
    if not db_enabled():
        return mem_store().get(key)
    url, _ = _sb()
    try:
        r = requests.get(f"{url}/rest/v1/analyses",
                         params={"cache_key": f"eq.{key}", "select": "*",
                                 "limit": 1},
                         headers=_hdr(), timeout=60)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        st.warning(f"Could not read saved reports: {e}")
        return None
    if not rows:
        return None
    row = rows[0]
    return {"key": key, "created": _fmt_time(row["created_at"]),
            "lat": row["lat"], "lon": row["lon"], "period": row["period"],
            "radius_km": row["radius_km"], "season": row["season"],
            "status": row["status"], "findings": row["findings"] or [],
            "lulc_map": base64.b64decode(row["lulc_map"]),
            "water_map": base64.b64decode(row["water_map"]),
            "pdf": base64.b64decode(row["pdf"]), "cached": True}


def save_record(view):
    """Persist a successful analysis. Duplicates are ignored by the database."""
    if not db_enabled():
        mem_store()[view["key"]] = dict(view, cached=True)
        return True
    url, _ = _sb()
    payload = {
        "cache_key": view["key"], "lat": view["lat"], "lon": view["lon"],
        "period": view["period"], "radius_km": view["radius_km"],
        "season": view["season"], "status": view["status"],
        "findings": view["findings"],
        "lulc_map": base64.b64encode(view["lulc_map"]).decode(),
        "water_map": base64.b64encode(view["water_map"]).decode(),
        "pdf": base64.b64encode(view["pdf"]).decode(),
    }
    try:
        r = requests.post(
            f"{url}/rest/v1/analyses", params={"on_conflict": "cache_key"},
            json=payload, timeout=120,
            headers=_hdr({"Content-Type": "application/json",
                          "Prefer": "resolution=ignore-duplicates,return=minimal"}))
        r.raise_for_status()
        return True
    except Exception as e:
        st.warning(f"The report could not be saved to the database: {e}")
        return False


def list_history(limit=20):
    if not db_enabled():
        rows = [dict(v) for v in mem_store().values()]
        rows.sort(key=lambda v: v["created"], reverse=True)
        return rows[:limit]
    url, _ = _sb()
    try:
        r = requests.get(
            f"{url}/rest/v1/analyses",
            params={"select": "cache_key,created_at,lat,lon,period,"
                              "radius_km,season,status",
                    "order": "created_at.desc", "limit": limit},
            headers=_hdr(), timeout=60)
        r.raise_for_status()
        return [{"key": x["cache_key"], "created": _fmt_time(x["created_at"]),
                 "lat": x["lat"], "lon": x["lon"], "period": x["period"],
                 "radius_km": x["radius_km"], "season": x["season"],
                 "status": x["status"]} for x in r.json()]
    except Exception as e:
        st.warning(f"Could not load report history: {e}")
        return []


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="LULC Change Analyzer (Beta) | www.tanmaysanyal.in",
                   page_icon="🛰️", layout="centered")

THEME_CSS = """
<style>
.stApp { background: #F5F0E1; }
.stMarkdown, .stMarkdown p, .stMarkdown li, label, h1, h2, h3,
[data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p
{ color: #1E2B25 !important; }
.block-container { padding-top: 2rem; padding-bottom: 4.5rem; }
div[data-testid="stForm"] { background: #EDE5CF; border: 1px solid #0B5D45;
                            border-radius: 12px; }
div[data-baseweb="input"], div[data-baseweb="select"] > div
{ background: #FFFDF6 !important; }
div[data-baseweb="input"] input, div[data-baseweb="select"] div
{ color: #1E2B25 !important; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button
{ background: #0B5D45; border: 1px solid #0B5D45; border-radius: 8px; }
div.stButton > button p, div[data-testid="stFormSubmitButton"] > button p,
div[data-testid="stDownloadButton"] > button p { color: #F5F0E1 !important; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover { background: #08473A; }
.site-banner { background: #0B5D45; border-radius: 12px; padding: 18px 22px;
               margin-bottom: 16px; }
.site-banner .t { color: #F5F0E1 !important; font-size: 1.9rem;
                  font-weight: 700; line-height: 1.2; }
.site-banner .beta { background: #F5F0E1; color: #0B5D45 !important;
                     font-size: 0.7rem; font-weight: 800; padding: 2px 9px;
                     border-radius: 999px; margin-left: 10px;
                     vertical-align: middle; letter-spacing: 1px; }
.site-banner .s { color: #E7DDBF !important; font-size: 0.95rem;
                  margin-top: 4px; }
.site-footer { position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
               background: #0B5D45; color: #F5F0E1; text-align: center;
               padding: 7px 10px; font-size: 0.85rem; letter-spacing: 0.3px; }
@media (max-width: 640px) {
  .site-banner { padding: 14px 14px; }
  .site-banner .t { font-size: 1.35rem; }
  .site-banner .s { font-size: 0.8rem; }
  .site-footer { font-size: 0.72rem; }
}
.stApp::before { content: ""; position: fixed; inset: 0; z-index: 999;
  pointer-events: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='380' height='230'><text x='24' y='150' transform='rotate(-25 190 115)' font-family='Arial' font-size='22' font-weight='bold' fill='rgb(11,93,69)' fill-opacity='0.09'>www.tanmaysanyal.in</text></svg>"); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="site-banner"><div class="t">🛰️ LULC Change Analyzer'
    '<span class="beta">BETA</span></div>'
    '<div class="s">Land use &amp; land cover change tool &nbsp;|&nbsp; '
    '<b>www.tanmaysanyal.in</b></div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="site-footer">www.tanmaysanyal.in &nbsp;·&nbsp; '
    'LULC Change Analyzer (Beta)</div>', unsafe_allow_html=True)
st.caption("Beta version: results are screening-level and still under testing.")
if not db_enabled():
    st.warning("Database is not connected: reports are kept only until the "
               "app restarts. Connect Supabase (see setup notes) to keep "
               "history permanently.")

DEFAULTS = {"in_lat": 23.07357, "in_lon": 88.53253,
            "in_period": list(PERIODS)[0], "in_radius": RADII_KM[0],
            "in_season": list(SEASONS)[0]}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("view", None)


def reset_after_download():
    """Runs when the PDF download button is clicked: reset the input form.
    Saved reports stay in the database / history."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["view"] = None


def render_view(view, ctx):
    st.subheader("Findings")
    if view.get("cached"):
        st.success(f"Saved report loaded (created {view['created']}); "
                   "no new analysis was run.")
    st.caption(f"{view['lat']:.5f}, {view['lon']:.5f} | {view['radius_km']} km "
               f"| {view['period']} | {view['season']} | Report ID "
               f"{view['key'][:8]}")
    prefix = "RELIABILITY WARNING: "
    for line in view["findings"]:
        if line.startswith(prefix):
            st.warning(line[len(prefix):])
        else:
            st.markdown("- " + line)
    st.image(view["lulc_map"], use_container_width=True)
    st.image(view["water_map"], use_container_width=True)
    st.download_button(
        "⬇️ Download PDF report", view["pdf"],
        file_name=f"LULC_{view['key'][:8]}.pdf", mime="application/pdf",
        type="primary", on_click=reset_after_download,
        key=f"dl_{ctx}_{view['key']}")
    st.caption("After the download the form resets. Your report stays saved "
               "under Previous reports.")


with st.form("inputs"):
    c1, c2 = st.columns(2)
    c1.number_input("Latitude", -90.0, 90.0, step=0.0001, format="%.5f",
                    key="in_lat")
    c2.number_input("Longitude", -180.0, 180.0, step=0.0001, format="%.5f",
                    key="in_lon")
    c3, c4 = st.columns(2)
    c3.selectbox("Period", list(PERIODS), key="in_period")
    c4.selectbox("Radius (km)", RADII_KM, key="in_radius")
    st.selectbox("Image season", list(SEASONS), key="in_season")
    go = st.form_submit_button("Analyze", type="primary")

if go:
    lat, lon = st.session_state["in_lat"], st.session_state["in_lon"]
    period, radius = st.session_state["in_period"], st.session_state["in_radius"]
    season_label = st.session_state["in_season"]
    key = make_key(lat, lon, period, radius, season_label)
    saved = load_record(key)
    if saved:
        st.session_state["view"] = saved
    else:
        try:
            init_ee()
            with st.status("Analyzing (this can take 2-6 minutes)...",
                           expanded=True) as status:
                res = run_analysis(lat, lon, radius, PERIODS[period], period,
                                   season_label, lambda m: st.write(m))
                pdf = build_pdf(res)
                status.update(label="Done", state="complete")
            view = {"key": key, "created": res["created"], "lat": lat,
                    "lon": lon, "period": period, "radius_km": radius,
                    "season": season_label, "status": "complete",
                    "findings": insights(res), "lulc_map": res["lulc_map"],
                    "water_map": res["water_map"], "pdf": pdf,
                    "cached": False}
            save_record(view)
            st.session_state["view"] = view
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if st.session_state.get("view"):
    render_view(st.session_state["view"], "main")

st.markdown("### Previous reports")
_rows = list_history()
if not _rows:
    st.caption("No saved reports yet.")
for _row in _rows:
    a, b = st.columns([4, 1])
    a.markdown(f"**{_row['created']}** · {_row['lat']:.4f}, {_row['lon']:.4f} "
               f"· {_row['radius_km']} km · {_row['period']} · "
               f"{_row['season']} · {_row['status']}")
    if b.button("Open", key=f"open_{_row['key']}"):
        v = load_record(_row["key"])
        if v:
            v["cached"] = True
            st.session_state["view"] = v
            st.rerun()
