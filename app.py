"""
LULC Change Analysis Tool Beta  (v2: publication-style maps)
-------------------------------------------------------
User gives latitude, longitude, a period and a radius.
Google Earth Engine (Landsat) -> land cover for each year -> change statistics,
framed maps on a basemap with roads/canals, population (GHSL), wildlife (GBIF),
and a downloadable PDF report.
"""
import datetime
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
from matplotlib.patches import Patch
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
BOTTLE = "#0B5D45"   # bottle green
BEIGE = "#F5F0E1"    # beige

CLASSES = {
    1: ("Water / pond / river", "#1f78b4"),
    2: ("Dense vegetation", "#1b7837"),
    3: ("Sparse vegetation / cropland", "#a6d96a"),
    4: ("Built-up / settlement", "#e6550d"),
    5: ("Bare land / other", "#f2e394"),
}
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
SEASONS = {
    "Nov-Feb (dry season, best for South Asia)": (11, 2),
    "Whole year": None,
}
RADII_KM = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
R_EARTH = 6378137.0


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def merc(lon, lat):
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


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
              .multiply(0.0000275).add(-0.2).updateMask(clear))
        return sr.copyProperties(img, ["system:time_start"])

    return f


def composite(year, region, season, win=1):
    """Cloud-free median composite of (year-win ... year+win)."""
    keys = sorted({k for y in range(year - win, year + win + 1)
                   for k in sensors_for(y)})
    merged = None
    for k in keys:
        col = (ee.ImageCollection(SENSORS[k][0])
               .filterBounds(region)
               .filterDate(f"{year - win}-01-01", f"{year + win + 1}-01-01"))
        if season:
            m0, m1 = season  # e.g. (11, 2) = Nov..Feb
            col = col.filter(ee.Filter.Or(
                ee.Filter.calendarRange(m0, 12, "month"),
                ee.Filter.calendarRange(1, m1, "month")))
        col = col.map(_prep(k))
        merged = col if merged is None else merged.merge(col)
    n = merged.size().getInfo()
    if n == 0:
        raise RuntimeError(
            f"No usable Landsat images for {year}. Try 'Whole year'.")
    return merged.median().clip(region), n


def classify(img):
    ndvi = img.normalizedDifference(["nir", "red"])
    mndwi = img.normalizedDifference(["green", "swir1"])
    ndbi = img.normalizedDifference(["swir1", "nir"])
    c = ee.Image(5)
    c = c.where(ndvi.gte(0.25).And(ndvi.lt(0.5)), 3)
    c = c.where(ndvi.gte(0.5), 2)
    c = c.where(ndvi.lt(0.25).And(ndbi.gt(0)), 4)
    c = c.where(mndwi.gt(0.1).And(ndvi.lt(0.3)), 1)
    c = c.rename("lulc").toInt().updateMask(img.select("red").mask())
    return c, ndvi.rename("ndvi")


def area_by(code_img, region, scale):
    r = (ee.Image.pixelArea().divide(1e6).addBands(code_img)
         .reduceRegion(reducer=ee.Reducer.sum().group(1, "c"),
                       geometry=region, scale=scale,
                       maxPixels=1e10, bestEffort=True).getInfo())
    return {int(g["c"]): g["sum"] for g in r.get("groups", [])}


def bbox(region):
    co = region.bounds().coordinates().getInfo()[0]
    xs = [p[0] for p in co]
    ys = [p[1] for p in co]
    return min(xs), min(ys), max(xs), max(ys)


def png(img, rect, vis, dim=1000):
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
def gbif(lat, lon, r_km, year):
    try:
        params = {"geometry": circle_wkt(lat, lon, r_km),
                  "year": f"{year - 2},{year + 2}", "kingdomKey": 1,
                  "hasCoordinate": "true", "limit": 0,
                  "facet": "SPECIES_KEY", "facetLimit": 1000}
        j = requests.get("https://api.gbif.org/v1/occurrence/search",
                         params=params, timeout=60).json()
        sp = len(j["facets"][0]["counts"]) if j.get("facets") else 0
        return {"records": j.get("count", 0), "species": sp}
    except Exception:
        return None


def osm_lines(lat, lon, r_km):
    """Major roads and canals/rivers (current OSM) in Web-Mercator coords."""
    out = {"major": [], "minor": [], "water": []}
    r = int(r_km * 1000)
    hw = ("motorway|trunk|primary" if r_km > 30
          else "motorway|trunk|primary|secondary")
    q = (f'[out:json][timeout:30];('
         f'way["highway"~"^({hw})$"](around:{r},{lat},{lon});'
         f'way["waterway"~"^(river|canal)$"](around:{r},{lat},{lon});'
         f');out geom;')
    try:
        j = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": q}, timeout=50).json()
    except Exception:
        return out
    for el in j.get("elements", []):
        g = el.get("geometry")
        if not g:
            continue
        xy = [merc(p["lon"], p["lat"]) for p in g]
        tags = el.get("tags", {})
        if "waterway" in tags:
            out["water"].append(xy)
        elif tags.get("highway") in ("motorway", "trunk", "primary"):
            out["major"].append(xy)
        else:
            out["minor"].append(xy)
    return out


# --------------------------------------------------------------------------
# Map drawing (publication style)
# --------------------------------------------------------------------------
def draw_map(panels, ext, lat, lon, r_km, osm, legend, note, ncols=2):
    """panels: list of (title, png_bytes). ext = (xmin, ymin, xmax, ymax)."""
    n = len(panels)
    ncols = 1 if n == 1 else ncols
    nrows = math.ceil(n / ncols)
    xmin, ymin, xmax, ymax = ext
    aspect = (ymax - ymin) / (xmax - xmin)
    pw = 5.2
    side = 2.7  # width reserved for the legend column
    figw, figh = pw * ncols + side, pw * aspect * nrows + 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(figw, figh), dpi=150,
                             squeeze=False)
    cx, cy = zip(*[merc(x, y) for x, y in circle_lonlat(lat, lon, r_km)])
    cos_lat = math.cos(math.radians(lat))
    width_km = (xmax - xmin) * cos_lat / 1000.0
    nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    bar_km = max([v for v in nice if v <= 0.3 * width_km] or [nice[0]])
    bar_frac = bar_km * 1000.0 / cos_lat / (xmax - xmin)

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
            # Esri Light Gray Canvas needs no API key (CARTO now requires one)
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
        arr = np.array(PILImage.open(io.BytesIO(data)).convert("RGBA"))
        arr = arr.astype(float)
        arr[..., 3] *= 0.85
        ax.imshow(arr.astype("uint8"), extent=(xmin, xmax, ymin, ymax),
                  origin="upper", interpolation="nearest", zorder=3)
        for key, col, lw in (("water", "#0b4ea2", 1.5),
                             ("minor", "#9e9e9e", 0.6),
                             ("major", "#d7191c", 1.6)):
            if osm and osm.get(key):
                ax.add_collection(LineCollection(osm[key], colors=col,
                                                 linewidths=lw, zorder=4))
        ax.plot(cx, cy, color="black", lw=1.1, ls="--", zorder=5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.8)
        ax.text(0.5, 0.5, SITE_NAME, transform=ax.transAxes, rotation=30,
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="#0B5D45", alpha=0.20, zorder=9)
        ax.text(0.04, 0.04, title, transform=ax.transAxes, zorder=10,
                fontsize=17 if len(title) <= 5 else 11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black", pad=4))
        ax.annotate("", xy=(0.93, 0.945), xytext=(0.93, 0.83),
                    xycoords="axes fraction", zorder=10,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
        ax.text(0.93, 0.955, "N", transform=ax.transAxes, ha="center",
                fontsize=10, fontweight="bold", zorder=10)
        ax.plot([0.94 - bar_frac, 0.94], [0.06, 0.06],
                transform=ax.transAxes, color="black", lw=3,
                solid_capstyle="butt", zorder=10)
        ax.text(0.94 - bar_frac / 2, 0.075, f"{bar_km:g} km",
                transform=ax.transAxes, ha="center", fontsize=8, zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5,
                          alpha=0.8))

    bottom = 0.8 / figh
    fig.tight_layout(rect=(0, bottom, 1 - side / figw, 1), w_pad=2.0, h_pad=1.5)
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


MAP_NOTE = (f"{SITE_NAME} | Land cover: Landsat (USGS/NASA), rule-based classification. "
            "Roads (red/grey) and canals/rivers (blue lines): OpenStreetMap "
            "contributors (current, for reference). Basemap: Esri, HERE, Garmin, "
            "OpenStreetMap contributors. Dashed line = study radius.")


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def run_analysis(lat, lon, r_km, years, season, progress):
    region = ee.Geometry.Point([lon, lat]).buffer(r_km * 1000)
    scale = 30 if r_km <= 10 else 60 if r_km <= 30 else 100 if r_km <= 60 else 150
    win = 1 if r_km <= 30 else 0  # one-year window for very large areas
    w, s, e, n = bbox(region)
    rect = ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)
    x0, y0 = merc(w, s)
    x1, y1 = merc(e, n)
    ext = (x0, y0, x1, y1)

    vis = {"min": 1, "max": 5, "palette": PALETTE}
    classes, areas, ndvi_mean, scenes, pngs = {}, {}, {}, {}, {}
    for y in years:
        progress(f"Processing {y}: composite, classification, statistics...")
        img, k = composite(y, region, season, win)
        c, nd = classify(img)
        classes[y] = c
        scenes[y] = k
        a = area_by(c, region, scale)
        for code in CLASSES:
            a.setdefault(code, 0.0)
        areas[y] = a
        ndvi_mean[y] = nd.reduceRegion(
            ee.Reducer.mean(), region, scale, maxPixels=1e10,
            bestEffort=True).get("ndvi").getInfo()
        pngs[y] = png(c, rect, vis)

    first, last = years[0], years[-1]
    progress("Measuring transitions and river/water change...")
    trans = area_by(classes[first].multiply(10).add(classes[last]),
                    region, scale)
    water = dict(
        lost=sum(v for k, v in trans.items() if k // 10 == 1 and k % 10 != 1),
        gain=sum(v for k, v in trans.items() if k // 10 != 1 and k % 10 == 1),
        kept=trans.get(11, 0.0))
    w1, w2 = classes[first].eq(1), classes[last].eq(1)
    wchg = (ee.Image(0).where(w1.And(w2.Not()), 1)
            .where(w1.Not().And(w2), 2).where(w1.And(w2), 3))
    wchg = wchg.updateMask(wchg.gt(0)).clip(region)
    water_png = png(wchg, rect, {"min": 1, "max": 3,
                                 "palette": ["d7191c", "33a02c", "1f78b4"]})

    progress("Fetching roads/canals, population and wildlife data...")
    osm = osm_lines(lat, lon, r_km)
    pop1, pop2 = population(first, region), population(last, region)
    g1, g2 = gbif(lat, lon, r_km, first), gbif(lat, lon, r_km, last)

    progress("Drawing maps...")
    road_items = [Line2D([0], [0], color="#d7191c", lw=1.8, label="Main road"),
                  Line2D([0], [0], color="#9e9e9e", lw=1.0, label="Other road"),
                  Line2D([0], [0], color="#0b4ea2", lw=1.8, label="Canal / river"),
                  Line2D([0], [0], color="black", lw=1.1, ls="--", label="Study radius")]
    legend = [Patch(facecolor=CLASSES[k][1], edgecolor="black", label=CLASSES[k][0])
              for k in sorted(CLASSES)] + road_items
    lulc_map = draw_map([(str(y), pngs[y]) for y in years], ext, lat, lon,
                        r_km, osm, legend, MAP_NOTE, ncols=2)
    wlegend = [Patch(facecolor="#1f78b4", edgecolor="black", label="Unchanged water"),
               Patch(facecolor="#d7191c", edgecolor="black", label="Water lost"),
               Patch(facecolor="#33a02c", edgecolor="black", label="New water")]
    water_map = draw_map([(f"{first}-{last}", water_png)], ext, lat, lon,
                         r_km, osm, wlegend, MAP_NOTE)

    dec = 2 if r_km <= 10 else 1
    return dict(lat=lat, lon=lon, r_km=r_km, years=years, scale=scale,
                season=("Nov-Feb" if season else "Whole year"), dec=dec,
                scenes=scenes, a=areas, trans=trans, ndvi=ndvi_mean,
                water=water, lulc_map=lulc_map, water_map=water_map,
                p1=pop1, p2=pop2, g1=g1, g2=g2,
                created=datetime.date.today().isoformat())


def pct(a, b):
    return (b - a) / a * 100.0 if a else 0.0


def insights(r):
    d = r["dec"]
    y1, y2 = r["years"][0], r["years"][-1]
    a1, a2 = r["a"][y1], r["a"][y2]
    area = math.pi * r["r_km"] ** 2
    out = []
    v1, v2 = a1[2] + a1[3], a2[2] + a2[3]
    out.append(f"Vegetation (dense + sparse) changed from {v1:.{d}f} km2 to "
               f"{v2:.{d}f} km2 ({pct(v1, v2):+.1f}%). Dense vegetation "
               f"changed by {pct(a1[2], a2[2]):+.1f}%.")
    out.append(f"Built-up area changed from {a1[4]:.{d}f} km2 to "
               f"{a2[4]:.{d}f} km2 ({a2[4] - a1[4]:+.{d}f} km2, "
               f"{pct(a1[4], a2[4]):+.0f}%).")
    w = r["water"]
    if w["lost"] + w["gain"] >= 0.005 * area:
        out.append(f"Water surface changed from {a1[1]:.{d}f} to "
                   f"{a2[1]:.{d}f} km2. About {w['lost']:.{d}f} km2 of former "
                   f"water is now land and {w['gain']:.{d}f} km2 of land is "
                   f"now water, indicating channel, pond or shoreline shift.")
    else:
        out.append("Water bodies show no significant change in this period.")
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
    if a1[2] - a2[2] > 0 and a2[4] > a1[4]:
        out.append("Dense vegetation loss together with built-up growth "
                   "suggests habitat fragmentation pressure on wildlife.")
    if abs(pct(a1[2], a2[2])) > 30:
        out.append("Note: a large change in dense vegetation can partly "
                   "reflect differences in image dates and sensors; verify "
                   "with local knowledge.")
    return out


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def chart_bytes(r):
    codes = sorted(CLASSES)
    years = r["years"]
    shades = plt.cm.Greens(np.linspace(0.35, 0.95, len(years)))
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    bw = 0.8 / len(years)
    for j, y in enumerate(years):
        xs = [i - 0.4 + bw * (j + 0.5) for i in range(len(codes))]
        ax.bar(xs, [r["a"][y][k] for k in codes], bw, label=str(y),
               color=shades[j])
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([CLASSES[k][0] for k in codes], rotation=20,
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
    canvas.setFillColor(colors.HexColor("#0B5D45"))
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
    canvas.drawString(2 * cm, 1 * cm, f"{SITE_NAME}  |  LULC Change Report")
    canvas.drawRightString(w - 2 * cm, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(r):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm,
                            rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="LULC Change Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"],
                        textColor=colors.HexColor("#0B5D45"))
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)
    d = r["dec"]
    years = r["years"]
    y1, y2 = years[0], years[-1]

    def tbl(data, widths=None):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B5D45")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#EFE8D4")]),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        return t

    s = [Paragraph("Land Use / Land Cover Change Report", h1),
         Paragraph(f"{SITE_NAME} | generated {r['created']}", small),
         Spacer(1, 8),
         tbl([["Parameter", "Value"],
              ["Latitude / Longitude", f"{r['lat']:.5f} / {r['lon']:.5f}"],
              ["Radius", f"{r['r_km']} km"],
              ["Years analysed", ", ".join(str(y) for y in years)],
              ["Image season", r["season"]],
              ["Landsat scenes used", " / ".join(str(r["scenes"][y]) for y in years)],
              ["Analysis resolution", f"{r['scale']} m"]],
             [6 * cm, 11 * cm])]

    s.append(Paragraph("1. Summary of findings", h2))
    for line in insights(r):
        s.append(Paragraph("&bull; " + line, body))

    s.append(KeepTogether([Paragraph("2. Land cover maps", h2),
                           img_flow(r["lulc_map"], 17)]))

    s.append(Paragraph("3. Land cover change", h2))
    head = ["Class"] + [f"{y} km2" for y in years] + ["Change km2", "Change %"]
    rows = [head]
    for k in sorted(CLASSES):
        a, b = r["a"][y1][k], r["a"][y2][k]
        rows.append([CLASSES[k][0]] + [f"{r['a'][y][k]:.{d}f}" for y in years]
                    + [f"{b - a:+.{d}f}", f"{pct(a, b):+.1f}"])
    first_w = 5.2 * cm
    rest = (17 * cm - first_w) / (len(head) - 1)
    s.append(tbl(rows, [first_w] + [rest] * (len(head) - 1)))
    s.append(Spacer(1, 6))
    s.append(Image(chart_bytes(r), width=15 * cm, height=7.2 * cm))
    s.append(Paragraph("Mean NDVI: " + ", ".join(
        f"{r['ndvi'][y]:.3f} ({y})" for y in years) + ".", body))

    s.append(Paragraph("4. River path and water bodies", h2))
    w = r["water"]
    s.append(Paragraph(
        f"{y1} to {y2}: unchanged water {w['kept']:.{d}f} km2, water lost "
        f"(now land) {w['lost']:.{d}f} km2, new water {w['gain']:.{d}f} km2.",
        body))
    s.append(KeepTogether([img_flow(r["water_map"], 9.5)]))

    s.append(Paragraph("5. Population", h2))
    if r["p1"] and r["p2"]:
        s.append(tbl([["Epoch", "Estimated population"],
                      [str(r["p1"]["epoch"]), f"{r['p1']['value']:,.0f}"],
                      [str(r["p2"]["epoch"]), f"{r['p2']['value']:,.0f}"]],
                     [6 * cm, 6 * cm]))
        s.append(Paragraph(
            "Source: JRC Global Human Settlement Layer (GHS-POP), nearest "
            "5-year epoch. Values are modelled estimates; uncertainty is high "
            "at small radii.", small))
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
            "together with the habitat (vegetation and water) change.", small))
    else:
        s.append(Paragraph("Wildlife records unavailable for this request.", body))

    s.append(Paragraph("7. Method and limitations", h2))
    s.append(Paragraph(
        "Land cover is derived from cloud-masked Landsat 5/7/8/9 Collection 2 "
        "Level-2 surface reflectance (median composite of the year before, "
        "the year, and the year after, restricted to the chosen season). "
        "Classes use index thresholds: NDVI (vegetation), MNDWI (water) and "
        "NDBI (built-up). This is a rule-based classification at 30 m "
        "resolution, not field-verified; accuracy is lower for early years, "
        "small features (narrow canals, small ponds), and areas with heavy "
        "cloud or seasonal flooding. Roads and canals drawn on the maps come "
        "from current OpenStreetMap data and are shown for orientation only. "
        "Use results as a screening-level analysis.", body))

    doc.build(s, onFirstPage=_page_mark, onLaterPages=_page_mark)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="LULC Change Analyzer | www.tanmaysanyal.in",
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
.site-banner .s { color: #E7DDBF !important; font-size: 0.95rem;
                  margin-top: 4px; }
.site-footer { position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
               background: #0B5D45; color: #F5F0E1; text-align: center;
               padding: 7px 10px; font-size: 0.85rem; letter-spacing: 0.3px; }
.stApp::before { content: ""; position: fixed; inset: 0; z-index: 999;
  pointer-events: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='380' height='230'><text x='24' y='150' transform='rotate(-25 190 115)' font-family='Arial' font-size='22' font-weight='bold' fill='rgb(11,93,69)' fill-opacity='0.09'>www.tanmaysanyal.in</text></svg>"); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="site-banner"><div class="t">🛰️ LULC Change Analyzer</div>'
    '<div class="s">Land use &amp; land cover change tool &nbsp;|&nbsp; '
    '<b>www.tanmaysanyal.in</b></div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="site-footer">www.tanmaysanyal.in &nbsp;·&nbsp; '
    'LULC Change Analyzer</div>', unsafe_allow_html=True)

with st.form("inputs"):
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, 23.07357, format="%.5f")
    lon = c2.number_input("Longitude", -180.0, 180.0, 88.53253, format="%.5f")
    c3, c4 = st.columns(2)
    period = c3.selectbox("Period", list(PERIODS))
    radius = c4.selectbox("Radius (km)", RADII_KM)
    season_label = st.selectbox("Image season", list(SEASONS))
    go = st.form_submit_button("Analyze", type="primary")

if go:
    try:
        init_ee()
        years = PERIODS[period]
        with st.status("Analyzing (this can take 1-5 minutes)...",
                       expanded=True) as status:
            res = run_analysis(lat, lon, radius, years, SEASONS[season_label],
                               lambda m: st.write(m))
            res["pdf"] = build_pdf(res)
            status.update(label="Done", state="complete")
        st.session_state["res"] = res
    except Exception as e:
        st.error(f"Something went wrong: {e}")

res = st.session_state.get("res")
if res:
    st.subheader("Findings")
    for line in insights(res):
        st.markdown(f"- {line}")
    st.image(res["lulc_map"], use_container_width=True)
    st.image(res["water_map"], use_container_width=True)
    st.download_button("⬇️ Download PDF report", res["pdf"],
                       file_name=f"LULC_{res['years'][0]}_{res['years'][-1]}.pdf",
                       mime="application/pdf", type="primary")
