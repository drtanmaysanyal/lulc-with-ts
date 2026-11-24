"""
LULC Change Analysis Tool
-------------------------
User gives latitude, longitude, a 10-year period and a radius.
The app uses Google Earth Engine (Landsat) to classify land cover for both
years, measures the change (vegetation, built-up, water/river path),
adds population (GHSL) and wildlife records (GBIF), and builds a PDF report.
"""
import io
import json
import math
import datetime

import ee
import requests
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
SITE_NAME = "tanmoychannel.in"

CLASSES = {
    1: ("Water", "#1f78b4"),
    2: ("Dense vegetation", "#1b7837"),
    3: ("Sparse vegetation / cropland", "#a6d96a"),
    4: ("Built-up", "#d7191c"),
    5: ("Bare land / other", "#e0c9a6"),
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
    "1990 to 2000": (1990, 2000),
    "2000 to 2010": (2000, 2010),
    "2010 to 2020": (2010, 2020),
}
RADII_KM = [1, 5, 10, 50]


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
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        sr = (img.select(src, NAMES).multiply(0.0000275).add(-0.2)
              .updateMask(clear))
        return sr.copyProperties(img, ["system:time_start"])

    return f


def composite(year, region):
    """Cloud-free median composite of (year-1 ... year+1)."""
    keys = sorted({k for y in (year - 1, year, year + 1)
                   for k in sensors_for(y)})
    merged = None
    for k in keys:
        col = (ee.ImageCollection(SENSORS[k][0])
               .filterBounds(region)
               .filterDate(f"{year - 1}-01-01", f"{year + 2}-01-01")
               .map(_prep(k)))
        merged = col if merged is None else merged.merge(col)
    n = merged.size().getInfo()
    if n == 0:
        raise RuntimeError(f"No Landsat images found for {year}.")
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
    """Area (km2) per integer code in a one-band image."""
    r = (ee.Image.pixelArea().divide(1e6).addBands(code_img)
         .reduceRegion(reducer=ee.Reducer.sum().group(1, "c"),
                       geometry=region, scale=scale,
                       maxPixels=1e10, bestEffort=True).getInfo())
    return {int(g["c"]): g["sum"] for g in r.get("groups", [])}


def png(img, region, vis):
    url = img.visualize(**vis).getThumbURL(
        {"region": region.bounds(), "dimensions": 600, "format": "png"})
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def population(year, region):
    try:
        epoch = min(max(int(round(year / 5.0)) * 5, 1975), 2030)
        img = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{epoch}").select(
            "population_count")
        v = img.reduceRegion(ee.Reducer.sum(), region, 100,
                             maxPixels=1e10, bestEffort=True
                             ).get("population_count").getInfo()
        return {"epoch": epoch, "value": v}
    except Exception:
        return None


# --------------------------------------------------------------------------
# GBIF (wildlife records)
# --------------------------------------------------------------------------
def circle_wkt(lat, lon, r_km, n=48):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        dlat = (r_km / 111.32) * math.sin(a)
        dlon = (r_km / (111.32 * math.cos(math.radians(lat)))) * math.cos(a)
        pts.append((lon + dlon, lat + dlat))
    pts.append(pts[0])
    return "POLYGON((" + ",".join(f"{x:.5f} {y:.5f}" for x, y in pts) + "))"


def gbif(lat, lon, r_km, year):
    try:
        params = {
            "geometry": circle_wkt(lat, lon, r_km),
            "year": f"{year - 2},{year + 2}",
            "kingdomKey": 1,  # Animalia
            "hasCoordinate": "true",
            "limit": 0,
            "facet": "SPECIES_KEY",
            "facetLimit": 1000,
        }
        j = requests.get("https://api.gbif.org/v1/occurrence/search",
                         params=params, timeout=60).json()
        sp = len(j["facets"][0]["counts"]) if j.get("facets") else 0
        return {"records": j.get("count", 0), "species": sp}
    except Exception:
        return None


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def run_analysis(lat, lon, r_km, y1, y2, progress):
    region = ee.Geometry.Point([lon, lat]).buffer(r_km * 1000)
    scale = 30 if r_km <= 10 else 100

    progress("Building cloud-free composites...")
    img1, n1 = composite(y1, region)
    img2, n2 = composite(y2, region)

    progress("Classifying land cover...")
    c1, ndvi1 = classify(img1)
    c2, ndvi2 = classify(img2)

    progress("Measuring areas...")
    a1 = area_by(c1, region, scale)
    a2 = area_by(c2, region, scale)
    for d in (a1, a2):
        for k in CLASSES:
            d.setdefault(k, 0.0)

    trans = area_by(c1.multiply(10).add(c2), region, scale)
    water_lost = sum(v for k, v in trans.items() if k // 10 == 1 and k % 10 != 1)
    water_gain = sum(v for k, v in trans.items() if k // 10 != 1 and k % 10 == 1)
    water_kept = trans.get(11, 0.0)

    def mean_ndvi(n):
        v = n.reduceRegion(ee.Reducer.mean(), region, scale, maxPixels=1e10,
                           bestEffort=True).get("ndvi").getInfo()
        return v

    progress("Computing vegetation index...")
    nd1, nd2 = mean_ndvi(ndvi1), mean_ndvi(ndvi2)

    progress("Rendering maps...")
    vis = {"min": 1, "max": 5, "palette": PALETTE}
    map1 = png(c1, region, vis)
    map2 = png(c2, region, vis)
    w1, w2 = c1.eq(1), c2.eq(1)
    wchg = (ee.Image(0).where(w1.And(w2.Not()), 1)
            .where(w1.Not().And(w2), 2).where(w1.And(w2), 3))
    wchg = wchg.updateMask(wchg.gt(0)).clip(region)
    water_map = png(wchg, region, {"min": 1, "max": 3,
                                   "palette": ["d7191c", "33a02c", "1f78b4"]})

    progress("Fetching population and wildlife data...")
    p1, p2 = population(y1, region), population(y2, region)
    g1, g2 = gbif(lat, lon, r_km, y1), gbif(lat, lon, r_km, y2)

    return dict(lat=lat, lon=lon, r_km=r_km, y1=y1, y2=y2, scale=scale,
                n1=n1, n2=n2, a1=a1, a2=a2, trans=trans, nd1=nd1, nd2=nd2,
                water=dict(lost=water_lost, gain=water_gain, kept=water_kept),
                map1=map1, map2=map2, water_map=water_map,
                p1=p1, p2=p2, g1=g1, g2=g2,
                created=datetime.date.today().isoformat())


def pct(a, b):
    return (b - a) / a * 100.0 if a else 0.0


def insights(r):
    a1, a2 = r["a1"], r["a2"]
    out = []
    v1, v2 = a1[2] + a1[3], a2[2] + a2[3]
    out.append(
        f"Vegetation cover changed from {v1:.1f} km2 to {v2:.1f} km2 "
        f"({pct(v1, v2):+.1f}%). Dense vegetation changed by "
        f"{pct(a1[2], a2[2]):+.1f}%.")
    out.append(
        f"Built-up area changed from {a1[4]:.1f} km2 to {a2[4]:.1f} km2 "
        f"({a2[4] - a1[4]:+.1f} km2).")
    w = r["water"]
    out.append(
        f"Water surface changed from {a1[1]:.1f} km2 to {a2[1]:.1f} km2. "
        f"About {w['lost']:.1f} km2 of former water is now land and "
        f"{w['gain']:.1f} km2 of land is now water, which indicates "
        f"channel or shoreline shift.")
    if r["p1"] and r["p2"] and r["p1"]["value"] and r["p2"]["value"]:
        out.append(
            f"Estimated population changed from {r['p1']['value']:,.0f} "
            f"({r['p1']['epoch']}) to {r['p2']['value']:,.0f} "
            f"({r['p2']['epoch']}), i.e. {pct(r['p1']['value'], r['p2']['value']):+.1f}%.")
    if r["g1"] and r["g2"]:
        out.append(
            f"Recorded animal species: {r['g1']['species']} around {r['y1']} "
            f"vs {r['g2']['species']} around {r['y2']} (GBIF).")
    loss = a1[2] - a2[2]
    if loss > 0 and a2[4] > a1[4]:
        out.append("Dense vegetation loss together with built-up growth "
                   "suggests habitat fragmentation pressure on wildlife.")
    return out


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def chart_bytes(r):
    labels = [CLASSES[k][0] for k in sorted(CLASSES)]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    ax.bar([i - 0.2 for i in x], [r["a1"][k] for k in sorted(CLASSES)], 0.4,
           label=str(r["y1"]), color="#7f8fa6")
    ax.bar([i + 0.2 for i in x], [r["a2"][k] for k in sorted(CLASSES)], 0.4,
           label=str(r["y2"]), color="#273c75")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
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


def build_pdf(r):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm,
                            rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm,
                            title="LULC Change Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#273c75"))
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)

    def tbl(data, widths=None):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#273c75")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f1f2f6")]),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        return t

    s = []
    s.append(Paragraph("Land Use / Land Cover Change Report", h1))
    s.append(Paragraph(f"{SITE_NAME} | generated {r['created']}", small))
    s.append(Spacer(1, 8))
    s.append(tbl([["Parameter", "Value"],
                  ["Latitude / Longitude", f"{r['lat']:.5f} / {r['lon']:.5f}"],
                  ["Radius", f"{r['r_km']} km"],
                  ["Period", f"{r['y1']} to {r['y2']}"],
                  ["Landsat scenes used", f"{r['n1']} / {r['n2']}"],
                  ["Analysis resolution", f"{r['scale']} m"]],
                 [6 * cm, 10 * cm]))

    s.append(Paragraph("1. Summary of findings", h2))
    for line in insights(r):
        s.append(Paragraph("&bull; " + line, body))

    s.append(Paragraph("2. Land cover change", h2))
    rows = [["Class", f"{r['y1']} km2", f"{r['y2']} km2", "Change km2", "Change %"]]
    for k in sorted(CLASSES):
        a, b = r["a1"][k], r["a2"][k]
        rows.append([CLASSES[k][0], f"{a:.2f}", f"{b:.2f}", f"{b - a:+.2f}",
                     f"{pct(a, b):+.1f}"])
    s.append(tbl(rows, [6 * cm, 2.5 * cm, 2.5 * cm, 2.6 * cm, 2.4 * cm]))
    s.append(Spacer(1, 6))
    s.append(Image(chart_bytes(r), width=15 * cm, height=7.2 * cm))
    s.append(Paragraph(f"Mean NDVI: {r['nd1']:.3f} ({r['y1']}) to "
                       f"{r['nd2']:.3f} ({r['y2']}).", body))

    maps = Table([[Image(io.BytesIO(r["map1"]), 7.6 * cm, 7.6 * cm),
                   Image(io.BytesIO(r["map2"]), 7.6 * cm, 7.6 * cm)],
                  [f"Land cover {r['y1']}", f"Land cover {r['y2']}"]])
    maps.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                              ("FONTSIZE", (0, 1), (-1, 1), 8)]))
    s.append(Spacer(1, 6))
    s.append(maps)
    leg = "  ".join(f'<font color="{CLASSES[k][1]}">&#9632;</font> {CLASSES[k][0]}'
                    for k in sorted(CLASSES))
    s.append(Paragraph(leg, small))

    s.append(Paragraph("3. River path and water bodies", h2))
    w = r["water"]
    s.append(Paragraph(
        f"Unchanged water: {w['kept']:.2f} km2. Water lost (now land): "
        f"{w['lost']:.2f} km2. New water: {w['gain']:.2f} km2.", body))
    s.append(Image(io.BytesIO(r["water_map"]), 8 * cm, 8 * cm))
    s.append(Paragraph('<font color="#1f78b4">&#9632;</font> unchanged &nbsp; '
                       '<font color="#d7191c">&#9632;</font> lost &nbsp; '
                       '<font color="#33a02c">&#9632;</font> new', small))

    s.append(Paragraph("4. Population", h2))
    if r["p1"] and r["p2"]:
        s.append(tbl([["Epoch", "Estimated population"],
                      [str(r["p1"]["epoch"]), f"{r['p1']['value']:,.0f}"],
                      [str(r["p2"]["epoch"]), f"{r['p2']['value']:,.0f}"]],
                     [6 * cm, 6 * cm]))
        s.append(Paragraph("Source: JRC Global Human Settlement Layer "
                           "(GHS-POP), nearest 5-year epoch.", small))
    else:
        s.append(Paragraph("Population data unavailable for this request.", body))

    s.append(Paragraph("5. Wildlife (indicative)", h2))
    if r["g1"] and r["g2"]:
        s.append(tbl([["Window", "Animal records", "Species"],
                      [f"{r['y1'] - 2}-{r['y1'] + 2}", str(r["g1"]["records"]), str(r["g1"]["species"])],
                      [f"{r['y2'] - 2}-{r['y2'] + 2}", str(r["g2"]["records"]), str(r["g2"]["species"])]],
                     [6 * cm, 5 * cm, 5 * cm]))
        s.append(Paragraph(
            "Source: GBIF occurrence records. These reflect observation "
            "effort as much as real animal numbers, so read them together "
            "with the habitat (vegetation and water) change above.", small))
    else:
        s.append(Paragraph("Wildlife records unavailable for this request.", body))

    s.append(Paragraph("6. Method and limitations", h2))
    s.append(Paragraph(
        "Land cover is derived from cloud-masked Landsat 5/7/8/9 Collection 2 "
        "Level-2 surface reflectance (median of the year before, the year, and "
        "the year after). Classes use index thresholds: NDVI (vegetation), "
        "MNDWI (water) and NDBI (built-up). This is a rule-based "
        "classification, not field-verified; accuracy is lower for early "
        "years, small features, and areas with heavy cloud or seasonal "
        "flooding. Use results as a screening-level analysis.", body))

    doc.build(s)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="LULC Change Analyzer", page_icon="🛰️",
                   layout="centered")
st.title("🛰️ LULC Change Analyzer")
st.caption("Land use and land cover change over 10 years for any location.")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, 22.5726, format="%.5f")
    lon = c2.number_input("Longitude", -180.0, 180.0, 88.3639, format="%.5f")
    c3, c4 = st.columns(2)
    period = c3.selectbox("Period", list(PERIODS))
    radius = c4.selectbox("Radius (km)", RADII_KM)
    go = st.form_submit_button("Analyze", type="primary")

if go:
    try:
        init_ee()
        y1, y2 = PERIODS[period]
        with st.status("Analyzing...", expanded=True) as status:
            res = run_analysis(lat, lon, radius, y1, y2,
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
    st.image([res["map1"], res["map2"]],
             caption=[f"Land cover {res['y1']}", f"Land cover {res['y2']}"])
    st.image(res["water_map"], caption="Water change (blue unchanged, red lost, green new)")
    st.download_button("⬇️ Download PDF report", res["pdf"],
                       file_name=f"LULC_{res['y1']}_{res['y2']}.pdf",
                       mime="application/pdf", type="primary")
