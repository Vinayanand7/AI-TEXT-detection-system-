import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import math

W, H = letter

# ══════════════════════════════════════════════════════════════
# THEMES
# ══════════════════════════════════════════════════════════════
THEMES = {
    "dark": {
        "bg":       "#0a0c10",
        "surface":  "#111318",
        "card":     "#181c24",
        "card_alt": "#13161e",
        "border":   "#252a36",
        "text":     "#e8eaf0",
        "muted":    "#6b7280",
        "accent":   "#00e5a0",
        "danger":   "#ff4466",
        "warn":     "#ffaa00",
        "blue":     "#4d8fff",
        "header_bar": "#111318",
        "badge_ai":   "#3d0f18",
        "badge_warn": "#3d2a00",
        "badge_ok":   "#003d2a",
    },
    "light": {
        "bg":       "#f4f6fa",
        "surface":  "#ffffff",
        "card":     "#ffffff",
        "card_alt": "#f0f2f8",
        "border":   "#d1d5e0",
        "text":     "#1a1d2e",
        "muted":    "#6b7280",
        "accent":   "#0a8f5c",
        "danger":   "#d9294a",
        "warn":     "#c47a00",
        "blue":     "#1a5fe0",
        "header_bar": "#1a1d2e",
        "badge_ai":   "#fde8ec",
        "badge_warn": "#fff3d6",
        "badge_ok":   "#d6f5ea",
    },
    "ocean": {
        "bg":       "#060d1a",
        "surface":  "#0b1626",
        "card":     "#0f1e35",
        "card_alt": "#0a1828",
        "border":   "#1a2f4a",
        "text":     "#cce0ff",
        "muted":    "#5a7a9e",
        "accent":   "#00b4d8",
        "danger":   "#ff5577",
        "warn":     "#ffa726",
        "blue":     "#4fc3f7",
        "header_bar": "#0b1626",
        "badge_ai":   "#2d0a10",
        "badge_warn": "#2d1a00",
        "badge_ok":   "#00232d",
    },
}

def get_theme(name="dark"):
    t = THEMES.get(name, THEMES["dark"])
    return {k: colors.HexColor(v) for k, v in t.items()}

def score_color(score, T):
    if score > 0.65: return T["danger"]
    if score > 0.40: return T["warn"]
    return T["accent"]

def verdict_text(score):
    if score > 0.65: return "LIKELY AI-GENERATED"
    if score > 0.40: return "UNCERTAIN"
    return "LIKELY HUMAN-WRITTEN"

# ══════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════
def rrect(c, x, y, w, h, r=8, fill=None, stroke=None, stroke_width=0.8):
    c.saveState()
    if fill:
        c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(stroke_width)
    c.roundRect(x, y, w, h, r,
                fill=1 if fill else 0,
                stroke=1 if stroke else 0)
    c.restoreState()

def progress_bar(c, x, y, w, h, pct, fill_col, track_col, r=3):
    rrect(c, x, y, w, h, r=r, fill=track_col)
    filled = w * max(0.0, min(float(pct), 1.0))
    if filled > 0:
        rrect(c, x, y, filled, h, r=r, fill=fill_col)

def draw_arc_donut(c, cx, cy, radius, score, score_col, track_col, line_w=10):
    # Track
    c.saveState()
    c.setStrokeColor(track_col)
    c.setLineWidth(line_w)
    c.circle(cx, cy, radius, fill=0, stroke=1)
    c.restoreState()
    # Fill arc
    c.saveState()
    c.setStrokeColor(score_col)
    c.setLineWidth(line_w)
    angle = score * 360
    if angle > 0:
        c.arc(cx - radius, cy - radius, cx + radius, cy + radius,
              startAng=90 - angle, extent=angle)
    c.restoreState()

def centered_string(c, text, font, size, cx, y, col):
    c.setFont(font, size)
    c.setFillColor(col)
    w = c.stringWidth(text, font, size)
    c.drawString(cx - w / 2, y, text)

def page_bg(c, T):
    c.setFillColor(T["bg"])
    c.rect(0, 0, W, H, fill=1, stroke=0)

def draw_header(c, T, page_num=1):
    # Bar
    c.setFillColor(T["header_bar"])
    c.rect(0, H - 52, W, 52, fill=1, stroke=0)
    c.setStrokeColor(T["border"])
    c.setLineWidth(0.8)
    c.line(0, H - 52, W, H - 52)
    # Logo
    c.setFont("Courier-Bold", 10)
    c.setFillColor(T["accent"])
    c.drawString(inch * 0.75, H - 33, "[ DETECTAI ]")
    # Right
    c.setFont("Courier", 7)
    c.setFillColor(T["muted"])
    c.drawRightString(W - inch * 0.75, H - 28,
                      f"AI DETECTION REPORT  //  PAGE {page_num}")

def draw_footer(c, T):
    c.setStrokeColor(T["border"])
    c.setLineWidth(0.5)
    c.line(inch * 0.75, 38, W - inch * 0.75, 38)
    c.setFont("Courier", 6.5)
    c.setFillColor(T["muted"])
    c.drawString(inch * 0.75, 26,
                 "Results are probabilistic and should not be used as sole evidence.")
    c.drawRightString(W - inch * 0.75, 26, "detectai.app")

def draw_section_title(c, T, x, y, label, line_end_x):
    c.setFont("Courier", 7)
    c.setFillColor(T["muted"])
    c.drawString(x, y, label)
    c.setStrokeColor(T["border"])
    c.setLineWidth(0.5)
    c.line(x, y - 7, line_end_x, y - 7)

# ══════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════
def build_text_report(result, username="user", theme="dark"):
    T = get_theme(theme)
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    final     = result.get("final_score", 0)
    confidence= result.get("confidence", 0)
    details   = result.get("details", {})
    clf       = details.get("classifier_score", 0)
    ppl_score = details.get("perplexity_score", 0)
    ppl_raw   = details.get("perplexity", 0)
    style_sc  = details.get("stylometry_score", 0)
    chunks    = details.get("num_chunks", 0)
    sc_col    = score_color(final, T)
    verdict   = verdict_text(final)

    page_bg(c, T)
    draw_header(c, T, 1)

    MX   = inch * 0.75          # left margin
    card_w = W - inch * 1.5
    y    = H - 90

    # ── meta row ─────────────────────────────────────────────
    c.setFont("Courier", 7)
    c.setFillColor(T["muted"])
    c.drawString(MX, y,
        f"USER: {username.upper()}   //   TEXT ANALYSIS   //   CHUNKS: {chunks}")
    y -= 26

    # ══ SCORE CARD ══════════════════════════════════════════
    # Layout: [donut | verdict+confidence | metrics (centred)]
    # Split card into thirds
    card_h = 180
    rrect(c, MX, y - card_h, card_w, card_h, r=10,
          fill=T["card"], stroke=T["border"])

    col1_cx = MX + card_w * 0.17          # donut center x
    col2_x  = MX + card_w * 0.34          # verdict block start
    col2_w  = card_w * 0.30               # verdict block width
    col3_x  = MX + card_w * 0.64          # metrics block start
    col3_w  = card_w * 0.32               # metrics block width

    cy_card = y - card_h / 2

    # — Donut —
    radius = 54
    draw_arc_donut(c, col1_cx, cy_card, radius, final, sc_col, T["border"])
    centered_string(c, f"{final*100:.1f}%",
                    "Courier-Bold", 17, col1_cx, cy_card + 4, sc_col)
    centered_string(c, "AI SCORE",
                    "Courier", 6, col1_cx, cy_card - 13, T["muted"])

    # — Verdict + Confidence (middle column) —
    # Badge
    badge_bg = T["badge_ai"] if sc_col == T["danger"] else \
               (T["badge_warn"] if sc_col == T["warn"] else T["badge_ok"])
    badge_y = cy_card + 32
    rrect(c, col2_x, badge_y, col2_w, 22, r=11, fill=badge_bg)
    c.setFont("Courier-Bold", 7.5)
    c.setFillColor(sc_col)
    bw = c.stringWidth(verdict, "Courier-Bold", 7.5)
    c.drawString(col2_x + (col2_w - bw) / 2, badge_y + 7, verdict)

    # Divider
    c.setStrokeColor(T["border"])
    c.setLineWidth(0.5)
    c.line(col2_x, badge_y - 12, col2_x + col2_w, badge_y - 12)

    # Confidence label + value
    c.setFont("Courier", 6.5)
    c.setFillColor(T["muted"])
    c.drawString(col2_x, badge_y - 26, "CONFIDENCE")
    c.setFont("Courier-Bold", 20)
    c.setFillColor(T["text"])
    c.drawString(col2_x, badge_y - 52, f"{confidence*100:.1f}%")

    c.setFont("Courier", 6.5)
    c.setFillColor(T["muted"])
    c.drawString(col2_x, badge_y - 66, "detection certainty")

    # Raw perplexity
    c.setFont("Courier", 6.5)
    c.setFillColor(T["muted"])
    c.drawString(col2_x, badge_y - 84, "RAW PERPLEXITY")
    c.setFont("Courier-Bold", 11)
    c.setFillColor(T["text"])
    c.drawString(col2_x, badge_y - 99, f"{ppl_raw:.2f}")

    # — Metric bars (right column, CENTRED inside col3) ——
    metrics = [
        ("CLASSIFIER",  clf,       sc_col),
        ("PERPLEXITY",  ppl_score, T["blue"]),
        ("STYLOMETRY",  style_sc,  T["muted"]),
    ]

    bar_w   = col3_w - 20          # leave 10px padding each side
    bar_x   = col3_x + 10
    m_start = cy_card + 55
    m_gap   = 44

    for label, val, col in metrics:
        # label left, value right
        c.setFont("Courier", 6.5)
        c.setFillColor(T["muted"])
        c.drawString(bar_x, m_start, label)
        c.setFont("Courier-Bold", 7.5)
        c.setFillColor(T["text"])
        pct_str = f"{val*100:.1f}%"
        pw = c.stringWidth(pct_str, "Courier-Bold", 7.5)
        c.drawString(bar_x + bar_w - pw, m_start, pct_str)
        # bar
        progress_bar(c, bar_x, m_start - 11, bar_w, 6, val, col, T["border"])
        m_start -= m_gap

    # vertical separator lines
    for sx in [col2_x - 10, col3_x - 10]:
        c.setStrokeColor(T["border"])
        c.setLineWidth(0.6)
        c.line(sx, y - card_h + 16, sx, y - 16)

    y -= card_h + 18

    # ══ STAT PILLS ══════════════════════════════════════════
    pill_defs = [
        ("RAW PERPLEXITY", f"{ppl_raw:.2f}",          "lower = more AI-like",     T["accent"]),
        ("CONFIDENCE",     f"{confidence*100:.1f}%",  "detection certainty",       T["blue"]),
        ("CHUNKS",         str(chunks),                "120-word windows",          T["muted"]),
    ]
    pill_w = (card_w - 16) / 3
    for i, (lbl, val, sub, col) in enumerate(pill_defs):
        px = MX + i * (pill_w + 8)
        rrect(c, px, y - 68, pill_w, 68, r=8, fill=T["card"], stroke=T["border"])
        # top accent line
        rrect(c, px, y - 3, pill_w, 3, r=0, fill=col)
        c.setFont("Courier", 6)
        c.setFillColor(T["muted"])
        c.drawString(px + 10, y - 18, lbl)
        c.setFont("Courier-Bold", 15)
        c.setFillColor(col)
        c.drawString(px + 10, y - 40, val)
        c.setFont("Courier", 6)
        c.setFillColor(T["muted"])
        c.drawString(px + 10, y - 56, sub)

    y -= 82

    # ══ SCORE BREAKDOWN TABLE ════════════════════════════════
    draw_section_title(c, T, MX, y, "SCORE BREAKDOWN", MX + card_w)
    y -= 18

    rows = [
        ("Classifier (RoBERTa)", clf,       "60% weight"),
        ("Perplexity (GPT-2)",   ppl_score, "25% weight"),
        ("Stylometry",           style_sc,  "15% weight"),
        ("Final Score",          final,     "weighted result"),
    ]

    # Column positions
    col_name  = MX + 12
    col_score = MX + card_w * 0.45
    col_note  = MX + card_w * 0.58
    col_bar   = MX + card_w * 0.72
    bar_table_w = card_w * 0.24

    for i, (name, val, note) in enumerate(rows):
        bg = T["card_alt"] if i % 2 == 0 else T["card"]
        rrect(c, MX, y - 26, card_w, 26, r=0, fill=bg)

        is_final = (i == len(rows) - 1)
        name_col = T["accent"] if is_final else T["text"]
        font_name = "Courier-Bold" if is_final else "Courier"

        c.setFont(font_name, 8)
        c.setFillColor(name_col)
        c.drawString(col_name, y - 16, name)

        c.setFont("Courier-Bold", 8)
        c.setFillColor(score_color(val, T))
        c.drawString(col_score, y - 16, f"{val*100:.2f}%")

        c.setFont("Courier", 7)
        c.setFillColor(T["muted"])
        c.drawString(col_note, y - 16, note)

        progress_bar(c, col_bar, y - 20, bar_table_w, 5,
                     val, score_color(val, T), T["border"])

        y -= 26

    draw_footer(c, T)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ══════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════
def build_pdf_report(result, username="user", theme="dark"):
    T = get_theme(theme)
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    overall    = result.get("overall_ai", 0) / 100
    pages_data = result.get("pages", [])
    top_pages  = result.get("top_pages", [])
    sc_col     = score_color(overall, T)
    verdict    = verdict_text(overall)

    MX     = inch * 0.75
    card_w = W - inch * 1.5
    page_num = 1

    page_bg(c, T)
    draw_header(c, T, page_num)

    y = H - 90

    c.setFont("Courier", 7)
    c.setFillColor(T["muted"])
    c.drawString(MX, y,
        f"USER: {username.upper()}   //   PDF ANALYSIS   //   PAGES: {len(pages_data)}")
    y -= 26

    # ══ OVERALL SCORE CARD ══════════════════════════════════
    card_h = 160
    rrect(c, MX, y - card_h, card_w, card_h, r=10,
          fill=T["card"], stroke=T["border"])

    col1_cx = MX + card_w * 0.16
    col2_x  = MX + card_w * 0.33
    col2_w  = card_w * 0.33
    col3_x  = MX + card_w * 0.67
    col3_w  = card_w * 0.28
    cy_card = y - card_h / 2

    # Donut
    draw_arc_donut(c, col1_cx, cy_card, 50, overall, sc_col, T["border"])
    centered_string(c, f"{overall*100:.1f}%",
                    "Courier-Bold", 16, col1_cx, cy_card + 3, sc_col)
    centered_string(c, "OVERALL AI",
                    "Courier", 6, col1_cx, cy_card - 14, T["muted"])

    # Verdict badge
    badge_bg = T["badge_ai"] if sc_col == T["danger"] else \
               (T["badge_warn"] if sc_col == T["warn"] else T["badge_ok"])
    badge_y = cy_card + 36
    bw2 = col2_w - 20
    rrect(c, col2_x, badge_y, bw2, 20, r=10, fill=badge_bg)
    c.setFont("Courier-Bold", 7)
    c.setFillColor(sc_col)
    vtw = c.stringWidth(verdict, "Courier-Bold", 7)
    c.drawString(col2_x + (bw2 - vtw) / 2, badge_y + 6, verdict)

    # Stats below badge
    stats = [
        ("PAGES ANALYZED", str(len(pages_data))),
        ("SCORING",        "WORD-WEIGHTED"),
    ]
    sy = badge_y - 18
    for lbl, val in stats:
        c.setFont("Courier", 6.5)
        c.setFillColor(T["muted"])
        c.drawString(col2_x, sy, lbl)
        c.setFont("Courier-Bold", 9)
        c.setFillColor(T["text"])
        c.drawString(col2_x, sy - 14, val)
        sy -= 36

    # Top pages (right column)
    c.setFont("Courier", 6.5)
    c.setFillColor(T["muted"])
    c.drawString(col3_x, cy_card + 55, "TOP PAGES")
    tp_y = cy_card + 40
    for p in top_pages[:3]:
        ps   = p.get("final_score", 0)
        pc   = score_color(ps, T)
        row_h = 22
        rrect(c, col3_x, tp_y - row_h + 4, col3_w, row_h, r=4, fill=T["card_alt"])
        c.setFont("Courier", 7)
        c.setFillColor(T["muted"])
        c.drawString(col3_x + 8, tp_y - 10, f"Page {p.get('page','?')}")
        c.setFont("Courier-Bold", 7)
        c.setFillColor(pc)
        c.drawRightString(col3_x + col3_w - 8, tp_y - 10, f"{ps*100:.1f}%")
        tp_y -= row_h + 4

    # Separators
    for sx in [col2_x - 10, col3_x - 10]:
        c.setStrokeColor(T["border"])
        c.setLineWidth(0.6)
        c.line(sx, y - card_h + 14, sx, y - 14)

    y -= card_h + 18

    # ══ PAGE-BY-PAGE TABLE ══════════════════════════════════
    draw_section_title(c, T, MX, y, "PAGE-BY-PAGE RESULTS", MX + card_w)
    y -= 18

    # Header row
    cols  = [50, 100, 100, 70, card_w - 340]
    hdrs  = ["PAGE", "AI SCORE", "CONFIDENCE", "WORDS", "VERDICT"]
    bar_col_start = MX + 10 + cols[0]  # where score bar goes

    rrect(c, MX, y - 22, card_w, 22, r=0, fill=T["surface"])
    hx = MX + 10
    for hdr, cw2 in zip(hdrs, cols):
        c.setFont("Courier", 6.5)
        c.setFillColor(T["muted"])
        c.drawString(hx, y - 14, hdr)
        hx += cw2
    y -= 22

    for i, p in enumerate(pages_data):
        if y < 72:
            draw_footer(c, T)
            c.showPage()
            page_num += 1
            page_bg(c, T)
            draw_header(c, T, page_num)
            y = H - 72

        ps   = p.get("final_score", 0)
        pcon = p.get("confidence", 0)
        pw   = p.get("word_count", 0)
        pc   = score_color(ps, T)
        pv   = verdict_text(ps)

        bg = T["card_alt"] if i % 2 == 0 else T["card"]
        rrect(c, MX, y - 24, card_w, 24, r=0, fill=bg)

        rx = MX + 10
        cell_vals = [
            (str(p.get("page", "?")), T["text"],  "Courier",      8),
            (f"{ps*100:.2f}%",        pc,          "Courier-Bold", 8),
            (f"{pcon*100:.1f}%",      T["text"],   "Courier",      8),
            (str(pw),                 T["muted"],  "Courier",      8),
            (pv,                      pc,          "Courier",      7),
        ]
        for (val, col, font, size), cw2 in zip(cell_vals, cols):
            c.setFont(font, size)
            c.setFillColor(col)
            c.drawString(rx, y - 15, val)
            rx += cw2

        # mini bar under score value
        progress_bar(c, MX + 10 + cols[0], y - 21, 80, 3, ps, pc, T["border"])

        y -= 24

    draw_footer(c, T)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ══════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    text_result = {
        "type": "text",
        "final_score": 0.325,
        "confidence": 0.829,
        "details": {
            "classifier_score": 0.171,
            "perplexity_score": 0.166,
            "perplexity": 17.39,
            "stylometry_score": 0.420,
            "num_chunks": 8,
        }
    }

    pdf_result = {
        "type": "pdf",
        "overall_ai": 82.12,
        "top_pages": [
            {"page": 3, "final_score": 0.8348},
            {"page": 1, "final_score": 0.8265},
        ],
        "pages": [
            {"page": 1, "final_score": 0.8265, "confidence": 0.81, "word_count": 312},
            {"page": 2, "final_score": 0.45,   "confidence": 0.55, "word_count": 198},
            {"page": 3, "final_score": 0.8348, "confidence": 0.88, "word_count": 420},
            {"page": 4, "final_score": 0.29,   "confidence": 0.62, "word_count": 150},
        ]
    }

    for theme in ["dark", "light", "ocean"]:
        buf = build_text_report(text_result, username="bhanu", theme=theme)
        with open(f"/home/claude/text_{theme}.pdf", "wb") as f:
            f.write(buf.read())

        buf2 = build_pdf_report(pdf_result, username="bhanu", theme=theme)
        with open(f"/home/claude/pdf_{theme}.pdf", "wb") as f:
            f.write(buf2.read())

    