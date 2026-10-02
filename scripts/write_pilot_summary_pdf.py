#!/usr/bin/env python3
"""Write a downloadable PDF brief of the 48-request last-token analysis."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "pilot_text48_summary.pdf"

NAVY = colors.HexColor("#1e3a5f")
TEAL = colors.HexColor("#0f766e")
MUTED = colors.HexColor("#4b5563")
RULE = colors.HexColor("#d1d5db")
ROW = colors.HexColor("#f3f4f6")


def styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=MUTED,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#111827"),
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            spaceBefore=2,
            spaceAfter=10,
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#111827"),
            leftIndent=8,
        ),
        "foot": ParagraphStyle(
            "FootCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
        ),
        "th": ParagraphStyle(
            "ThCustom",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "td": ParagraphStyle(
            "TdCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
        ),
        "tdl": ParagraphStyle(
            "TdlCustom",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            alignment=TA_LEFT,
        ),
    }


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(0.7 * inch, letter[1] - 0.45 * inch, letter[0] - 0.7 * inch, letter[1] - 0.45 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.7 * inch, letter[1] - 0.38 * inch, "knowledge_inv  ·  Qwen3-VL-8B last-token PCA brief")
    canvas.drawRightString(letter[0] - 0.7 * inch, letter[1] - 0.38 * inch, "48 text requests  ·  not causal")
    canvas.line(0.7 * inch, 0.5 * inch, letter[0] - 0.7 * inch, 0.5 * inch)
    canvas.drawString(0.7 * inch, 0.35 * inch, "runs/pilot_text48  ·  last prompt token at layers 8 / 17 / 35")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def fig_image(path: Path, width: float) -> Image:
    img = Image(str(path))
    img.drawWidth = width
    img.drawHeight = width * img.imageHeight / img.imageWidth
    return img


def eta_chart() -> Image:
    cats = ["L8 PC1", "L8 PC2", "L8 PC3", "L17 PC1", "L17 PC2", "L17 PC3", "L35 PC1", "L35 PC2", "L35 PC3"]
    cond = np.array([96.7, 5.4, 79.3, 97.9, 75.4, 10.4, 92.2, 60.9, 11.3])
    req = np.array([2.8, 91.1, 15.8, 1.5, 16.1, 82.6, 6.0, 29.2, 76.0])
    left = 100 - cond - req
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    x = np.arange(len(cats))
    ax.bar(x, cond, color="#2563eb", label="Prompt condition (direct / plan / FBS)")
    ax.bar(x, req, bottom=cond, color="#d97706", label="Which CAD request")
    ax.bar(x, left, bottom=cond + req, color="#9ca3af", label="Leftover (shift size depends on request)")
    ax.set_xticks(x, cats, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Share of that PC (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Eta-squared on 144 PC scores (48 requests × 3 conditions)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img.drawWidth = 7.1 * inch
    img.drawHeight = 2.55 * inch
    return img


def simple_table(header: list[str], rows: list[list], col_widths: list[float], s: dict) -> Table:
    data = [[Paragraph(h, s["th"]) for h in header]]
    for row in rows:
        styled = []
        for i, cell in enumerate(row):
            styled.append(Paragraph(str(cell), s["tdl"] if i == 0 else s["td"]))
        data.append(styled)
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW]),
                ("GRID", (0, 0), (-1, -1), 0.3, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build() -> None:
    s = styles()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.7 * inch,
        title="Last-token PCA brief — 48 text requests",
        author="knowledge_inv",
    )
    width = 7.1 * inch
    story = []

    story.append(Paragraph("Last-token geometry of CAD-edit decisions", s["title"]))
    story.append(
        Paragraph(
            "Qwen3-VL-8B-Instruct  ·  neuralCAD-Edit 48 eligible text requests  ·  "
            "three prompt conditions  ·  layers 8 / 17 / 35",
            s["subtitle"],
        )
    )
    story.append(Paragraph("What this brief is", s["h1"]))
    story.append(
        Paragraph(
            "A compact write-up of the 48-request pilot: what was measured, what the "
            "3D PCA plots show, how much of each principal component lines up with "
            "prompt condition versus which CAD request, and how PC1 relates to prompt "
            "length. It is a summary of the live canvases, not a new experiment.",
            s["body"],
        )
    )
    story.append(Paragraph("Setup", s["h1"]))
    story.append(
        ListFlowable(
            [
                ListItem(
                    Paragraph(
                        "<b>Model.</b> Qwen3-VL-8B-Instruct, revision 0c351dd, BF16. "
                        "Hidden size 4096; language layers 0–35.",
                        s["bullet"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "<b>Data.</b> All 48 eligible neuralCAD-Edit text requests "
                        "(genuine text instruction, original STEP, all original views). "
                        "Evaluation edits were not shown to the model.",
                        s["bullet"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "<b>Conditions.</b> <i>Direct</i>: image + request, then decide. "
                        "<i>Plan</i>: model writes a generic plan, then decides with that "
                        "plan in context. <i>FBS</i>: model writes a function–behavior–structure "
                        "analysis, then decides with that analysis in context. Plan is the "
                        "control for “extra generated text” versus structured FBS.",
                        s["bullet"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "<b>Vector.</b> Last <i>prompt</i> token (index prompt_length − 1) "
                        "at decoder layers 8, 17, and 35 — the state after reading the image "
                        "and decision prompt, before the answer is written. Vectors are "
                        "L2-normalized; PCA to 3 components is fit separately per layer "
                        "(144 points = 48 × 3).",
                        s["bullet"],
                    )
                ),
            ],
            bulletType="bullet",
            leftIndent=12,
            bulletFontName="Helvetica",
            bulletFontSize=9,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        simple_table(
            ["", "Layer 8", "Layer 17", "Layer 35"],
            [
                ["Variance in 3 PCs", "63.9%", "69.3%", "59.1%"],
                ["PC1 share of that variance", "50.2%", "54.0%", "37.5%"],
                ["Mean cosine dist. direct–plan", "0.0067", "0.0104", "0.0263"],
                ["Mean cosine dist. direct–FBS", "0.0069", "0.0098", "0.0272"],
                ["Mean cosine dist. plan–FBS", "0.0016", "0.0030", "0.0130"],
                ["PC1 vs prompt length (Pearson r)", "−0.994", "−0.987", "−0.977"],
            ],
            [2.6 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch],
            s,
        )
    )
    story.append(
        Paragraph(
            "Table 1. Headline numbers. Cosine distance = 1 − cosine similarity of the "
            "last prompt token. PCA axes are not shared across layers.",
            s["caption"],
        )
    )

    story.append(Paragraph("What the three PCs track (post-hoc)", s["h1"]))
    story.append(
        Paragraph(
            "PCA does not name its axes. The labels below come from scoring the 144 "
            "points and measuring eta-squared against two groupings: prompt condition "
            "(3 piles of 48) and request identity (48 piles of 3). Sign is arbitrary. "
            "Layer-8 PC2 is not the same direction as layer-35 PC2.",
            s["body"],
        )
    )
    story.append(
        simple_table(
            ["Axis", "Best reading", "Layer 8", "Layer 17", "Layer 35"],
            [
                [
                    "PC1",
                    "Extra generated text (direct vs plan/FBS together)",
                    "cond η² 97%",
                    "cond η² 98%",
                    "cond η² 92%",
                ],
                [
                    "PC2",
                    "Which request at L8; plan vs FBS at L17/L35",
                    "request 91%",
                    "condition 75%",
                    "condition 61%",
                ],
                [
                    "PC3",
                    "Plan vs FBS at L8; which request at L17/L35",
                    "condition 79%",
                    "request 83%",
                    "request 76%",
                ],
            ],
            [0.7 * inch, 2.6 * inch, 1.27 * inch, 1.27 * inch, 1.26 * inch],
            s,
        )
    )
    story.append(
        Paragraph(
            "Table 2. Post-hoc PC readings. “Condition” = direct / plan / FBS. "
            "Plan versus FBS is the knowledge-control contrast; it is never PC1.",
            s["caption"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("3D PCA — same three PCs at layers 8, 17, 35", s["h1"]))
    story.append(
        fig_image(ROOT / "reports" / "last_token_pca3d_text48" / "layers_8_17_35_pca3d.png", width)
    )
    story.append(
        Paragraph(
            "Figure 1. Last prompt token in 3D PCA. Blue circles = direct; green squares = "
            "plan; orange triangles = FBS. Lines go from each request’s direct point to its "
            "plan and FBS points. PCA is fit separately per layer, so the three boxes do "
            "not share axes. Rotate the interactive copy: "
            "reports/last_token_pca3d_text48/layers_8_17_35_pca3d.html",
            s["caption"],
        )
    )
    story.append(
        fig_image(ROOT / "reports" / "last_token_pca3d_text48" / "layer_8_pca_faces.png", width)
    )
    story.append(
        Paragraph(
            "Figure 2. Layer 8 flattened into three faces: PC1 vs PC2, PC1 vs PC3, and "
            "PC2 vs PC3. PC1 vs PC2 still mixes request identity onto the vertical axis. "
            "PC1 vs PC3 is the clean three-cluster view (direct / plan / FBS). PC2 vs PC3 "
            "drops the extra-text axis and shows request identity against plan vs FBS.",
            s["caption"],
        )
    )
    story.append(
        fig_image(ROOT / "reports" / "last_token_pca3d_text48" / "pc2_vs_pc3_layers.png", width)
    )
    story.append(
        Paragraph(
            "Figure 2b. PC2 vs PC3 at layers 8, 17, and 35. Extra-text PC1 is left out, "
            "so the view is request identity against plan vs FBS. At layer 8 those roles "
            "are PC2 (request) vs PC3 (plan/FBS); at layers 17 and 35 they swap.",
            s["caption"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Share of each PC explained by condition vs request", s["h1"]))
    story.append(
        Paragraph(
            "Each PC is one number per point. Eta-squared asks: if those 144 numbers are "
            "sorted into piles, how much of the spread is between piles rather than inside "
            "them? Prompt condition uses 3 piles of 48. Request identity uses 48 piles of 3 "
            "(direct, plan, and FBS of one CAD edit). The two bars need not sum to 100%; "
            "the grey remainder is leftover — the size of the condition shift depends on "
            "which request. These percents are of that PC only, not of the original 4096-d "
            "state. Layer-8 PC1 is 50.2% of total variance; 96.7% of that axis is condition, "
            "so about half of last-token variance at layer 8 lines up with extra generated text.",
            s["body"],
        )
    )
    story.append(eta_chart())
    story.append(
        Paragraph(
            "Figure 3. Stacked eta-squared. Blue = prompt condition; orange = which CAD "
            "request; grey = leftover interaction.",
            s["caption"],
        )
    )
    story.append(Paragraph("PC1 against prompt length", s["h1"]))
    story.append(
        Paragraph(
            "Prompt length is the tokenized decision prompt, including a constant 441 "
            "image-pad tokens (so length differences are all text). Direct prompts are "
            "~511–623 tokens; plan and FBS are ~757–872 because the generated analysis is "
            "pasted into the decision prompt. That gap drives most of the overall correlation. "
            "A slope remains inside each cloud: among directs only, longer instructions still "
            "have lower PC1 (layer 8 r = −0.93). Length is confounded with condition and with "
            "which request was written longer. This does not prove PC1 is only a token counter.",
            s["body"],
        )
    )
    story.append(fig_image(ROOT / "reports" / "pc1_vs_prompt_length" / "pc1_vs_prompt_length.png", width))
    story.append(
        Paragraph(
            "Figure 4. PC1 vs prompt length. Grey line is OLS over all 144 points. "
            "Pearson r: layer 8 −0.994, layer 17 −0.987, layer 35 −0.977.",
            s["caption"],
        )
    )
    story.append(
        simple_table(
            ["Layer", "All 144", "direct only", "plan only", "FBS only"],
            [
                ["8", "−0.994", "−0.930", "−0.847", "−0.811"],
                ["17", "−0.987", "−0.927", "−0.452", "−0.645"],
                ["35", "−0.977", "−0.821", "−0.574", "−0.499"],
            ],
            [1.4 * inch, 1.425 * inch, 1.425 * inch, 1.425 * inch, 1.425 * inch],
            s,
        )
    )
    story.append(
        Paragraph(
            "Table 3. Pearson r of PC1 with prompt length. Mean length: direct 535, "
            "plan 782, FBS 784 tokens.",
            s["caption"],
        )
    )

    story.append(Paragraph("Cosine distance of the last prompt token", s["h1"]))
    story.append(
        Paragraph(
            "The last row of the saved activation matrix is usually a generated token, "
            "so it is not comparable across conditions. Distance here is always last "
            "prompt token to last prompt token. Plan versus FBS stays smaller than "
            "direct versus either extra-text condition at every layer, matching PC1: "
            "the large move is “I wrote extra text,” not “I wrote FBS rather than a plan.”",
            s["body"],
        )
    )
    story.append(
        fig_image(ROOT / "reports" / "last_token_cosine_text48" / "mean_cosine_distance.png", width * 0.92)
    )
    story.append(
        Paragraph(
            "Figure 5. Mean cosine distance over 48 requests. Axis: 1 − cosine similarity.",
            s["caption"],
        )
    )
    story.append(Paragraph("What this does not show", s["h1"]))
    story.append(
        ListFlowable(
            [
                ListItem(
                    Paragraph(
                        "It does not show that FBS knowledge caused a CAD decision. "
                        "We compared last-token states, not edit quality against held-out human edits.",
                        s["bullet"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "PC1 is aligned with extra generated text and with prompt length. "
                        "Those two are almost the same experimental contrast in this design.",
                        s["bullet"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "About 36–41% of last-token variance sits outside the three plotted PCs, "
                        "depending on layer. Distances in the 3D plots are approximate.",
                        s["bullet"],
                    )
                ),
            ],
            bulletType="bullet",
            leftIndent=12,
        )
    )
    story.append(Paragraph("Files", s["h1"]))
    story.append(
        Paragraph(
            "PDF: reports/pilot_text48_summary.pdf. Runs: runs/pilot_text48. "
            "3D HTML: reports/last_token_pca3d_text48/layers_8_17_35_pca3d.html. "
            "Coordinates: reports/last_token_pca3d_text48/pca3d_coordinates.json. "
            "Length join: reports/pc1_vs_prompt_length/. Manifest: manifests/pilot_text_all.jsonl.",
            s["body"],
        )
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
