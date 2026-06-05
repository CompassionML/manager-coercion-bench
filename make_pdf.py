"""
Convert REPORT.md -> report.pdf with the two figures embedded inline.

Word COM automation won't launch in a headless session, so we render the PDF
directly with reportlab instead of going through .docx.

Fonts: we register DejaVu Sans (shipped inside matplotlib, so guaranteed
present) for *everything*, because the report body uses geometric glyphs
(▽ ✗ ★ ▢, middle dots, en/em dashes) that the standard PDF base-14 fonts do
not contain — DejaVu covers them.

Same lightweight Markdown handling as make_docx.py: headings (#..####),
bullet/numbered lists, blockquotes, inline **bold**/*italic*/`code`, ![alt](path)
images, and --- rules.
"""

import os
import re

import matplotlib
from PIL import Image as PILImage
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, Paragraph, SimpleDocTemplate,
                                Spacer)

SRC = "REPORT.md"
OUT = "report.pdf"

INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")
IMG = re.compile(r"^!\[(.*)\]\((.*)\)\s*$")
HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
NUM = re.compile(r"^(\d+)\.\s+(.*)$")

# ---- fonts: DejaVu (full Unicode coverage) from matplotlib's bundle ----------
FONTDIR = os.path.join(os.path.dirname(matplotlib.__file__),
                       "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONTDIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONTDIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Italic", os.path.join(FONTDIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldItalic", os.path.join(FONTDIR, "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", os.path.join(FONTDIR, "DejaVuSansMono.ttf")))
registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                   italic="DejaVu-Italic", boldItalic="DejaVu-BoldItalic")

INK = "#1a2540"
GREY = "#666666"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_inline(text):
    """Markdown inline -> reportlab mini-markup, XML-escaped."""
    text = text.replace(r"\*", "*")
    out, pos = [], 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            out.append(esc(text[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith("**"):
            out.append("<b>" + esc(tok[2:-2]) + "</b>")
        elif tok.startswith("`"):
            out.append('<font name="DejaVuMono" size="9">' + esc(tok[1:-1]) + "</font>")
        else:
            out.append("<i>" + esc(tok[1:-1]) + "</i>")
        pos = m.end()
    if pos < len(text):
        out.append(esc(text[pos:]))
    return "".join(out)


# ---- styles ------------------------------------------------------------------
S = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=19,
                            leading=23, textColor=INK, spaceAfter=2),
    "subtitle": ParagraphStyle("subtitle", fontName="DejaVu-Italic", fontSize=12.5,
                               leading=16, textColor=GREY, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=14.5, leading=18,
                         textColor=INK, spaceBefore=16, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="DejaVu-Bold", fontSize=11.5, leading=15,
                         textColor=INK, spaceBefore=10, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=10, leading=14.5,
                           spaceAfter=7, alignment=TA_JUSTIFY),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=10, leading=14,
                             spaceAfter=3, leftIndent=16, firstLineIndent=-10),
    "quote": ParagraphStyle("quote", fontName="DejaVu-Italic", fontSize=9.5,
                            leading=13.5, textColor="#333333", leftIndent=18,
                            rightIndent=10, spaceBefore=2, spaceAfter=7,
                            borderColor="#cccccc"),
    "caption": ParagraphStyle("caption", fontName="DejaVu-Italic", fontSize=8.5,
                              leading=11, textColor=GREY, alignment=TA_CENTER,
                              spaceBefore=4, spaceAfter=12),
}

CONTENT_W = letter[0] - 1.7 * inch     # 0.85" margins each side
MAX_IMG_H = 8.3 * inch


def image_flowable(path):
    iw, ih = PILImage.open(path).size
    w = CONTENT_W
    h = w * ih / iw
    if h > MAX_IMG_H:
        h = MAX_IMG_H
        w = h * iw / ih
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    return img


def main():
    story = []
    lines = open(SRC, encoding="utf-8").read().splitlines()
    for line in lines:
        s = line.rstrip()
        if not s.strip():
            continue
        if s.strip() == "---":
            story.append(Spacer(1, 2))
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color="#cccccc", spaceBefore=2, spaceAfter=8))
            continue

        mimg = IMG.match(s)
        if mimg:
            alt, path = mimg.group(1), mimg.group(2)
            story.append(image_flowable(path))
            story.append(Paragraph(md_inline(alt), S["caption"]))
            continue

        mhead = HEAD.match(s)
        if mhead:
            level = len(mhead.group(1))
            txt = md_inline(mhead.group(2).strip())
            if level == 1:
                story.append(Paragraph(txt, S["title"]))
            elif level == 2:
                story.append(Paragraph(txt, S["h2"]))
            elif level == 3 and mhead.group(2).strip().startswith("A Milgram"):
                story.append(Paragraph(txt, S["subtitle"]))
            else:
                story.append(Paragraph(txt, S["h3"]))
            continue

        if s.startswith("> "):
            story.append(Paragraph(md_inline(s[2:]), S["quote"]))
            continue
        if s.startswith("- "):
            story.append(Paragraph("•&nbsp;&nbsp;" + md_inline(s[2:]), S["bullet"]))
            continue
        mnum = NUM.match(s)
        if mnum:
            story.append(Paragraph(f"{mnum.group(1)}.&nbsp;&nbsp;"
                                   + md_inline(mnum.group(2)), S["bullet"]))
            continue

        story.append(Paragraph(md_inline(s), S["body"]))

    doc = SimpleDocTemplate(OUT, pagesize=letter,
                            leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title="Coercion and Deception in AI-to-AI Management")
    n = len(story)  # build() drains the list, so capture the count first
    doc.build(story)
    print(f"wrote {OUT}  |  flowables={n}")


if __name__ == "__main__":
    main()
