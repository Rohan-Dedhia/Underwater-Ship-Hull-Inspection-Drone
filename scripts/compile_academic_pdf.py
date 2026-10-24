import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "outputs" / "reports" / "academic_report.md"
PDF_PATH = ROOT / "outputs" / "reports" / "academic_report.pdf"


def compile_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#0e3854"), spaceAfter=6)
    subtitle_style = ParagraphStyle("DocSub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#555555"), spaceAfter=12)
    h2_style = ParagraphStyle("Heading2C", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#1e3d59"), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("BodyC", parent=styles["Normal"], fontSize=8.5, leading=11.5, textColor=colors.HexColor("#222222"), spaceAfter=5)
    bullet_style = ParagraphStyle("BulletC", parent=styles["Normal"], fontSize=8.5, leading=11, textColor=colors.HexColor("#333333"), leftIndent=12, spaceAfter=2)

    story = []

    # Title & Subtitle
    story.append(Paragraph("AI-Powered Autonomous Underwater ROV Ship Hull Inspection", title_style))
    story.append(Paragraph("<b>End-to-End Machine Learning, Computer Vision, and Robotics Simulation Pipeline</b><br/>Fundamentals of Machine Learning (FOML) &bull; Academic Research Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0e3854"), spaceAfter=10))

    with open(MD_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cell_style = ParagraphStyle("CellC", parent=body_style, fontSize=7.5, leading=9, spaceAfter=0)

    def md_inline(text):
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'\$(.*?)\$', lambda m: m.group(1).replace(r'\tau', 'tau'), text)
        return text.replace("&", "&amp;").replace("&amp;bull;", "&bull;").replace("&amp;times;", "&times;")

    def flush_table(rows):
        """Render a Markdown table (header, separator, body rows) as a ReportLab Table."""
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        cells = [r for r in cells if not all(set(c) <= set(":- ") for c in r)]
        if not cells:
            return
        data = [[Paragraph(md_inline(c), cell_style) for c in r] for r in cells]
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(table)
        story.append(Spacer(1, 6))

    table_rows = []
    current_section = None
    for line in lines:
        line_str = line.strip()
        if line_str.startswith("|"):
            table_rows.append(line_str)
            continue
        if table_rows:
            flush_table(table_rows)
            table_rows = []
        if not line_str:
            continue
        if line_str.startswith("# ") or line_str.startswith("**Course"):
            continue
        elif line_str.startswith("## "):
            sec_title = line_str.replace("## ", "")
            story.append(Spacer(1, 4))
            story.append(Paragraph(sec_title, h2_style))
        elif line_str.startswith("### "):
            sub_title = line_str.replace("### ", "")
            story.append(Paragraph(f"<b>{sub_title}</b>", body_style))
        elif line_str.startswith("- ") or line_str.startswith("* "):
            text = line_str[2:]
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            text = text.replace("&", "&amp;").replace("&amp;bull;", "&bull;").replace("&amp;times;", "&times;")
            story.append(Paragraph(f"&bull; {text}", bullet_style))
        else:
            text = line_str
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            text = text.replace("&", "&amp;").replace("&amp;bull;", "&bull;").replace("&amp;times;", "&times;")
            story.append(Paragraph(text, body_style))

    if table_rows:
        flush_table(table_rows)
    doc.build(story)
    print(f"Academic Report PDF compiled successfully: {PDF_PATH}")


if __name__ == "__main__":
    compile_pdf()
