"""
MILESTONE 4 - Compliance Report
--------------------------------
Generate a PDF safety compliance report using
dashboard statistics and retrieved safety rules.
"""

from datetime import datetime

from fpdf import FPDF

from knowledge_base import search
from llm import ask_llm


def safe(text):
    """Make text compatible with the PDF font."""
    replacements = {
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "-",
        "✓": "v",
        "≥": ">=",
        "…": "...",
    }

    text = str(text)

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode(
        "latin-1",
        "replace"
    ).decode("latin-1")


def write_summary(k, counts):
    """Generate a short safety summary using Gemini."""

    violation_text = (
        ", ".join(counts.index)
        if len(counts)
        else "No violations recorded"
    )

    # Retrieve rules related to the detected violations.
    rules = search(
        f"safety rules for {violation_text}",
        top_k=3,
    )

    rules_text = "\n".join(
        f"[{d['source']} #{d['chunk']}] {d['text']}"
        for d in rules
    )

    prompt = f"""
You are preparing a workplace safety compliance report.

Write a short professional summary based ONLY on the
statistics and safety rules provided below.

Include:
- overall scan result
- detected violations
- relevant safety requirements
- two practical corrective actions

Do not invent rules or facts.
Maximum 120 words.

Statistics:
Scans: {k['scans']}
Compliance rate: {k['compliance']}%
Total violations: {k['violations']}
Violations by type: {violation_text}

Safety rules:
{rules_text or "No relevant safety rules found."}
"""

    summary = ask_llm(prompt)

    if summary:
        return summary

    return (
        f"Scans analysed: {k['scans']}. "
        f"Compliance rate: {k['compliance']}%. "
        f"Total violations: {k['violations']}. "
        f"Violations detected: {violation_text}."
    )


def build_pdf(k, counts, scans, summary):
    """Create the compliance report and return PDF bytes."""

    pdf = FPDF()
    pdf.add_page()

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------

    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(
        0,
        8,
        "AI-Powered Visual Data Analytics and "
        "Business Intelligence Platform",
    )

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0,
        8,
        "Safety Compliance Report",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Helvetica", size=10)
    pdf.cell(
        0,
        6,
        "Generated: "
        + datetime.now().strftime("%Y-%m-%d %H:%M"),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.ln(4)

    def heading(text):
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(
            0,
            9,
            text,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("Helvetica", size=10)

    def row(label, value):
        pdf.cell(90, 7, safe(label), border=1)
        pdf.cell(
            90,
            7,
            safe(value),
            border=1,
            new_x="LMARGIN",
            new_y="NEXT",
        )

    # ---------------------------------------------------------
    # Key numbers
    # ---------------------------------------------------------

    heading("1. Key Numbers")

    row("Images / videos checked", k["scans"])

    row(
        "Compliance rate",
        f"{k['compliance']}%"
        if k["compliance"] is not None
        else "-",
    )

    row("Total violations", k["violations"])

    row(
        "User satisfaction",
        f"{k['satisfaction']}%"
        if k["satisfaction"] is not None
        else "-",
    )

    pdf.ln(4)

    # ---------------------------------------------------------
    # AI summary
    # ---------------------------------------------------------

    heading("2. AI Safety Summary")

    pdf.multi_cell(
        0,
        6,
        safe(summary),
    )

    pdf.ln(4)

    # ---------------------------------------------------------
    # Violations
    # ---------------------------------------------------------

    heading("3. Violations by Type")

    if len(counts):
        for name, number in counts.items():
            row(name, number)
    else:
        pdf.cell(
            0,
            7,
            "No violations recorded.",
            new_x="LMARGIN",
            new_y="NEXT",
        )

    pdf.ln(4)

    # ---------------------------------------------------------
    # Recent scans
    # ---------------------------------------------------------

    heading("4. Recent Scans")

    if scans.empty:
        pdf.cell(
            0,
            7,
            "No scans recorded.",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    else:
        recent = scans.sort_values(
            "time",
            ascending=False,
        ).head(15)

        for scan in recent.itertuples():
            text = (
                f"{scan.time:%Y-%m-%d %H:%M} | "
                f"{scan.source} | "
                f"{scan.violations} violation(s) | "
                f"{scan.violation_types}"
            )

            pdf.cell(
                0,
                6,
                safe(text),
                new_x="LMARGIN",
                new_y="NEXT",
            )

    return bytes(pdf.output())
