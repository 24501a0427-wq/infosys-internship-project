"""
MILESTONE 4 - Dashboard Storage
--------------------------------
Store scan history and AI feedback in CSV files.
"""
import os
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
SCANS_FILE = os.path.join(LOG_DIR, "scans.csv")
FEEDBACK_FILE = os.path.join(LOG_DIR, "feedback.csv")

def _append(path, row):
    """Append one row to a CSV file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )

def _read(path, columns):
    """Read a CSV log safely."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce"
        )
    return df

# Scan history
def log_scan(source, summary, media="image"):
    """Save one image/video analysis result."""
    _append(SCANS_FILE, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "media": media,
        "violations": len(summary["violations"]),
        "violation_types": ";".join(summary["violations"]),
        "compliant": summary.get("compliant"),
    })

def load_scans():
    """Load previous scan results."""
    df = _read(
        SCANS_FILE,
        [
            "time",
            "source",
            "media",
            "violations",
            "violation_types",
            "compliant",
        ],
    )
    if not df.empty:
        df["violation_types"] = (
            df["violation_types"].fillna("")
        )
    return df

def violation_counts(scans):
    """Count each violation type."""
    if scans.empty:
        return pd.Series(dtype="object")
    items = [
        v
        for text in scans["violation_types"]
        for v in str(text).split(";")
        if v
    ]
    return pd.Series(
        items,
        dtype="object"
    ).value_counts()

# AI feedback

def log_feedback(question, liked):
    """Save user 👍 / 👎 feedback."""
    _append(
        FEEDBACK_FILE,
        {
            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "question": question,
            "liked": int(bool(liked)),
        },
    )

def satisfaction():
    """Return percentage of positive AI feedback."""
    df = _read(
        FEEDBACK_FILE,
        ["time", "question", "liked"],
    )
    if df.empty:
        return None
    return round(
        df["liked"].mean() * 100,
        1,
    )

# Dashboard KPIs

def kpis(scans):
    """Calculate dashboard KPI values."""
    total = len(scans)
    if total == 0:
        return {
            "scans": 0,
            "compliance": None,
            "violations": 0,
            "satisfaction": satisfaction(),
        }
    # Only definite True/False results are used for compliance.
    known = scans["compliant"].dropna()
    compliance = (
        round(
            known.astype(bool).mean() * 100,
            1,
        )
        if not known.empty
        else None
    )
    return {
        "scans": total,
        "compliance": compliance,
        "violations": int(
            scans["violations"].sum()
        ),
        "satisfaction": satisfaction(),
    }

def clear_all():
    """Clear scan and feedback history."""
    for path in (
        SCANS_FILE,
        FEEDBACK_FILE,
    ):
        if os.path.exists(path):
            os.remove(path)