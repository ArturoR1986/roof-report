import streamlit as st
from datetime import datetime
from PIL import Image
import textwrap

# -------------- Helper functions -------------- #

def infer_severity_and_urgency(notes: str):
    """
    Very simple heuristic to guess severity & urgency
    based on keywords in the notes.
    This is intentionally basic so it never gets in the way.
    """
    text = notes.lower()

    severity = "Moderate"
    urgency = "Soon"

    high_keywords = ["active leak", "leak", "emergency", "major", "severe", "collapse"]
    low_keywords = ["hairline", "minor", "cosmetic", "monitor", "stain only"]

    if any(word in text for word in high_keywords):
        severity = "High"
        urgency = "Immediate"
    elif any(word in text for word in low_keywords):
        severity = "Low"
        urgency = "Routine"

    return severity, urgency


def generate_report(notes: str, severity: str | None = None, urgency: str | None = None):
    """
    Turn rough notes into a structured report.
    This is template-based on purpose: simple, predictable, easy to adjust.
    """

    # Fallback if user left notes empty
    if not notes.strip():
        notes = "No detailed notes were provided. Observations based on visual inspection only."

    auto_sev, auto_urg = infer_severity_and_urgency(notes)

    if severity is None or severity == "Auto":
        severity = auto_sev
    if urgency is None or urgency == "Auto":
        urgency = auto_urg

    # Very simple "parsing" of notes into sections.
    # You can refine this over time.
    issue_observed = notes.strip()

    probable_cause = (
        "Based on the description provided, the issue is likely related to either "
        "age-related membrane wear, drainage limitations, or localized damage. "
        "Further on-site investigation is recommended to confirm root cause."
    )

    recommendations = textwrap.dedent(
        """
        - Perform a closer inspection of the affected area, including surrounding seams and penetrations.
        - Remove debris and verify that all drains, scuppers, and gutters are clear and functional.
        - Repair or replace damaged membrane, flashings, or sealant as needed.
        - Monitor the area after the next significant rainfall to confirm that the issue has been resolved.
        """
    ).strip()

    report = f"""
### 1. Issue Observed
{issue_observed}

### 2. Probable Cause
{probable_cause}

### 3. Recommendations
{recommendations}

### 4. Severity
**{severity}**

### 5. Urgency
**{urgency}**
"""
    return report


# -------------- Streamlit UI -------------- #

st.set_page_config(
    page_title="Roof Photo → Report (Demo)",
    page_icon="🧱",
    layout="centered"
)

st.title("🧱 Roof Photo → Report Generator")
st.caption("Demo tool – turns a photo + rough notes into a clean, structured report.")

st.markdown("---")

# Upload section
uploaded_image = st.file_uploader(
    "1️⃣ Upload a roof photo (JPEG/PNG)",
    type=["jpg", "jpeg", "png"]
)

notes = st.text_area(
    "2️⃣ Paste or type your rough notes / observations",
    placeholder=(
        "Example:\n"
        "- Active leak reported above unit 302\n"
        "- Ponding water around drain\n"
        "- Blistering in modified bitumen near parapet\n"
        "- Temporary repair already in place"
    ),
    height=180
)

col1, col2, col3 = st.columns(3)
with col1:
    severity_choice = st.selectbox(
        "Severity",
        options=["Auto", "Low", "Moderate", "High"],
        index=0,
        help="You can override the automatic severity if you want."
    )
with col2:
    urgency_choice = st.selectbox(
        "Urgency",
        options=["Auto", "Routine", "Soon", "Immediate"],
        index=0,
        help="You can override the automatic urgency if you want."
    )
with col3:
    show_meta = st.checkbox("Show meta info", value=True)

generate_btn = st.button("⚙️ Generate Report", type="primary")

st.markdown("---")

if generate_btn:
    if uploaded_image is None and not notes.strip():
        st.warning("Please upload at least a photo **or** provide some notes.")
    else:
        # Display the image (if provided)
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            st.subheader("Photo Preview")
            st.image(img, use_container_width=True)

        st.subheader("Generated Report")

        report_md = generate_report(
            notes=notes,
            severity=severity_choice,
            urgency=urgency_choice
        )
        st.markdown(report_md)

        if show_meta:
            st.markdown("---")
            st.caption(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
                "Note: This is a demo tool intended to support field documentation. "
                "Final assessment should always be confirmed by a qualified roofing professional."
            )
