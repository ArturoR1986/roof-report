import os
import streamlit as st
from datetime import datetime
from PIL import Image
from openai import OpenAI

# -------------- OpenAI client -------------- #

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# -------------- Domain dictionaries -------------- #

ROOF_SYSTEMS = [
    "Unknown / Not specified",
    "SBS modified bitumen",
    "BUR (built-up roof)",
    "TPO",
    "EPDM",
    "PVC",
    "Metal roof",
    "Shingle roof",
]

ISSUE_TYPES = [
    "General moisture concern",
    "Active leak at interior",
    "Ponding water",
    "Blistering / ridging",
    "Open seam / lap",
    "Damaged flashing",
    "Membrane puncture / tear",
    "Mechanical damage (traffic / tools)",
    "Clogged drain / scupper / gutter",
    "Debris on roof",
]

LOCATIONS = [
    "Not specified",
    "Field of roof",
    "Perimeter",
    "At drain / scupper",
    "At penetration (pipe / curb / unit)",
    "At parapet / wall detail",
    "At roof edge / metal flashing",
]

# -------------- Helper functions (rules) -------------- #

def infer_severity_and_urgency(notes: str, issue_type: str, active_leak: bool):
    text = (notes or "").lower()

    severity = "Moderate"
    urgency = "Soon"

    if active_leak or "active leak" in text or "leak" in text:
        return "High", "Immediate"

    high_issues = {
        "Membrane puncture / tear",
        "Open seam / lap",
        "Damaged flashing",
        "Clogged drain / scupper / gutter",
    }
    low_issues = {
        "Debris on roof",
        "Blistering / ridging",
        "Mechanical damage (traffic / tools)",
        "General moisture concern",
    }

    if issue_type in high_issues:
        severity, urgency = "High", "Soon"
    elif issue_type in low_issues:
        severity, urgency = "Low", "Routine"

    return severity, urgency


def build_probable_cause(system: str, issue_type: str, location: str) -> str:
    base = []

    if issue_type == "Ponding water":
        base.append(
            "Limited drainage in this area, likely due to insufficient slope, "
            "improper crickets, or partially obstructed drains or scuppers."
        )
    elif issue_type == "Blistering / ridging":
        base.append(
            "Trapped moisture or poor adhesion within the membrane or underlying layers, "
            "often related to age, installation quality, or prior moisture entry."
        )
    elif issue_type == "Open seam / lap":
        base.append(
            "Seam failure from aging, thermal movement, or inadequate original bonding."
        )
    elif issue_type == "Damaged flashing":
        base.append(
            "Movement of building components, thermal cycling, or mechanical impact at the detail."
        )
    elif issue_type == "Membrane puncture / tear":
        base.append(
            "Mechanical damage from foot traffic, dropped tools, or displaced equipment."
        )
    elif issue_type == "Mechanical damage (traffic / tools)":
        base.append(
            "Concentrated activity in this area without adequate protection or walkways."
        )
    elif issue_type == "Clogged drain / scupper / gutter":
        base.append(
            "Accumulated debris restricting water flow at drain, scupper, or gutter components."
        )
    elif issue_type == "Debris on roof":
        base.append(
            "Wind-blown debris or housekeeping issues, which can trap moisture and block drainage."
        )
    elif issue_type == "Active leak at interior":
        base.append(
            "Water migration through the roof assembly, likely originating near a detail, "
            "penetration, or historical repair in the vicinity of the reported leak."
        )
    else:
        base.append(
            "Moisture-related concern observed in this area. Exact source to be confirmed through further investigation."
        )

    if system in {"SBS modified bitumen", "BUR (built-up roof)"}:
        base.append(
            " On this type of system, age-related wear, prior repairs, and surface cracking can "
            "contribute to moisture entry if not maintained."
        )
    elif system in {"TPO", "EPDM", "PVC"}:
        base.append(
            " Single-ply membranes are sensitive to seam integrity, punctures, and detail flashing "
            "conditions, especially around penetrations and terminations."
        )
    elif system == "Metal roof":
        base.append(
            " Metal systems often develop issues at fasteners, panel laps, and terminations as sealants age "
            "and movement occurs."
        )
    elif system == "Shingle roof":
        base.append(
            " Shingle roofs typically leak at flashings, transitions, or where fasteners and sealants have aged."
        )

    if location == "At drain / scupper":
        base.append(
            " Issues at drainage points can accelerate membrane wear and increase the risk of interior leakage."
        )
    elif location == "At penetration (pipe / curb / unit)":
        base.append(
            " Penetration details are common leak sources if flashing height, terminations, or sealants are compromised."
        )
    elif location == "At parapet / wall detail":
        base.append(
            " Transitions between horizontal and vertical surfaces are sensitive to movement and detailing quality."
        )
    elif location == "At roof edge / metal flashing":
        base.append(
            " Edge metal, terminations, and perimeter details are exposed to wind uplift and weathering."
        )

    return "".join(base)


def build_recommendations(issue_type: str, active_leak: bool) -> str:
    recs = []

    if active_leak:
        recs.append(
            "- Address active leak as a priority to limit interior damage and disruption."
        )

    if issue_type == "Ponding water":
        recs.extend([
            "- Clear debris from drains, scuppers, and nearby areas to restore flow.",
            "- Verify slope and consider adding tapered insulation or crickets if ponding persists.",
        ])
    elif issue_type == "Blistering / ridging":
        recs.extend([
            "- Monitor blistered or ridged areas for growth or splitting.",
            "- Repair or replace compromised sections where membrane integrity is at risk.",
        ])
    elif issue_type == "Open seam / lap":
        recs.extend([
            "- Clean and properly prepare the seam area.",
            "- Install compatible membrane or flashing repair per manufacturer guidance.",
        ])
    elif issue_type == "Damaged flashing":
        recs.extend([
            "- Remove loose or failed flashing materials.",
            "- Install new flashing with proper terminations and sealant.",
        ])
    elif issue_type == "Membrane puncture / tear":
        recs.extend([
            "- Trim loose or damaged membrane.",
            "- Install a properly sized patch extending beyond the damaged area.",
        ])
    elif issue_type == "Mechanical damage (traffic / tools)":
        recs.extend([
            "- Repair damaged areas and consider adding walkway pads in high-traffic zones.",
        ])
    elif issue_type == "Clogged drain / scupper / gutter":
        recs.extend([
            "- Remove debris and verify that water can drain freely.",
            "- Implement regular housekeeping or maintenance to keep drainage points clear.",
        ])
    elif issue_type == "Debris on roof":
        recs.extend([
            "- Remove debris from roof surface and drainage areas.",
            "- Implement a routine housekeeping schedule.",
        ])
    else:
        recs.extend([
            "- Perform a closer inspection of the affected area and adjacent details.",
            "- Complete localized repairs as required to restore watertightness.",
        ])

    recs.append(
        "- Reinspect after a significant rainfall event to confirm that the issue has been resolved."
    )

    return "\n".join(recs)


def generate_report(
    system: str,
    issue_type: str,
    location: str,
    notes: str,
    active_leak: bool,
    severity_choice: str,
    urgency_choice: str,
):
    notes = (notes or "").strip()

    auto_sev, auto_urg = infer_severity_and_urgency(notes, issue_type, active_leak)

    severity = auto_sev if severity_choice == "Auto" else severity_choice
    urgency = auto_urg if urgency_choice == "Auto" else urgency_choice

    parts = []

    if issue_type != "General moisture concern":
        parts.append(issue_type)
    else:
        parts.append("Moisture-related concern reported")

    if system != "Unknown / Not specified":
        parts.append(f"on a {system} system")

    if location != "Not specified":
        parts.append(f"at/near: {location.lower()}")

    issue_observed = " ".join(parts) + "."

    if notes:
        issue_observed += f"\n\nField notes: {notes}"

    probable_cause = build_probable_cause(system, issue_type, location)
    recommendations = build_recommendations(issue_type, active_leak)

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
    return report, severity, urgency

# -------------- GPT enhancement -------------- #

def enhance_with_gpt(client: OpenAI | None, base_report: str, meta: dict) -> str:
    """
    Ask GPT to rewrite the report for clarity while respecting the facts.
    If client is None (no API key), just return the base report unchanged.
    """
    if client is None:
        return base_report

    system = meta.get("system")
    issue_type = meta.get("issue_type")
    location = meta.get("location")
    severity = meta.get("severity")
    urgency = meta.get("urgency")

    system_prompt = (
        "You are a roofing inspection report editor. "
        "You will receive a draft report and some structured context. "
        "Your job is to rewrite the report so it is clear, concise, and professional.\n\n"
        "IMPORTANT RULES:\n"
        "- Do NOT invent roof conditions or membrane types that are not explicitly given.\n"
        "- Do NOT change the severity or urgency values: keep them exactly the same.\n"
        "- Do NOT add details about locations or components that are not mentioned.\n"
        "- If information is missing or not specified, leave it as general; do NOT guess.\n"
        "- Keep the same general structure: Issue Observed, Probable Cause, Recommendations, Severity, Urgency.\n"
        "- You may improve wording and flow, and you may merge or split sentences for clarity.\n"
    )

    user_content = f"""
Context:
- Roof system: {system}
- Main issue: {issue_type}
- Location: {location}
- Severity: {severity}
- Urgency: {urgency}

Draft report (markdown):
{base_report}

Rewrite this report following the rules above.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Fallback: if anything goes wrong, just return the base report
        return base_report + f"\n\n> Note: AI enhancement failed: {e}"

# -------------- Streamlit UI -------------- #

st.set_page_config(
    page_title="Roof Photo → Report (Demo)",
    page_icon="🧱",
    layout="centered"
)

st.title("🧱 Roof Photo → Report Generator")
st.caption("Turns a roof photo + a few quick selections into a structured inspection-style summary.")

st.markdown("---")

uploaded_image = st.file_uploader(
    "1️⃣ Upload a roof photo (JPEG/PNG)",
    type=["jpg", "jpeg", "png"]
)

system = st.selectbox(
    "2️⃣ Roof system (if known)",
    options=ROOF_SYSTEMS,
    index=0,
)

issue_type = st.selectbox(
    "3️⃣ Main issue observed",
    options=ISSUE_TYPES,
    index=0,
)

location = st.selectbox(
    "4️⃣ Approximate location on roof",
    options=LOCATIONS,
    index=0,
)

active_leak = st.checkbox(
    "Active interior leak reported for this area",
    value=False
)

notes = st.text_area(
    "5️⃣ Optional field notes / observations",
    placeholder=(
        "Example:\n"
        "- Leak reported above unit 302\n"
        "- Ceiling tile staining near window\n"
        "- Existing patch at drain looks aged"
    ),
    height=160
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    severity_choice = st.selectbox(
        "Severity",
        options=["Auto", "Low", "Moderate", "High"],
        index=0,
    )
with col2:
    urgency_choice = st.selectbox(
        "Urgency",
        options=["Auto", "Routine", "Soon", "Immediate"],
        index=0,
    )
with col3:
    show_meta = st.checkbox("Show meta info", value=True)
with col4:
    use_ai = st.checkbox("Enhance wording with AI", value=False)

generate_btn = st.button("⚙️ Generate Report", type="primary")

st.markdown("---")

if generate_btn:
    if uploaded_image is None and not notes.strip():
        st.warning("Please upload at least a photo **or** provide some notes.")
    else:
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            st.subheader("Photo Preview")
            st.image(img, use_container_width=True)

        st.subheader("Generated Report")

        base_report, sev, urg = generate_report(
            system=system,
            issue_type=issue_type,
            location=location,
            notes=notes,
            active_leak=active_leak,
            severity_choice=severity_choice,
            urgency_choice=urgency_choice,
        )

        client = get_openai_client()
        if use_ai and client is None:
            st.info(
                "AI enhancement is enabled, but no OPENAI_API_KEY is configured. "
                "Showing base report instead."
            )
            final_report = base_report
        elif use_ai and client is not None:
            meta = {
                "system": system,
                "issue_type": issue_type,
                "location": location,
                "severity": sev,
                "urgency": urg,
            }
            final_report = enhance_with_gpt(client, base_report, meta)
        else:
            final_report = base_report

        st.markdown(final_report)

        if show_meta:
            st.markdown("---")
            st.caption(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
                "Note: This tool supports field documentation. Final assessment should always be "
                "confirmed by a qualified roofing professional."
            )
