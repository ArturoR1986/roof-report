import os
import json
import streamlit as st
from datetime import datetime
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError  # ✅ added

# -------------------- OpenAI client -------------------- #

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

# -------------------- Prompting -------------------- #

NORMALIZE_SYSTEM_PROMPT = """
You are an assistant for roofing service documentation.
Your job: take messy field notes (often incomplete, shorthand, or voice transcript)
and produce a clean structured record WITHOUT inventing facts.

CRITICAL RULES:
- Do NOT guess membrane type, roof system, building details, or causes if not explicitly stated.
- If something is unknown, set it to "Not specified".
- If the note is unclear, ask clarifying questions.
- Use only information present in the notes; you may rephrase for clarity.
- Output MUST be valid JSON only, no extra text.

Return JSON with exactly these keys:
{
  "job_summary": string,
  "roof_system": string,              // e.g., "TPO", "SBS modified bitumen", or "Not specified"
  "primary_issue": string,            // e.g., "Active leak", "Ponding", "Open seam", or "Not specified"
  "location": string,                 // e.g., "at drain", "at penetration", or "Not specified"
  "active_leak_reported": boolean,
  "observations": [string],           // bullets; only what is supported by notes
  "constraints_or_unknowns": [string],// missing info, unclear items
  "recommended_next_steps": [string], // practical steps; keep conservative
  "severity": "Low"|"Moderate"|"High",
  "urgency": "Routine"|"Soon"|"Immediate",
  "clarifying_questions": [string]    // ask only if needed
}
"""

REPORT_TEMPLATE_HEADER = """### 1. Issue Observed
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

def safe_json_load(text: str):
    # If the model returns non-JSON, this will throw and be caught upstream.
    return json.loads(text)

def normalize_notes_with_gpt(client: OpenAI, notes: str):
    """
    Returns: (data_or_None, raw_json_or_None, err_message_or_None)
    Never throws to Streamlit UI.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": NORMALIZE_SYSTEM_PROMPT},
                {"role": "user", "content": notes},
            ],
            temperature=0.2,
            max_tokens=800,  # ✅ cost / verbosity control
        )
        raw = completion.choices[0].message.content
        data = safe_json_load(raw)
        return data, raw, None

    except RateLimitError:
        return None, None, (
            "AI is temporarily unavailable due to usage/rate limits (often billing/credits/limits). "
            "Use Manual Normalize below or try again later."
        )
    except (APITimeoutError, APIError) as e:
        return None, None, f"AI service error: {e}"
    except json.JSONDecodeError:
        return None, None, (
            "AI returned a response that was not valid JSON. "
            "Try again, or use Manual Normalize below."
        )
    except Exception as e:
        return None, None, f"Unexpected error: {e}"

def build_probable_cause(structured: dict) -> str:
    issue = (structured.get("primary_issue") or "Not specified").lower()
    roof_system = structured.get("roof_system") or "Not specified"
    location = structured.get("location") or "Not specified"

    if issue in ["active leak", "leak", "active interior leak", "active leak at interior"]:
        base = ("Water entry is reported. The source may be near a roof detail or condition in the vicinity, "
                "but requires confirmation on-site.")
    elif "pond" in issue:
        base = ("Ponding/standing water is indicated. This is commonly associated with drainage limitations, "
                "slope conditions, or obstructions, and should be confirmed by inspection.")
    elif "seam" in issue or "open lap" in issue:
        base = ("An opening or weakness at seams/laps is indicated. Seam integrity issues can lead to water entry "
                "and should be verified and repaired per system requirements.")
    elif "flashing" in issue:
        base = ("A flashing/detail concern is indicated. Movement, aging sealant, or termination issues can contribute "
                "to leakage risk and should be inspected.")
    else:
        base = ("Cause is not specified in the notes and should be confirmed through closer inspection of the area "
                "and adjacent details.")

    extras = []
    if roof_system != "Not specified":
        extras.append(f"Roof system noted: {roof_system}.")
    if location != "Not specified":
        extras.append(f"Location noted: {location}.")

    if extras:
        return base + " " + " ".join(extras)
    return base

def build_issue_observed(structured: dict) -> str:
    parts = []
    issue = structured.get("primary_issue") or "Not specified"
    roof_system = structured.get("roof_system") or "Not specified"
    location = structured.get("location") or "Not specified"
    job_summary = structured.get("job_summary") or ""

    if job_summary.strip():
        parts.append(job_summary.strip())
    else:
        parts.append(f"Primary issue: {issue}.")

    if roof_system != "Not specified":
        parts.append(f"Roof system: {roof_system}.")
    if location != "Not specified":
        parts.append(f"Location: {location}.")

    obs = structured.get("observations") or []
    if obs:
        parts.append("\nObservations:")
        for o in obs:
            parts.append(f"- {o}")

    unknowns = structured.get("constraints_or_unknowns") or []
    if unknowns:
        parts.append("\nUnknown / needs confirmation:")
        for u in unknowns:
            parts.append(f"- {u}")

    return "\n".join(parts)

def build_recommendations(structured: dict) -> str:
    steps = structured.get("recommended_next_steps") or []
    if not steps:
        steps = [
            "Perform closer inspection of the reported area and surrounding details.",
            "Complete localized repairs as required to restore watertightness.",
            "Reinspect after the next significant rainfall event."
        ]
    return "\n".join([f"- {s}" for s in steps])

def build_report(structured: dict) -> str:
    issue_observed = build_issue_observed(structured)
    probable_cause = build_probable_cause(structured)
    recommendations = build_recommendations(structured)
    severity = structured.get("severity") or "Moderate"
    urgency = structured.get("urgency") or "Soon"

    return REPORT_TEMPLATE_HEADER.format(
        issue_observed=issue_observed,
        probable_cause=probable_cause,
        recommendations=recommendations,
        severity=severity,
        urgency=urgency
    )

# -------------------- Streamlit UI -------------------- #

st.set_page_config(page_title="Roof Notes → Report", page_icon="🧱", layout="centered")

st.title("🧱 Roof Notes → Report Generator")
st.caption("Turns messy field notes into a consistent report. (Editor-first approach)")

client = get_client()
if client is None:
    st.warning("OPENAI_API_KEY not found. AI Normalize will not run. Manual Normalize is available below.")
else:
    st.success("AI is available (OPENAI_API_KEY found).")

st.markdown("---")

notes = st.text_area(
    "Paste messy notes / voice transcript",
    placeholder="Example: leak above 302, staining ceiling tile, old patch near curb. was raining last night. check RTU area.",
    height=180
)

colA, colB = st.columns(2)
with colA:
    normalize_btn = st.button("🧹 Normalize Notes (AI)", type="primary")
with colB:
    generate_btn = st.button("📝 Generate Report")

if "structured" not in st.session_state:
    st.session_state.structured = None

# ---- AI Normalize ----
if normalize_btn:
    if not notes.strip():
        st.error("Paste some notes first.")
    elif client is None:
        st.error("AI not available. Set OPENAI_API_KEY first (Streamlit Cloud Secrets).")
    else:
        with st.spinner("Normalizing notes…"):
            structured, raw_json, err = normalize_notes_with_gpt(client, notes)

        if err:
            st.error(err)
            st.info("Tip: Use Manual Normalize below to keep working even if AI is unavailable.")
        else:
            st.session_state.structured = structured
            st.subheader("Normalized Record (structured)")
            st.json(st.session_state.structured)

            questions = structured.get("clarifying_questions") or []
            if questions:
                st.info("Clarifying questions (answering these will improve accuracy):")
                for q in questions:
                    st.write(f"- {q}")

# ---- Manual Normalize (fallback) ----
st.markdown("---")
st.subheader("Manual Normalize (fallback)")

m_col1, m_col2 = st.columns(2)
with m_col1:
    m_roof_system = st.selectbox(
        "Roof system",
        ["Not specified", "TPO", "EPDM", "PVC", "SBS modified bitumen", "BUR", "Metal", "Shingle"]
    )
    m_primary_issue = st.selectbox(
        "Primary issue",
        ["Not specified", "Active leak", "Ponding", "Open seam / lap", "Flashing concern",
         "Puncture / tear", "Clogged drain / scupper", "Debris", "Moisture concern"]
    )

with m_col2:
    m_location = st.selectbox(
        "Location",
        ["Not specified", "Field", "Perimeter", "At drain / scupper", "At penetration",
         "At parapet / wall detail", "At roof edge / metal"]
    )
    m_active_leak = st.checkbox("Active leak reported", value=False)

m_obs = st.text_area(
    "Observations (bullets or sentences)",
    height=120,
    placeholder="- What did you see?\n- What did the customer report?"
)
m_steps = st.text_area(
    "Recommended next steps (optional)",
    height=120,
    placeholder="- What should happen next?"
)

m_severity = st.selectbox("Severity", ["Low", "Moderate", "High"], index=1)
m_urgency = st.selectbox("Urgency", ["Routine", "Soon", "Immediate"], index=1)

manual_btn = st.button("✅ Use Manual Normalized Data")
if manual_btn:
    st.session_state.structured = {
        "job_summary": "Manual entry provided.",
        "roof_system": m_roof_system,
        "primary_issue": m_primary_issue,
        "location": m_location,
        "active_leak_reported": bool(m_active_leak),
        "observations": [x.strip("- ").strip() for x in m_obs.splitlines() if x.strip()],
        "constraints_or_unknowns": [],
        "recommended_next_steps": [x.strip("- ").strip() for x in m_steps.splitlines() if x.strip()] if m_steps.strip() else [],
        "severity": m_severity,
        "urgency": m_urgency,
        "clarifying_questions": [],
    }
    st.success("Manual normalized record loaded.")
    st.json(st.session_state.structured)

# ---- Generate Report ----
if generate_btn:
    if st.session_state.structured is None:
        st.error("Click 'Normalize Notes (AI)' or 'Use Manual Normalized Data' first.")
    else:
        st.subheader("Generated Report")
        report = build_report(st.session_state.structured)
        st.markdown(report)

        st.markdown("---")
        st.caption(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
            "Note: This tool supports documentation. Final assessment should be confirmed by a qualified roofing professional."
        )
