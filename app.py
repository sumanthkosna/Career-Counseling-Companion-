"""
app.py — Streamlit UI for the Agentic Career Counseling Companion
──────────────────────────────────────────────────────────────────
Run with:  streamlit run app.py
"""

import json
import streamlit as st

import config
from career_agent import run_career_counseling

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Career Counseling Companion",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0f3460;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f3460;
        border-left: 4px solid #3b82d4;
        padding-left: 0.6rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .career-card {
        background: #f0f7ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .career-card h3 { color: #1e40af; margin: 0 0 0.3rem 0; }
    .match-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-excellent { background:#dcfce7; color:#166534; }
    .badge-strong    { background:#dbeafe; color:#1e40af; }
    .badge-good      { background:#fef9c3; color:#854d0e; }
    .badge-moderate  { background:#fee2e2; color:#991b1b; }
    .gap-high   { color:#dc2626; font-weight:600; }
    .gap-medium { color:#d97706; font-weight:600; }
    .gap-low    { color:#16a34a; font-weight:600; }
    .phase-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .action-item { padding: 0.3rem 0; }
    .disclaimer {
        background:#fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color:#78350f;
        margin-top: 1.5rem;
    }
    .ibm-badge {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────

def badge_class(match_level: str) -> str:
    mapping = {
        "Excellent Match": "badge-excellent",
        "Strong Match": "badge-strong",
        "Good Match": "badge-good",
        "Moderate Match": "badge-moderate",
    }
    return mapping.get(match_level, "badge-good")


def priority_class(priority: str) -> str:
    return {"High": "gap-high", "Medium": "gap-medium", "Low": "gap-low"}.get(priority, "")


def multiline_input_to_list(text: str) -> list[str]:
    """Convert comma or newline separated input to a clean list."""
    if not text.strip():
        return []
    items = [i.strip() for i in text.replace("\n", ",").split(",") if i.strip()]
    return items


# ── Sidebar — Credential Status ────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    if config.validate_credentials():
        st.success("✅ IBM watsonx credentials loaded")
        st.caption(f"Model: `{config.MODEL_ID}`")
        st.caption(f"URL: `{config.WATSONX_URL}`")
    else:
        st.error("❌ watsonx credentials missing")
        st.info(
            "Copy `.env.example` → `.env` and add your:\n"
            "- `WATSONX_API_KEY`\n"
            "- `WATSONX_PROJECT_ID`\n"
            "- `WATSONX_URL`"
        )

    st.divider()
    st.markdown("## 🏗️ Architecture")
    st.caption(
        "**Agents in pipeline:**\n"
        "1. Academic Performance\n"
        "2. Interest & Preference\n"
        "3. Skill Assessment\n"
        "4. Career Matching *(RAG)*\n"
        "5. Skill Gap Analysis *(RAG)*\n"
        "6. Learning Roadmap *(RAG)*\n"
        "7. Labor Market Trends *(RAG)*\n"
        "8. Final Career Advisor"
    )
    st.caption("🤖 Powered by IBM Granite via watsonx.ai")
    st.divider()
    st.caption("Problem Statement No. 15 — Agentic AI Career Counseling Companion")


# ── Main Header ────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🎓 Agentic Career Counseling Companion</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Powered by IBM watsonx.ai &amp; IBM Granite — '
    'Personalized AI career guidance for every student</div>',
    unsafe_allow_html=True,
)

# ── Student Profile Form ────────────────────────────────────────────────────

st.markdown('<div class="section-header">📋 Student Profile</div>', unsafe_allow_html=True)

with st.form("student_profile_form"):
    # ── Personal Information ──────────────────────────────────────────────
    st.markdown("#### 👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="e.g. Arjun Sharma")
    with col2:
        email = st.text_input("Email (optional)", placeholder="e.g. arjun@example.com")

    # ── Academic Background ───────────────────────────────────────────────
    st.markdown("#### 🎓 Academic Background")
    col1, col2, col3 = st.columns(3)
    with col1:
        degree = st.selectbox(
            "Degree Program",
            ["B.Tech / B.E.", "B.Sc.", "BCA", "B.Com", "MBA", "M.Tech / M.E.", "M.Sc.", "MCA", "Ph.D.", "Other"],
        )
    with col2:
        specialization = st.text_input("Specialization / Branch", placeholder="e.g. Computer Science")
    with col3:
        cgpa = st.text_input("CGPA or Percentage", placeholder="e.g. 8.2 or 78%")

    col1, col2 = st.columns(2)
    with col1:
        academic_subjects = st.text_area(
            "Academic Subjects Studied",
            placeholder="Data Structures, Algorithms, DBMS, Operating Systems, Machine Learning ...",
            height=90,
        )
    with col2:
        col_s, col_w = st.columns(2)
        with col_s:
            strong_subjects = st.text_area(
                "Strong Subjects ✅",
                placeholder="Mathematics, DBMS ...",
                height=90,
            )
        with col_w:
            weak_subjects = st.text_area(
                "Weak Subjects ⚠️",
                placeholder="Networking, OOPS ...",
                height=90,
            )

    # ── Skills ────────────────────────────────────────────────────────────
    st.markdown("#### 🛠️ Skills")
    col1, col2 = st.columns(2)
    with col1:
        technical_skills = st.text_area(
            "Technical Skills",
            placeholder="Python, SQL, TensorFlow, React, Docker ...",
            height=90,
        )
    with col2:
        soft_skills = st.text_area(
            "Soft Skills",
            placeholder="Communication, Teamwork, Problem Solving ...",
            height=90,
        )

    # ── Interests & Preferences ───────────────────────────────────────────
    st.markdown("#### 💡 Interests & Preferences")
    col1, col2, col3 = st.columns(3)
    with col1:
        interests = st.text_area(
            "Interests",
            placeholder="Artificial Intelligence, Web Development, Cybersecurity ...",
            height=90,
        )
    with col2:
        hobbies = st.text_area(
            "Hobbies",
            placeholder="Competitive Programming, Open-source, Photography ...",
            height=90,
        )
    with col3:
        preferred_work_type = st.selectbox(
            "Preferred Work Type",
            [
                "Remote / Work from Home",
                "Hybrid (Remote + Office)",
                "On-site / Office",
                "Flexible / No preference",
                "Start-up environment",
                "Large enterprise / MNC",
                "Research / Academia",
            ],
        )

    career_goal = st.text_area(
        "Career Goal / Dream Role",
        placeholder="e.g. I want to become a Machine Learning Engineer at a top AI company and work on cutting-edge NLP models ...",
        height=80,
    )

    # ── Experience ────────────────────────────────────────────────────────
    st.markdown("#### 📁 Experience & Credentials")
    col1, col2, col3 = st.columns(3)
    with col1:
        projects = st.text_area(
            "Projects",
            placeholder="Sentiment analysis web app, E-commerce site, IoT home automation ...",
            height=90,
        )
    with col2:
        internships = st.text_area(
            "Internships / Work Experience",
            placeholder="6-month ML internship at XYZ Corp, Part-time web dev ...",
            height=90,
        )
    with col3:
        certifications = st.text_area(
            "Certifications",
            placeholder="Google Data Analytics, AWS Cloud Practitioner ...",
            height=90,
        )

    st.divider()

    # ── Submit ────────────────────────────────────────────────────────────
    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        submitted = st.form_submit_button(
            "🔍 Analyze My Career",
            type="primary",
            use_container_width=True,
        )
    with col_note:
        st.caption(
            "Analysis runs 8 AI agents sequentially (~60–120 sec depending on model latency). "
            "Ensure your IBM watsonx credentials are configured before submitting."
        )


# ── Analysis & Report Rendering ────────────────────────────────────────────

if submitted:
    # ── Input validation ─────────────────────────────────────────────────
    if not name.strip():
        st.error("Please enter your name.")
        st.stop()
    if not degree or not specialization.strip():
        st.error("Please select your degree and enter your specialization.")
        st.stop()
    if not config.validate_credentials():
        st.error(
            "IBM watsonx credentials are not configured. "
            "Please add your API key, Project ID, and URL to the `.env` file and restart the app."
        )
        st.stop()

    # ── Build student profile dict ────────────────────────────────────────
    student_profile = {
        "name":               name.strip(),
        "email":              email.strip(),
        "degree":             degree,
        "specialization":     specialization.strip(),
        "cgpa":               cgpa.strip(),
        "academic_subjects":  multiline_input_to_list(academic_subjects),
        "strong_subjects":    multiline_input_to_list(strong_subjects),
        "weak_subjects":      multiline_input_to_list(weak_subjects),
        "technical_skills":   multiline_input_to_list(technical_skills),
        "soft_skills":        multiline_input_to_list(soft_skills),
        "interests":          multiline_input_to_list(interests),
        "hobbies":            multiline_input_to_list(hobbies),
        "preferred_work_type": preferred_work_type,
        "career_goal":        career_goal.strip(),
        "projects":           multiline_input_to_list(projects),
        "internships":        multiline_input_to_list(internships),
        "certifications":     multiline_input_to_list(certifications),
    }

    # ── Progress display ──────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">⚙️ Running Agent Pipeline</div>', unsafe_allow_html=True)

    agent_names = [
        "Academic Performance Agent",
        "Interest & Preference Agent",
        "Skill Assessment Agent",
        "Career Matching Agent",
        "Skill Gap Analysis Agent",
        "Learning Roadmap Agent",
        "Labor Market Trend Agent",
        "Final Career Advisor Agent",
    ]

    progress_bar     = st.progress(0, text="Initialising pipeline…")
    status_container = st.empty()
    agent_statuses   = {name: "⏳ Waiting" for name in agent_names}
    agent_status_placeholder = st.empty()

    def render_agent_table():
        rows = "".join(
            f"<tr><td style='padding:4px 12px;'>{n}</td>"
            f"<td style='padding:4px 12px;'>{s}</td></tr>"
            for n, s in agent_statuses.items()
        )
        agent_status_placeholder.markdown(
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr>"
            f"<th style='text-align:left;padding:4px 12px;color:#555;'>Agent</th>"
            f"<th style='text-align:left;padding:4px 12px;color:#555;'>Status</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )

    render_agent_table()
    completed_count = [0]

    def on_progress(agent_name: str, status: str):
        if status == "running":
            agent_statuses[agent_name] = "🔄 Running…"
        elif status == "complete":
            agent_statuses[agent_name] = "✅ Complete"
            completed_count[0] += 1
            pct = completed_count[0] / len(agent_names)
            progress_bar.progress(pct, text=f"Completed: {agent_name}")
        else:
            agent_statuses[agent_name] = f"❌ {status}"
        render_agent_table()

    # ── Execute pipeline ──────────────────────────────────────────────────
    try:
        with st.spinner("Analysing your career profile with IBM Granite…"):
            report = run_career_counseling(student_profile, on_progress=on_progress)
    except Exception as exc:
        st.error(f"❌ Pipeline error: {exc}")
        st.info(
            "Possible causes:\n"
            "- Invalid IBM watsonx API key or Project ID\n"
            "- Network connectivity issue\n"
            "- Model quota exceeded\n\n"
            "Check your `.env` file and try again."
        )
        st.stop()

    progress_bar.progress(1.0, text="✅ Analysis complete!")
    status_container.success("All 8 agents completed successfully.")

    # ─────────────────────────────────────────────────────────────────────
    # RENDER THE COMPLETE CAREER COUNSELING REPORT
    # ─────────────────────────────────────────────────────────────────────

    st.divider()
    st.markdown(
        f'<div class="main-title">📊 Career Counseling Report — {name}</div>',
        unsafe_allow_html=True,
    )
    meta = report.get("pipeline_metadata", {})
    st.caption(
        f"Model: `{meta.get('model_used','')}` | "
        f"Agents: {meta.get('agents_run','')} | "
        f"Processing time: {meta.get('elapsed_seconds','?')}s | "
        f"Knowledge base files: {len(meta.get('kb_files_used',[]))}"
    )

    final = report.get("final_advice", {})
    career_matches = report.get("career_matches", {})
    skill_gap      = report.get("skill_gap_analysis", {})
    roadmap_data   = report.get("roadmap", {})
    market_data    = report.get("market_analysis", {})
    academic_data  = report.get("academic_analysis", {})

    # ─── 1. Final Recommendation Summary ─────────────────────────────────
    st.markdown('<div class="section-header">🏆 Career Recommendation Summary</div>', unsafe_allow_html=True)
    summary = final.get("final_recommendation_summary", "")
    if summary:
        st.markdown(summary)

    top_rec = final.get("top_recommendation", "")
    if top_rec:
        st.success(f"**Primary Career Recommendation:** {top_rec}")

    # ─── 2. Top 3 Career Matches ──────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Top 3 Suitable Careers</div>', unsafe_allow_html=True)

    top_careers = career_matches.get("top_careers", [])
    if top_careers:
        cols = st.columns(min(len(top_careers), 3))
        for i, career in enumerate(top_careers[:3]):
            with cols[i]:
                ml  = career.get("match_level", "")
                bc  = badge_class(ml)
                reasons = career.get("reasons", [])
                st.markdown(
                    f'<div class="career-card">'
                    f'<h3>#{i+1} {career.get("career_title","")}</h3>'
                    f'<span class="match-badge {bc}">{ml}</span>&nbsp;&nbsp;'
                    f'<strong>{career.get("match_score","?")}% match</strong>'
                    f'<hr style="margin:0.5rem 0;">'
                    + "".join(f"<p style='margin:0.2rem 0;font-size:0.9rem;'>✓ {r}</p>" for r in reasons)
                    + "</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No career match data available.")

    matching_summary = career_matches.get("matching_summary", "")
    if matching_summary:
        st.caption(matching_summary)

    # ─── 3. Academic Profile ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Academic Profile Analysis</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        score = academic_data.get("academic_strength_score", "-")
        st.metric("Academic Strength Score", f"{score} / 10")
    with col2:
        band = academic_data.get("cgpa_percentile_band", "-")
        st.metric("CGPA Band", band)
    with col3:
        strong = academic_data.get("strong_areas", [])
        st.metric("Strong Areas", len(strong))

    acad_summary = academic_data.get("academic_summary", "")
    if acad_summary:
        st.markdown(acad_summary)

    acad_insights = academic_data.get("academic_insights", [])
    if acad_insights:
        with st.expander("📖 Academic Insights"):
            for insight in acad_insights:
                st.markdown(f"- {insight}")

    # ─── 4. Student Strengths ─────────────────────────────────────────────
    st.markdown('<div class="section-header">💪 Student Strengths</div>', unsafe_allow_html=True)
    strengths = final.get("key_strengths", []) or skill_gap.get("student_strengths", [])
    if strengths:
        cols = st.columns(min(len(strengths), 4))
        for i, strength in enumerate(strengths):
            with cols[i % 4]:
                st.success(f"✅ {strength}")
    else:
        st.info("Strength analysis not available.")

    # ─── 5. Skill Gap Analysis ────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Skill Gap Analysis</div>', unsafe_allow_html=True)

    gap_summary = skill_gap.get("skill_gap_summary", "")
    if gap_summary:
        st.markdown(gap_summary)

    prioritised_gaps = skill_gap.get("prioritised_gaps", [])
    if prioritised_gaps:
        st.markdown("**Prioritised Skill Gaps:**")
        cols = st.columns(3)
        col_headers = {"High": 0, "Medium": 1, "Low": 2}
        for priority_label, col_idx in col_headers.items():
            with cols[col_idx]:
                st.markdown(f"**{priority_label} Priority**")
                found = [g for g in prioritised_gaps if g.get("priority") == priority_label]
                for g in found:
                    pc = priority_class(g.get("priority", ""))
                    st.markdown(
                        f'<p class="{pc}" style="margin:0.2rem 0;">⚡ {g.get("skill","")}'
                        f'</p><small style="color:#666;">{g.get("reason","")}</small>',
                        unsafe_allow_html=True,
                    )
                if not found:
                    st.caption("None")

    gaps_by_career = skill_gap.get("gaps_by_career", [])
    if gaps_by_career:
        with st.expander("🔎 Gaps per Career Path"):
            for gbc in gaps_by_career:
                sev = gbc.get("gap_severity", "")
                sev_color = {"High": "#dc2626", "Medium": "#d97706", "Low": "#16a34a"}.get(sev, "#555")
                st.markdown(
                    f"**{gbc.get('career_title','')}** — "
                    f"<span style='color:{sev_color};font-weight:600;'>{sev} Gap Severity</span>",
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2)
                with col1:
                    missing_tech = gbc.get("missing_technical_skills", [])
                    if missing_tech:
                        st.caption("Technical Skills Needed: " + ", ".join(missing_tech))
                with col2:
                    missing_soft = gbc.get("missing_soft_skills", [])
                    if missing_soft:
                        st.caption("Soft Skills Needed: " + ", ".join(missing_soft))
                st.divider()

    # ─── 6. Personalized Learning Roadmap ────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Personalized Learning Roadmap</div>', unsafe_allow_html=True)

    roadmap_summary = roadmap_data.get("roadmap_summary", "")
    total_duration  = roadmap_data.get("total_estimated_duration", "")
    if roadmap_summary:
        st.markdown(roadmap_summary)
    if total_duration:
        st.info(f"⏱️ Estimated Total Duration: **{total_duration}**")

    phases = roadmap_data.get("phases", [])
    if phases:
        for phase in phases:
            with st.expander(
                f"📌 {phase.get('phase_name','')} — {phase.get('duration','')}",
                expanded=True,
            ):
                col1, col2 = st.columns(2)
                with col1:
                    focus_areas = phase.get("focus_areas", [])
                    if focus_areas:
                        st.markdown("**Focus Areas:**")
                        for f in focus_areas:
                            st.markdown(f"- {f}")
                    resources = phase.get("resources", [])
                    if resources:
                        st.markdown("**Resources:**")
                        for r in resources:
                            st.markdown(f"- 📚 {r}")
                with col2:
                    milestones = phase.get("milestones", [])
                    if milestones:
                        st.markdown("**Milestones:**")
                        for m in milestones:
                            st.markdown(f"- 🎯 {m}")

    # ─── 7. Recommended Projects ──────────────────────────────────────────
    rec_projects = roadmap_data.get("recommended_projects", [])
    if rec_projects:
        st.markdown('<div class="section-header">🧪 Recommended Projects</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(rec_projects), 2))
        for i, proj in enumerate(rec_projects):
            with cols[i % 2]:
                st.markdown(f"**🔧 {proj.get('title','')}**")
                desc = proj.get("description", "")
                if desc:
                    st.caption(desc)
                skills = proj.get("skills_practiced", [])
                if skills:
                    st.caption("Skills: " + " · ".join(skills))
                st.markdown("---")

    # ─── 8. Recommended Certifications ───────────────────────────────────
    rec_certs = roadmap_data.get("recommended_certifications", [])
    if rec_certs:
        st.markdown('<div class="section-header">🏅 Recommended Certifications</div>', unsafe_allow_html=True)
        cert_cols = st.columns(min(len(rec_certs), 3))
        for i, cert in enumerate(rec_certs):
            with cert_cols[i % 3]:
                name_c    = cert.get("name", cert.get("title", ""))
                provider  = cert.get("provider", "")
                level     = cert.get("level", "")
                when_to   = cert.get("when_to_pursue", "")
                st.markdown(
                    f"🎓 **{name_c}**  \n"
                    f"Provider: {provider}  \n"
                    f"Level: {level}"
                    + (f"  \nTiming: {when_to}" if when_to else "")
                )

    # ─── 9. Interview Preparation ─────────────────────────────────────────
    st.markdown('<div class="section-header">🎤 Interview Preparation</div>', unsafe_allow_html=True)
    interview_prep = final.get("interview_preparation", {})
    col1, col2 = st.columns(2)
    with col1:
        core_topics = interview_prep.get("core_topics", [])
        if core_topics:
            st.markdown("**Core Interview Topics:**")
            for t in core_topics:
                st.markdown(f"- 📌 {t}")
        tips = interview_prep.get("tips", [])
        if tips:
            st.markdown("**Preparation Tips:**")
            for tip in tips:
                st.markdown(f"- 💡 {tip}")
    with col2:
        prep_resources = interview_prep.get("preparation_resources", [])
        if prep_resources:
            st.markdown("**Recommended Preparation Resources:**")
            for r in prep_resources:
                st.markdown(f"- 📖 {r}")

    # ─── 10. Labor Market Insights ────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Labor Market Insights</div>', unsafe_allow_html=True)
    market_summary = market_data.get("market_summary", "")
    if market_summary:
        st.markdown(market_summary)

    career_market_items = market_data.get("career_market_data", [])
    if career_market_items:
        mkt_cols = st.columns(min(len(career_market_items), 3))
        for i, cm in enumerate(career_market_items[:3]):
            with mkt_cols[i]:
                demand = cm.get("demand_trend", "N/A")
                growth = cm.get("yoy_growth", "N/A")
                remote = cm.get("remote_opportunity", "N/A")
                sectors = cm.get("hiring_sectors", [])
                emerging = cm.get("emerging_skills", [])
                st.markdown(f"**{cm.get('career_title','')}**")
                st.metric("Demand", demand)
                st.metric("YoY Growth", f"{growth}%")
                st.caption(f"Remote: {remote}")
                if sectors:
                    st.caption("Hiring: " + " · ".join(sectors[:3]))
                if emerging:
                    st.caption("Emerging: " + " · ".join(emerging[:3]))

    strategic_rec = market_data.get("strategic_recommendation", "")
    if strategic_rec:
        st.info(f"💡 **Strategic Insight:** {strategic_rec}")

    emerging_skills = market_data.get("emerging_skills_to_watch", [])
    if emerging_skills:
        with st.expander("🔭 Emerging Skills to Watch"):
            for sk in emerging_skills:
                st.markdown(f"- 🚀 {sk}")

    # ─── 11. Immediate Action Plan ────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ Immediate Action Plan</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🗓️ Next 30 Days:**")
        actions_30 = final.get("action_plan_30_days", [])
        if actions_30:
            for a in actions_30:
                st.markdown(f'<div class="action-item">☑️ {a}</div>', unsafe_allow_html=True)
        else:
            st.caption("No 30-day plan generated.")
    with col2:
        st.markdown("**📅 Next 90 Days:**")
        actions_90 = final.get("action_plan_90_days", [])
        if actions_90:
            for a in actions_90:
                st.markdown(f'<div class="action-item">☑️ {a}</div>', unsafe_allow_html=True)
        else:
            st.caption("No 90-day plan generated.")

    # ─── Encouragement Note ───────────────────────────────────────────────
    encouragement = final.get("encouragement_note", "")
    if encouragement:
        st.info(f"✨ {encouragement}")

    # ─── Disclaimer ───────────────────────────────────────────────────────
    disclaimer = final.get(
        "disclaimer",
        "This report is for guidance only. Career outcomes depend on many individual and "
        "external factors and cannot be guaranteed by this system.",
    )
    st.markdown(
        f'<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> {disclaimer}</div>',
        unsafe_allow_html=True,
    )

    # ─── Download Report ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💾 Download Report")
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=report_json,
        file_name=f"career_report_{name.replace(' ','_').lower()}.json",
        mime="application/json",
        use_container_width=True,
    )

    # ─── IBM / Power footer ───────────────────────────────────────────────
    st.markdown(
        '<div class="ibm-badge">Powered by IBM watsonx.ai · IBM Granite · '
        'Agentic Career Counseling Companion · Problem Statement No. 15</div>',
        unsafe_allow_html=True,
    )

else:
    # ── Landing state when form is not yet submitted ──────────────────────
    st.info(
        "👆 Fill in your student profile above and click **Analyze My Career** "
        "to receive your personalized AI-powered career counseling report."
    )

    st.markdown("### What you'll receive:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**🎯 Career Matching**
- Top 3 suitable career paths
- Match score & level
- Specific matching reasons
        """)
    with col2:
        st.markdown("""
**🔍 Gap & Roadmap**
- Prioritised skill gaps (H/M/L)
- Phased learning roadmap
- Project recommendations
- Certifications to pursue
        """)
    with col3:
        st.markdown("""
**📈 Market & Action**
- Labor market insights
- Emerging skills to watch
- 30-day & 90-day action plan
- Interview preparation guide
        """)

    st.markdown(
        '<div class="ibm-badge">Powered by IBM watsonx.ai · IBM Granite · '
        'Agentic Career Counseling Companion · Problem Statement No. 15</div>',
        unsafe_allow_html=True,
    )
