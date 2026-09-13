# 🎓 Agentic Career Counseling Companion
### Problem Statement No. 15 — IBM watsonx.ai & IBM Granite

An intelligent, multi-agent AI career counselor that analyses a student's academic performance, interests, skills, career goals, skill gaps, and labor-market trends to provide deeply personalised career recommendations — powered by **IBM watsonx.ai** and **IBM Granite**.

---

## 🏗️ Architecture Overview

```
Student Profile (Streamlit UI)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│            Career Counselor / Orchestrator Agent         │
│                   (career_agent.py)                      │
│                                                          │
│  1. Academic Performance Agent  (academic_agent.py)      │
│  2. Interest & Preference Agent (interest_agent.py)      │
│  3. Skill Assessment Agent      (skill_agent.py)         │
│  4. Career Matching Agent       (career_matching_agent)  │◄──┐
│  5. Skill Gap Analysis Agent    (skill_gap_agent.py)     │   │
│  6. Learning Roadmap Agent      (roadmap_agent.py)       │   │ RAG
│  7. Labor Market Trend Agent    (market_agent.py)        │   │ Knowledge
│  8. Final Career Advisor Agent  (career_agent.py)        │   │ Base
└─────────────────────────────────────────────────────────┘   │
                   │                                           │
                   ▼                                           │
         Complete Career Report                    ┌──────────┴──────────┐
                                                   │   knowledge_base/    │
                                                   │  career_paths.json   │
                                                   │  career_req.json     │
                                                   │  skills_database.json│
                                                   │  learning_res.json   │
                                                   │  certifications.json │
                                                   │  interview_topics.json│
                                                   │  market_trends.json  │
                                                   └─────────────────────┘
```

**AI Model:** IBM Granite 3.3 8B Instruct via IBM watsonx.ai SDK  
**Orchestration:** Python-based `AgentPipeline` class (maps directly to IBM watsonx Orchestrate skills/flows)  
**RAG:** JSON knowledge base loaded at inference time via `knowledge_base_loader.py`

---

## 📁 Project Structure

```
career-counseling-companion/
│
├── app.py                      # Streamlit UI — student form + report renderer
├── career_agent.py             # Orchestrator + Final Career Advisor Agent
├── academic_agent.py           # Academic Performance Agent
├── interest_agent.py           # Interest & Preference Agent
├── skill_agent.py              # Skill Assessment Agent
├── career_matching_agent.py    # Career Matching Agent (RAG)
├── skill_gap_agent.py          # Skill Gap Analysis Agent (RAG)
├── roadmap_agent.py            # Learning Roadmap Agent (RAG)
├── market_agent.py             # Labor Market Trend Agent (RAG)
├── knowledge_base_loader.py    # Central RAG KB loader
├── config.py                   # Configuration (env vars)
│
├── knowledge_base/
│   ├── career_paths.json       # 10 career paths with metadata
│   ├── career_requirements.json # Skills & certs per career
│   ├── skills_database.json    # Skills taxonomy
│   ├── learning_resources.json # Curated learning resources
│   ├── certifications.json     # Certifications per career
│   ├── interview_topics.json   # Interview prep per career
│   └── market_trends.json      # Labor market data 2025
│
├── requirements.txt
├── .env.example                # Template — copy to .env
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone / Download the Project

```bash
git clone <repo-url>
cd career-counseling-companion
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure IBM watsonx Credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
WATSONX_API_KEY=your_ibm_watsonx_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
MODEL_ID=ibm/granite-3-3-8b-instruct
```

**How to get credentials:**
1. Log in at [cloud.ibm.com](https://cloud.ibm.com)
2. Create an IBM watsonx.ai service instance
3. **API Key:** Manage → Access → API keys → Create
4. **Project ID:** watsonx.ai → Projects → Your project → Manage → General → Project ID
5. **URL:** Use the endpoint for your region (default: `https://us-south.ml.cloud.ibm.com`)

### 5. Run the Application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🤖 Agents & Workflow

| # | Agent | File | Role |
|---|-------|------|------|
| 1 | Academic Performance | `academic_agent.py` | Analyses degree, CGPA, subjects, strong/weak areas |
| 2 | Interest & Preference | `interest_agent.py` | Maps interests, hobbies, work style to career domains |
| 3 | Skill Assessment | `skill_agent.py` | Evaluates tech/soft skills, projects, experience |
| 4 | Career Matching | `career_matching_agent.py` | RAG-based top-3 career recommendations |
| 5 | Skill Gap Analysis | `skill_gap_agent.py` | H/M/L priority gaps vs career requirements |
| 6 | Learning Roadmap | `roadmap_agent.py` | Phased roadmap with resources & milestones |
| 7 | Labor Market Trend | `market_agent.py` | Market demand, growth, emerging skills |
| 8 | Final Career Advisor | `career_agent.py` | Synthesises all into final counseling report |

**Orchestrator:** `AgentPipeline` in `career_agent.py` runs agents sequentially, passing context forward.

---

## 📊 Report Contents

The generated report includes:

- ✅ **Top 3 Career Recommendations** — match score, level, specific reasons
- 💪 **Student Strengths** — identified from academic + skill analysis
- 🔍 **Skill Gaps** — High / Medium / Low priority with reasoning
- 🗺️ **Learning Roadmap** — 3 phases with focus areas, resources, milestones
- 🧪 **Recommended Projects** — hands-on projects to build the skill set
- 🏅 **Recommended Certifications** — with provider, level, and timing
- 🎤 **Interview Preparation** — topics, resources, and tips
- 📈 **Labor Market Insights** — demand, YoY growth, remote opportunities
- ⚡ **30-day & 90-day Action Plan** — concrete steps to take immediately
- 💾 **Downloadable JSON Report** — full structured report for reference

---

## 🔧 IBM watsonx Orchestrate Integration

The project is structured so each agent function maps cleanly to an **IBM watsonx Orchestrate skill**:

```python
# Each agent exposes a single callable that can be wrapped as an Orchestrate skill:
from academic_agent        import analyze_academic_performance   # Skill 1
from interest_agent        import analyze_interests              # Skill 2
from skill_agent           import assess_skills                  # Skill 3
from career_matching_agent import match_careers                  # Skill 4 (RAG)
from skill_gap_agent       import analyze_skill_gaps             # Skill 5 (RAG)
from roadmap_agent         import generate_roadmap               # Skill 6 (RAG)
from market_agent          import analyze_market_trends          # Skill 7 (RAG)
# Final advisor is in career_agent.py                            # Skill 8
```

To deploy on IBM watsonx Orchestrate:
1. Package each agent as a skill using the Orchestrate Builder
2. Define the pipeline flow in the Orchestrate Flow Designer
3. Connect the Granite model as the AI backend for each skill
4. Replace the `AgentPipeline` class with Orchestrate flow invocation

---

## 🛠️ Supported Career Paths (Knowledge Base)

| Career | Domain | Demand |
|--------|--------|--------|
| Software Engineer | Technology | Very High |
| Data Scientist | Data & AI | Very High |
| AI / ML Engineer | Artificial Intelligence | Very High |
| Cybersecurity Analyst | Security | Very High |
| Cloud Engineer | Cloud Computing | High |
| Product Manager | Product & Strategy | High |
| UX / UI Designer | Design | High |
| Data Engineer | Data & Analytics | High |
| Business / Systems Analyst | Business & Technology | Moderate |
| Embedded Systems / IoT Engineer | Hardware & IoT | Moderate |

---

## 📋 Student Profile Fields

| Category | Fields |
|----------|--------|
| Personal | Name, Email |
| Academic | Degree, Specialization, CGPA, Subjects, Strong/Weak Subjects |
| Skills | Technical Skills, Soft Skills |
| Preferences | Interests, Hobbies, Preferred Work Type, Career Goal |
| Experience | Projects, Internships, Certifications |

---

## 🔒 Security & Privacy

- All IBM watsonx credentials are loaded from environment variables — **never hard-coded**
- The `.env` file is excluded from version control via `.gitignore`
- Student data is processed in-memory and not stored externally
- No student PII is logged or persisted

---

## 🛠️ Customisation

**Add a new career path:**
1. Add entry to `knowledge_base/career_paths.json`
2. Add requirements to `knowledge_base/career_requirements.json`
3. Add certifications to `knowledge_base/certifications.json`
4. Add interview topics to `knowledge_base/interview_topics.json`
5. Add market data to `knowledge_base/market_trends.json`

**Change the AI model:**
Set `MODEL_ID` in `.env` to any supported IBM Granite model:
- `ibm/granite-3-3-8b-instruct` (default, fast)
- `ibm/granite-3-8b-instruct`
- `ibm/granite-13b-instruct-v2`

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `ibm-watsonx-ai` | IBM watsonx.ai Python SDK (Granite model inference) |
| `streamlit` | Web UI framework |
| `python-dotenv` | Load `.env` credentials |
| `pandas` | Data handling utilities |
| `pydantic` | Data validation |

---

## ⚠️ Disclaimer

This application is for **educational and guidance purposes only**. Career recommendations are based on the information provided by the student and the knowledge base — they do not constitute professional career advice, guarantee employment, or predict salary outcomes. Students should consult qualified career counselors for comprehensive guidance.

---

*Agentic Career Counseling Companion — Problem Statement No. 15*  
*Powered by IBM watsonx.ai & IBM Granite*
