import streamlit as st
import pdfplumber
import docx
import os
import json
from groq import Groq
import random

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st.set_page_config(page_title="AI Resume Ranker", page_icon="🚀", layout="centered")
STRONG_WORDS = ["Built","Developed","Designed","Implemented","Architected","Optimized","Led","Created"]
# ------------------- FILE EXTRACTION -------------------
def extract_text(file):
    try:
        name = file.name.lower()
        if name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            return text.strip()
        elif name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join(p.text for p in doc.paragraphs if p.text)
        elif name.endswith(".txt"):
            return file.read().decode("utf-8")
        return ""
    except:
        return ""

# ---------------- SAFE JSON -------------------
def safe_json(content):
    try:
        return json.loads(content)
    except:
        try:
            cleaned = content.replace("```json","").replace("```","").strip()
            return json.loads(cleaned)
        except:
            return {
                "skills": [],
                "projects": [],
                "internship": {"type":"none"},
                "experience": {"years":0,"company_type":"none"},
                "cgpa": {"value":0},
                "certifications": {"top_tier":0,"coursera":0,"college":0},
                "action_words": [],
                "sections": [],
                "keywords_resume": [],
                "keywords_jd": []
            }

# ---------------- EXTRACT STRUCTURE DATA----------------
def extract_structured_data(resume_text, jd):
    prompt = f"""
Extract structured resume data.
Return ONLY JSON:
{{
  "skills": [{{"name": "", "relevance": "primary/secondary/peripheral", "proof": "strong/medium/weak"}}],
  "projects": [{{"name": "", "has_github": false, "has_live": false, "complexity": "high/medium/low", "relevance": "direct/related/unrelated"}}],
  "internship": {{"type": "paid_reputed/paid_small/unpaid_good/unpaid_basic/virtual/none"}},
  "experience": {{"years": 0, "company_type": "mnc/startup/small/none"}},
  "cgpa": {{"value": 0}},
  "certifications": {{"top_tier": 0, "coursera": 0, "college": 0}},
  "action_words": [],
  "sections": [],
  "keywords_resume": [],
  "keywords_jd": []
}}
JOB:
{jd}
RESUME:
{resume_text}
"""
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[{"role":"user","content":prompt}]
    )
    return safe_json(res.choices[0].message.content)

# ---------------- ROAST AI GENERATOR ----------------
def generate_roast(score, text):
    prompt = f"""
Roast resume in 2 lines only.
Score: {score}/10
{text}
"""
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.9,
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content.strip()

# ---------------- SCORING ----------------
def score_skills(data, jd=None):
    skills = data.get("skills", [])
    if skills and isinstance(skills[0], str):
        skills = [{"name": s, "relevance": "secondary", "proof": "medium"} for s in skills]
    if not skills:
        return 0, "Weak skills", ""
    weights = []
    jd_text = jd.lower() if isinstance(jd, str) and jd else None
    core_skills = ["java", "sql", "oop", "data structures", "jdbc"]
    for s in skills:
        name = s.get("name", "").lower()
        relevance = s.get("relevance", "secondary")
        proof = s.get("proof", "medium")
        R = {"primary": 1, "secondary": 0.7, "peripheral": 0.4}.get(relevance, 0.5)
        P = {"strong": 1, "medium": 0.7, "weak": 0.4}.get(proof, 0.5)
        base = R * P
        boost = 1.0
        if relevance == "primary" and proof == "strong":
            boost += 0.10
        if jd_text and name in jd_text:
            boost += 0.05
        if name in core_skills:
            boost += 0.05
        base *= boost
        importance = {
            "primary": 1.5,
            "secondary": 1.0,
            "peripheral": 0.6
        }.get(relevance, 1)
        weights.append(base * importance)
    weights = sorted(weights, reverse=True)
    top_weights = weights[:5]
    Q = (sum(top_weights) / len(top_weights)) * 0.95
    peripheral_count = sum(1 for s in skills if s.get("relevance") == "peripheral")
    if peripheral_count > len(skills) * 0.4:
        Q *= 0.8
    # v- volume
    V = (
        1.0 if len(skills) >= 8 else
        0.9 if len(skills) >= 5 else
        0.8
    )
    raw_score = 3 * Q * V
    raw_score *= 0.88
    raw_score = raw_score / (1 + 0.10 * raw_score)
    return round(raw_score, 1), "Skills Evaluated", ""

def score_projects(data, jd):
    total = 0
    jd_text = jd.lower() if isinstance(jd, str) and jd else ""
    for p in data.get("projects", []):
        base = 0
        if p.get("complexity") == "high":
            base = 0.9
        elif p.get("complexity") == "medium":
            base = 0.65
        else:
            base = 0.45
        if p.get("has_live"):
            base += 0.25
        if p.get("has_github"):
            base += 0.2
        if jd_text:
            project_name = p.get("name", "").lower()
            if any(k in jd_text for k in ["web", "frontend", "react", "html", "css", "javascript"]):
                if any(k in project_name for k in ["web", "app", "dashboard", "frontend"]):
                    base += 0.2
            elif any(k in jd_text for k in ["data", "ml", "machine learning", "analysis"]):
                if any(k in project_name for k in ["ml", "data", "prediction", "analysis", "nlp"]):
                    base += 0.2
            elif any(k in jd_text for k in ["software", "api", "backend", "system"]):
                if any(k in project_name for k in ["api", "system", "backend"]):
                    base += 0.15
        total += min(base, 1)
    return round(min(total, 3), 1), total, None

def score_experience(data):
    y=data.get("experience",{}).get("years",0)
    c=data.get("experience",{}).get("company_type","none")
    if y>=5 and c=="mnc": return 1.7
    elif y>=3: return 1.2
    elif y>=1: return 0.6
    return 0

def score_cgpa(v):
    return 1 if v>=9 else 0.8 if v>=8 else 0.6 if v>=7 else 0.4 if v>=6 else 0.2

def score_cert(c):
    if c.get("top_tier",0)>=2: return 1
    if c.get("top_tier",0)>=1: return 0.8
    if c.get("coursera",0)>=1: return 0.6
    return 0.4

def score_action(words):
    return 0.5 if any(w.capitalize() in STRONG_WORDS for w in words) else 0.3 if words else 0

def score_sections(sec):
    return 0.5 if len(sec)>=6 else 0.3

def score_ats(data, jd=""):
    resume_skills = set(k.lower() for k in data.get("keywords_resume", []))
    jd_skills = set(k.lower() for k in data.get("keywords_jd", [])) if jd else set()
    if jd_skills:
        match = len(resume_skills & jd_skills)
        keyword_score = (match / len(jd_skills)) * 40 if jd_skills else 0
    else:
        keyword_score = min(len(resume_skills) * 2.5, 40)
    skills = data.get("skills", [])
    skill_score = 0
    for s in skills:
        r = {"primary": 1.0, "secondary": 0.7, "peripheral": 0.4}.get(s.get("relevance"), 0.5)
        p = {"strong": 1.0, "medium": 0.7, "weak": 0.4}.get(s.get("proof"), 0.5)
        skill_score += r * p
    skill_score = min(skill_score, 20)
    project_score = 0
    for p in data.get("projects", []):
        base = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(p.get("complexity", "low"), 0.4)
        if p.get("has_github"):
            base += 0.2
        if p.get("has_live"):
            base += 0.3
        project_score += min(base, 1)
    project_score = min(project_score * 3, 15)
    exp = data.get("experience", {})
    years = exp.get("years", 0)
    if years >= 5:
        exp_score = 10
    elif years >= 3:
        exp_score = 7
    elif years >= 1:
        exp_score = 4
    else:
        exp_score = 1
    cgpa = data.get("cgpa", {}).get("value", 0)
    cgpa_score = 5 if cgpa >= 9 else 4 if cgpa >= 8 else 3 if cgpa >= 7 else 2 if cgpa >= 6 else 1
    cert = data.get("certifications", {})
    cert_score = min(cert.get("top_tier", 0) * 2 + cert.get("coursera", 0), 10)
    academic_score = min(cgpa_score + cert_score, 15)
    if jd_skills:
        coverage = len(resume_skills & jd_skills) / len(jd_skills)
        jd_score = coverage * 10
    else:                                
        jd_score = 6   # no JD → neutral scoring  # default balanced score
    total = (
        keyword_score +
        skill_score +
        project_score +
        exp_score +
        academic_score +
        jd_score
    )
    return round(min(total, 100), 1)

COURSES = {
    "python":"Python for Everybody | Coursera | https://coursera.org",
    "sql":"SQL Basics | Udemy | https://udemy.com",
    "machine learning":"ML Crash Course | Google | https://developers.google.com",
    "data analysis":"Data Analysis | freeCodeCamp | https://freecodecamp.org",
    "api":"REST API Guide | Postman | https://postman.com"
}

def course_suggestions(skills):
    result = []

    for k in skills[:6]:
        key = k.lower().strip()
        if key in COURSES:
            name, platform, link = COURSES[key]
        else:
            name = k.title()
            if "machine" in key or "ai" in key:
                platform = "Coursera"
                link = f"https://www.coursera.org/search?query={k.replace(' ','%20')}"
            elif "data" in key or "analysis" in key:
                platform = "freeCodeCamp"
                link = f"https://www.freecodecamp.org/news/search/?query={k.replace(' ','%20')}"
            else:
                platform = "YouTube"
                link = f"https://www.youtube.com/results?search_query={k.replace(' ','+')}+full+course"
        result.append(f"{name} → {platform} → {link}")
    return result
# ---------------- ANALYSIS ----------------
def analyze(data, jd, user_type, text):
    s, _, _ = score_skills(data, jd)
    p, _, _ = score_projects(data, jd)
    if user_type == "Fresher":
        exp_label = "Internship"
        internship = data.get("internship", {}).get("type", "none")
        exp_score = {"paid_reputed":1, "paid_small":0.8, "unpaid_good":0.6}.get(internship, 0)
        exp_val = exp_score
    else:
        exp_label = "Experience"
        exp_score = score_experience(data)
        exp_val = exp_score
    cg  = score_cgpa(data.get("cgpa", {}).get("value", 0))
    ce  = score_cert(data.get("certifications", {}))
    ac  = score_action(data.get("action_words", []))
    sec = score_sections(data.get("sections", []))
    base_score = s + p + exp_score + cg + ce + ac + sec
    if jd and jd.strip():   # ONLY if JD is provided 
        ats = score_ats(data, jd)
        jd_factor = 0.7 + (ats / 100) * 0.3
        total = base_score * jd_factor
    else:
        ats = score_ats(data)  
        total = base_score      
    if user_type == "Fresher" and total < 9:
        total = min(total, 8.9)
    total = round(min(total, 9.5), 1)
    strengths = []
    if s >= 2.5:
        strengths.append("Excellent technical skills")
    elif s >= 2:
        strengths.append("Strong technical skills")
    if p >= 2.5:
        strengths.append("High-impact projects")
    elif p >= 2:
        strengths.append("Good project work")
    if exp_score >= 0.8:
        strengths.append("Strong practical experience")
    elif exp_score >= 0.6:
        strengths.append("Some practical experience")
    if cg >= 0.8:
        strengths.append("Good academics")
    if ce >= 0.8:
        strengths.append("Strong certifications")
    if ac >= 0.5:
        strengths.append("Good use of action words")
    if sec >= 0.5:
        strengths.append("Well-structured resume")
    strengths.append("Overall balanced profile")
    weaknesses = []
    if s < 1:
        weaknesses.append("Weak technical skills")
    elif s < 1.5:
        weaknesses.append("Improve technical skills")
    if p < 1:
        weaknesses.append("Lack of projects")
    elif p < 1.5:
        weaknesses.append("Improve project quality")
    if exp_score == 0:
        weaknesses.append("No practical experience")
    if cg < 0.6:
        weaknesses.append("Low academic score")
    if ce < 0.6:
        weaknesses.append("Add certifications")
    if ac < 0.3:
        weaknesses.append("Use strong action words")
    if sec < 0.3:
        weaknesses.append("Improve resume structure")
    if total >= 9:
        weaknesses.append("Optimize for top-tier roles (FAANG-level projects, system design)")
        weaknesses.append("Increase impact with large-scale real-world applications")
    elif total >= 8.5:
        weaknesses.append("Add one high-impact live project with real users")
        weaknesses.append("Strengthen profile with advanced certifications or specialization")
    elif total >= 7:
        weaknesses.append("Improve project depth and practical exposure")
    else:
        weaknesses.append("Focus on building strong fundamentals and real projects")
    skills_list = [x.get("name","") for x in data.get("skills",[]) if x.get("name")]
    jd_skills = data.get("keywords_jd", [])
    missing = list(set(jd_skills) - set(skills_list))
    roast = generate_roast(total, text)
    return f"""
SCORE: {total}/10
GRADE: {'Excellent' if total>=9 else 'Good' if total>=7 else 'Average'}
STRONG POINTS:
{chr(10).join(f"- {x}" for x in strengths)}
WEAK POINTS:
{chr(10).join(f"- {x}" for x in weaknesses)}
SKILL GAP:
YOU HAVE: {skills_list}
YOU NEED: {missing}
Skills: {s}/3
Projects: {p}/3
{exp_label}: {exp_val}
CGPA: {cg}/1
Certifications: {ce}/1
Action Words: {ac}/0.5
Sections: {sec}/0.5
ATS SCORE: {ats}%
ROAST:
{roast}
"""
# ---------------- UI ----------------
if "page" not in st.session_state:
    st.session_state.page = "upload"
if st.session_state.page == "upload":
    st.markdown("""
        <style>
        .block-container { max-width: 1200px !important; padding-top: 3rem !important; }
        .centered-text { text-align: center; }
        /* This ensures the radio buttons themselves are centered in their column */
        div[data-testid="stRadio"] > div {
            justify-content: center !important;
            display: flex !important;
        }
        /* Styling for the Profile Header */
        .profile-header {
            text-align: center;
            font-weight: bold;
            font-size: 20px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='centered-text'>🚀 AI Resume Ranker</h1>", unsafe_allow_html=True)
    st.markdown("<p class='centered-text'>Professional Analysis • Smart Suggestions • Career Roadmap</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='profile-header'>Profile Experience Level</div>", unsafe_allow_html=True)
    c1, c2, c3, col_radio, c5, c6, c7 = st.columns([2, 1, 1, 4, 1, 1, 1])
    
    with col_radio:
        user_type = st.radio(
            "Profile Type", 
            ["Fresher" + "&nbsp;"*35, "&nbsp;"*5 + "Experienced"], 
            horizontal=True, 
            label_visibility="collapsed"
        )
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_body, _ = st.columns([1, 5, 1])
    with col_body:
        st.markdown("<h3 class='centered-text'>📝 Job Description</h3>", unsafe_allow_html=True)
        jd = st.text_area("JD", placeholder="Paste the job requirements here...", height=150, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 class='centered-text'>📂 Upload Resume</h3>", unsafe_allow_html=True)
        file = st.file_uploader("Upload", type=["pdf", "docx", "txt"], label_visibility="collapsed")
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Analyze My Resume", use_container_width=True):
            if file:
                st.session_state.user_type = user_type.replace("&nbsp;", "").strip()
                st.session_state.jd = jd
                st.session_state.file_text = extract_text(file)
                st.session_state.page = "result"
                st.rerun()
    st.markdown("""
        <div style='text-align: center; color: gray; margin-top: 50px;'>
            <hr style='width: 30%; margin: auto; border: 0.1px solid #333;'>
            <br>AI Resume Ranker v3.0 | 2026
        </div>
    """, unsafe_allow_html=True)
elif st.session_state.page == "result":
    with st.spinner("Analysing your resume..."):
        text      = st.session_state.file_text
        jd        = st.session_state.jd
        user_type = st.session_state.user_type
        data = extract_structured_data(text, jd if jd else "")
        raw  = analyze(data, jd, user_type, text)
    lines = raw.strip().splitlines()

    def grab(prefix):
        for l in lines:
            if l.strip().startswith(prefix):
                return l.split(":",1)[-1].strip()
        return ""
    score = grab("SCORE")
    grade = grab("GRADE")
    ats   = grab("ATS SCORE")
    strong, weak, roast = [], [], []
    section = ""
    you_have_raw = ""
    you_need_raw = ""
    for l in lines:
        s = l.strip()
        if "STRONG POINTS" in s:      section="strong"; continue
        if "WEAK POINTS"   in s:      section="weak";   continue
        if "ROAST"         in s:      section="roast";  continue
        if s.startswith("YOU HAVE:"): you_have_raw = s.replace("YOU HAVE:","").strip(); continue
        if s.startswith("YOU NEED:"): you_need_raw = s.replace("YOU NEED:","").strip(); continue
        if section=="strong" and s.startswith("-"):
            strong.append(s[1:].strip())
        elif section=="weak" and s.startswith("-"):
            weak.append(s[1:].strip())
        elif section=="roast" and s:
            roast.append(s)

    import re
    def parse_list(raw):
        items = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw)
        return [a or b for a, b in items]
    you_have = parse_list(you_have_raw)
    you_need = parse_list(you_need_raw)

    RELATED_SKILLS_MAP = {
        "html":             ["css", "javascript", "bootstrap", "tailwind", "accessibility", "seo"],
        "css":              ["html", "javascript", "sass", "tailwind", "styled-components", "flexbox"],
        "javascript":       ["typescript", "react", "nodejs", "html", "css", "es6+", "jest"],
        "typescript":       ["javascript", "react", "angular", "nodejs", "nestjs", "typeorm"],
        "react":            ["javascript", "typescript", "nextjs", "redux", "react-query", "material-ui"],
        "nextjs":           ["react", "javascript", "typescript", "nodejs", "vercel", "ssr"],
        "angular":          ["typescript", "javascript", "rxjs", "ngrx", "angular material"],
        "vue":              ["javascript", "typescript", "nuxt", "vuex", "vuetify"],
        "nodejs":           ["javascript", "express", "mongodb", "api", "microservices", "socket.io"],
        "express":          ["nodejs", "javascript", "mongodb", "api", "middleware", "jwt"],
        "python":           ["django", "flask", "pandas", "numpy", "sql", "fastapi", "pytest"],
        "django":           ["python", "sql", "postgresql", "rest framework", "celery", "redis"],
        "flask":            ["python", "sql", "api", "jinja2", "sqlalchemy"],
        "java":             ["spring", "sql", "maven", "hibernate", "junit", "microservices"],
        "spring":           ["java", "sql", "docker", "kubernetes", "cloud", "security"],
        "kotlin":           ["java", "android", "spring", "coroutines"],
        "android":          ["kotlin", "java", "firebase", "jetpack compose", "retrofit"],
        "swift":            ["ios", "xcode", "firebase", "swiftui", "combine"],
        "sql":              ["mysql", "postgresql", "mongodb", "data analysis", "normalization", "indexing"],
        "mysql":            ["sql", "postgresql", "python", "php", "database design"],
        "postgresql":       ["sql", "python", "django", "docker", "postgis"],
        "mongodb":          ["nodejs", "express", "python", "api", "nosql", "aggregation"],
        "php":              ["mysql", "html", "css", "laravel", "symfony", "wordpress"],
        "machine learning": ["python", "pandas", "numpy", "tensorflow", "deep learning", "scikit-learn", "pytorch"],
        "deep learning":    ["tensorflow", "python", "nlp", "pytorch", "neural networks", "computer vision"],
        "tensorflow":       ["python", "machine learning", "deep learning", "keras", "tensorboard"],
        "nlp":              ["python", "machine learning", "tensorflow", "spacy", "nltk", "transformers"],
        "data analysis":    ["python", "pandas", "sql", "tableau", "power bi", "excel", "statistics"],
        "pandas":           ["python", "numpy", "sql", "data analysis", "matplotlib", "seaborn"],
        "numpy":            ["python", "pandas", "machine learning", "scipy", "linear algebra"],
        "tableau":          ["data analysis", "excel", "sql", "power bi", "dashboarding"],
        "power bi":         ["excel", "sql", "data analysis", "tableau", "dax"],
        "excel":            ["sql", "power bi", "python", "vlookup", "pivots"],
        "r":                ["python", "data analysis", "tableau", "sql", "ggplot2"],
        "docker":           ["kubernetes", "linux", "git", "aws", "ci/cd", "containers"],
        "kubernetes":       ["docker", "aws", "linux", "devops", "helm", "terraform"],
        "aws":              ["docker", "linux", "python", "terraform", "s3", "lambda", "ec2"],
        "git":              ["linux", "docker", "github actions", "gitlab", "version control"],
        "linux":            ["git", "docker", "aws", "shell scripting", "bash", "ubuntu"],
        "api":              ["nodejs", "python", "postman", "docker", "swagger", "graphql"],
        "redis":            ["nodejs", "python", "docker", "mongodb", "caching"],
        "graphql":          ["api", "nodejs", "react", "typescript", "apollo"],
        "firebase":         ["javascript", "react", "android", "nodejs", "firestore", "auth"],
        "selenium":         ["python", "java", "javascript", "api", "cypress", "unit testing"],
        "c++":              ["data structures", "algorithms", "python", "java", "qt", "opengl"],
        "c":                ["c++", "data structures", "linux", "algorithms", "embedded systems"],
        "data structures":  ["algorithms", "python", "java", "c++", "big o notation"],
    }

    COURSE_DB = {
        "python":           ("Python",           "freeCodeCamp", "https://www.freecodecamp.org/learn/scientific-computing-with-python/"),
        "java":             ("Java",             "freeCodeCamp", "https://www.freecodecamp.org/news/learn-java-free-java-courses-for-beginners/"),
        "javascript":       ("JavaScript",       "freeCodeCamp", "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/"),
        "typescript":       ("TypeScript",       "freeCodeCamp", "https://www.freecodecamp.org/news/learn-typescript-beginners-guide/"),
        "mysql":            ("MySQL",            "freeCodeCamp", "https://www.freecodecamp.org/news/learn-sql-free-relational-database-courses-for-beginners/"),
        "sql":              ("SQL",              "freeCodeCamp", "https://www.freecodecamp.org/news/learn-sql-free-relational-database-courses-for-beginners/"),
        "html":             ("HTML",             "freeCodeCamp", "https://www.freecodecamp.org/learn/responsive-web-design/"),
        "css":              ("CSS",              "freeCodeCamp", "https://www.freecodecamp.org/learn/responsive-web-design/"),
        "bootstrap":        ("Bootstrap",        "freeCodeCamp", "https://www.freecodecamp.org/news/full-bootstrap-5-tutorial-for-beginners/"),
        "tailwind":         ("Tailwind CSS",     "freeCodeCamp", "https://www.freecodecamp.org/news/what-is-tailwind-css-a-beginners-guide/"),
        "sass":             ("Sass/SCSS",        "freeCodeCamp", "https://www.freecodecamp.org/news/the-beginners-guide-to-sass/"),
        "react":            ("React",            "freeCodeCamp", "https://www.freecodecamp.org/learn/front-end-development-libraries/"),
        "nextjs":           ("Next.js",          "freeCodeCamp", "https://www.freecodecamp.org/news/nextjs-tutorial/"),
        "redux":            ("Redux",            "freeCodeCamp", "https://www.freecodecamp.org/news/redux-tutorial-for-beginners/"),
        "angular":          ("Angular",          "freeCodeCamp", "https://www.freecodecamp.org/news/angular-tutorial-course/"),
        "vue":              ("Vue.js",           "freeCodeCamp", "https://www.freecodecamp.org/news/vue-js-full-course/"),
        "node":             ("Node.js",          "freeCodeCamp", "https://www.freecodecamp.org/news/free-8-hour-node-express-course/"),
        "nodejs":           ("Node.js",          "freeCodeCamp", "https://www.freecodecamp.org/news/free-8-hour-node-express-course/"),
        "express":          ("Express.js",       "freeCodeCamp", "https://www.freecodecamp.org/news/free-8-hour-node-express-course/"),
        "machine learning": ("Machine Learning", "Google",       "https://developers.google.com/machine-learning/crash-course"),
        "ml":               ("Machine Learning", "Google",       "https://developers.google.com/machine-learning/crash-course"),
        "deep learning":    ("Deep Learning",    "fast.ai",      "https://www.fast.ai/"),
        "tensorflow":       ("TensorFlow",       "TensorFlow",   "https://www.tensorflow.org/tutorials"),
        "nlp":              ("NLP",              "Hugging Face", "https://huggingface.co/learn/nlp-course/chapter1/1"),
        "data analysis":    ("Data Analysis",    "Kaggle",       "https://www.kaggle.com/learn/pandas"),
        "pandas":           ("Pandas",           "Kaggle",       "https://www.kaggle.com/learn/pandas"),
        "numpy":            ("NumPy",            "Kaggle",       "https://www.kaggle.com/learn/intro-to-programming"),
        "django":           ("Django",           "freeCodeCamp", "https://www.freecodecamp.org/news/learn-django-3-and-start-creating-websites-with-python/"),
        "flask":            ("Flask",            "freeCodeCamp", "https://www.freecodecamp.org/news/how-to-use-flask-every-layout/"),
        "spring":           ("Spring Boot",      "freeCodeCamp", "https://www.freecodecamp.org/news/spring-boot-tutorial/"),
        "php":              ("PHP",              "freeCodeCamp", "https://www.freecodecamp.org/news/the-php-handbook/"),
        "c++":              ("C++",              "freeCodeCamp", "https://www.freecodecamp.org/news/learn-c-with-free-31-hour-course/"),
        "c":                ("C",               "freeCodeCamp", "https://www.freecodecamp.org/news/learn-c-with-free-31-hour-course/"),
        "git":              ("Git & GitHub",     "freeCodeCamp", "https://www.freecodecamp.org/news/git-and-github-for-beginners/"),
        "docker":           ("Docker",           "freeCodeCamp", "https://www.freecodecamp.org/news/the-docker-handbook/"),
        "kubernetes":       ("Kubernetes",       "freeCodeCamp", "https://www.freecodecamp.org/news/learn-kubernetes-in-under-3-hours/"),
        "aws":              ("AWS",              "AWS Training", "https://explore.skillbuilder.aws/learn"),
        "linux":            ("Linux",            "freeCodeCamp", "https://www.freecodecamp.org/news/the-linux-commands-handbook/"),
        "mongodb":          ("MongoDB",          "MongoDB Uni",  "https://university.mongodb.com/"),
        "postgresql":       ("PostgreSQL",       "freeCodeCamp", "https://www.freecodecamp.org/news/postgresql-full-course/"),
        "redis":            ("Redis",            "Redis Uni",    "https://university.redis.com/"),
        "api":              ("REST APIs",        "Postman",      "https://learning.postman.com/"),
        "graphql":          ("GraphQL",          "freeCodeCamp", "https://www.freecodecamp.org/news/learn-graphql-i/"),
        "data structures":  ("DSA",              "freeCodeCamp", "https://www.freecodecamp.org/learn/coding-interview-prep/"),
        "algorithms":       ("Algorithms",       "freeCodeCamp", "https://www.freecodecamp.org/learn/coding-interview-prep/"),
        "excel":            ("Excel",            "freeCodeCamp", "https://www.freecodecamp.org/news/how-to-use-microsoft-excel-beginner-to-advanced/"),
        "power bi":         ("Power BI",         "Microsoft",    "https://learn.microsoft.com/en-us/training/powerplatform/power-bi"),
        "tableau":          ("Tableau",          "Tableau",      "https://www.tableau.com/learn/training"),
        "r":                ("R Language",       "Kaggle",       "https://www.kaggle.com/learn/r"),
        "selenium":         ("Selenium",         "freeCodeCamp", "https://www.freecodecamp.org/news/web-scraping-python-tutorial-how-to-scrape-data-from-a-website/"),
        "swift":            ("Swift",            "Apple Dev",    "https://developer.apple.com/tutorials/swiftui"),
        "kotlin":           ("Kotlin",           "JetBrains",    "https://kotlinlang.org/docs/getting-started.html"),
        "android":          ("Android Dev",      "Google",       "https://developer.android.com/courses"),
        "firebase":         ("Firebase",         "Google",       "https://firebase.google.com/docs"),
        "maven":            ("Maven",            "freeCodeCamp", "https://www.freecodecamp.org/news/maven-tutorial-build-lifecycle-phases-commands-plugins-and-much-more/"),
        "hibernate":        ("Hibernate",        "freeCodeCamp", "https://www.freecodecamp.org/news/hibernate-tutorial/"),
        "shell scripting":  ("Shell Scripting",  "freeCodeCamp", "https://www.freecodecamp.org/news/shell-scripting-crash-course-how-to-write-bash-scripts-in-linux/"),
        "postman":          ("Postman API",      "Postman",      "https://learning.postman.com/docs/getting-started/introduction/"),
        "terraform":        ("Terraform",        "freeCodeCamp", "https://www.freecodecamp.org/news/what-is-terraform-learn-infrastructure-as-code/"),
    }
    def get_course(skill):
        key = skill.lower().strip()
        if key in COURSE_DB:
            return COURSE_DB[key]
        for k, v in COURSE_DB.items():
            if k in key or key in k:
                return v
        return (
            skill.title(),
            "YouTube",
            f"https://www.youtube.com/results?search_query={skill.replace(' ','+')}+tutorial+free"
        )

    jd_lower = jd.lower() if jd else ""
    seen = set()
    must_learn   = []
    upgrade      = []
    also_learn   = []
    strengthen   = []

    for sk in you_need:
        k = sk.lower()
        if k not in seen:
            must_learn.append(sk)
            seen.add(k)

    for sk in you_have:
        k = sk.lower()
        if k not in seen:
            if jd_lower and k in jd_lower:
                upgrade.append(sk)
            else:
                strengthen.append(sk)
            seen.add(k)

    for sk in you_have:
        related = RELATED_SKILLS_MAP.get(sk.lower(), [])
        for rel in related:
            if rel not in seen:
                also_learn.append(rel)
                seen.add(rel)
    st.markdown("""
        <style>
        /* Maximize content area for wider view */
        .block-container { max-width: 1400px !important; padding: 2rem !important; }
        .course-row { padding: 8px 0; border-bottom: 1px solid #333; }
        .badge { padding: 4px 12px; border-radius: 20px; font-size: 11px;
                 font-weight: 700; color: white; display: inline-block; text-transform: uppercase; }
        .metric-card { background: #1a1c24; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("## 📊 Detailed Resume Analysis & Roadmap")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🎯 Match Score",  score)
    with c2: st.metric("🏅 Profile Grade",  grade)
    with c3: st.metric("🤖 ATS Readiness",    ats)
    st.divider()
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("### ✅ Strengths")
        if strong:
            for x in strong:
                st.success(f"**{x}**")
        else:
            st.info("No major strengths highlighted.")

    with right_col:
        st.markdown("### ⚠️ Improvement Areas")
        if weak:
            for x in weak:
                st.warning(f"**{x}**")
        else:
            st.info("Strong profile — consider adding more advanced improvements to stand out")
            st.success("Tip  :  Add one live project or certification 🚀")
    st.divider()
    st.subheader("📚 Personalized Learning Pathways")
    st.markdown("""
        <div style='display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px;'>
            <span class='badge' style='background:#d63031;'>🔴 Must Learn</span>
            <span class='badge' style='background:#e17055;'>🟡 Upgrade</span>
            <span class='badge' style='background:#0984e3;'>🔵 Also Learn</span>
            <span class='badge' style='background:#00b894;'>🟢 Strengthen</span>
        </div>
    """, unsafe_allow_html=True)
    def render_course_row(idx, skill, label, color):
        course_name, platform, url = get_course(skill)
        cols = st.columns([0.5, 3, 2, 1.5, 2])
        cols[0].markdown(f"**{idx}**")
        cols[1].markdown(f"**{course_name}**")
        cols[2].markdown(f"`{platform}`")
        cols[3].markdown(f"<span class='badge' style='background:{color};'>{label}</span>", unsafe_allow_html=True)
        cols[4].markdown(f"[🔗 Enroll Now]({url})")
    idx = 1
    if must_learn:
        with st.expander("🔴 High Priority: Missing Skills for Job Match", expanded=True):
            for sk in must_learn[:8]:
                render_course_row(idx, sk, "Must Learn", "#d63031")
                idx += 1

    if upgrade:
        with st.expander("🟡 Medium Priority: Advanced Mastery Needed", expanded=True):
            for sk in upgrade[:6]:
                render_course_row(idx, sk, "Upgrade", "#e17055")
                idx += 1

    if also_learn:
        with st.expander("🔵 Career Growth: Smart Suggestions Based on Your Profile", expanded=False):
            st.caption("These complementary skills will make your profile more competitive.")
            for sk in also_learn[:10]:
                render_course_row(idx, sk, "Also Learn", "#0984e3")
                idx += 1

    if strengthen:
        with st.expander("🟢 Maintenance: Deepen Current Skills", expanded=False):
            for sk in strengthen[:6]:
                render_course_row(idx, sk, "Strengthen", "#00b894")
                idx += 1
    st.divider()
    st.subheader("🔥 The Brutal Truth (Roast)")
    st.error(" ".join(roast) or "You're too boring to even roast.")
    st.divider()
    if st.button("🔄 Start New Analysis", use_container_width=True):
        st.session_state.page = "upload"
        st.rerun()