"""
Seeds Launchpad with:
  - Demo accounts for all three roles (student, researcher, admin)
  - A spread of realistic final-year projects across the 7 supported departments,
    including Computer Science projects with live-demo and source-code links
  - A mix of approval states (approved / pending / rejected-with-note) so the
    review workflow has something to show off immediately
  - A couple of sample discussion threads (question + answer)
"""

import sqlite3
from datetime import datetime

from werkzeug.security import generate_password_hash

from app import init_db, DB_PATH, auto_summary, SIMILARITY_FLAG_THRESHOLD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMCU = "Eswatini Medical Christian University"

DEMO_PASSWORD = "launchpad123"

USERS = [
    dict(name="Olwethu Dlamini", email="student@demo.com", role="student"),
    dict(name="Harmony N'senda", email="researcher@demo.com", role="researcher"),
    dict(name="Mr. M. Zwane", email="admin@demo.com", role="admin"),
]

# Every project defaults to approved unless overridden below, so the public
# catalog isn't empty on first run.
PROJECTS = [
    dict(
        title="Assessment of Quality and Pharmaceutical Equivalence of Selected Ciprofloxacin Brands Marketed in Eswatini",
        abstract="Ciprofloxacin is a quinolone-derivative antibiotic with a broad spectrum of activity used to treat urinary tract, respiratory tract, skin, and soft-tissue infections. The rise in generic drug products from multiple sources has increased the risk of counterfeit and substandard medicines reaching prescribers who substitute one brand for another. This study assessed three ciprofloxacin tablet brands sold in Eswatini, purchased from local wholesalers, subjecting them to standard physicochemical tests of weight uniformity, hardness, disintegration, dissolution, and friability to determine pharmaceutical equivalence and interchangeability.",
        student_name="Harmony N'senda",
        supervisor="Department of Pharmacy Faculty",
        keywords="ciprofloxacin, pharmaceutical equivalence, quality control, quinolone antibiotics, dissolution testing",
        department="Pharmacy",
        institution=EMCU,
        topic_area="Health",
        year=2024,
        status="Completed",
        owner_email="researcher@demo.com",
    ),
    dict(
        title="Job Recruitment System",
        abstract="Machine learning plays a growing role in modern human-resource procurement. This project uses supervised learning to classify resumes and job descriptions as 'qualified' or 'not qualified' for a given role, with natural language processing extracting structured information from unstructured resume text. The system aims to improve efficiency in the hiring pipeline, reduce screening cost, and help organisations identify the most suitable candidates based on skills and attributes rather than keyword-matching alone.",
        student_name="Olwethu Dlamini",
        supervisor="Mr. M. Zwane",
        keywords="machine learning, natural language processing, recruitment, hr technology, resume classification",
        department="Computer Science",
        institution=EMCU,
        topic_area="Higher Education",
        year=2026,
        status="Ongoing",
        live_demo_url="https://job-recruitment-demo.onrender.com",
        github_url="https://github.com/olwethu-dlamini/job-recruitment-system",
        owner_email="student@demo.com",
    ),
    dict(
        title="Secure Biometric Authentication on ATMs",
        abstract="Traditional password-based authentication for ATMs is vulnerable to hacking, phishing, and unauthorized access. This project presents a biometric authentication system combining facial recognition and fingerprint scanning. The system captures a live face image and compares it against a stored reference; only on a match does the user proceed to fingerprint verification against a local database. Both biometric checks must succeed before authentication, providing a stronger alternative to PIN-based access that resists brute-force and social-engineering attacks.",
        student_name="James Baker",
        supervisor="Mr. Makhubu",
        keywords="biometric authentication, facial recognition, fingerprint scanning, atm security, banking technology",
        department="Computer Science",
        institution=EMCU,
        topic_area="Finance",
        year=2025,
        status="Completed",
        github_url="https://github.com/jbaker/secure-atm-biometrics",
        owner_email="student@demo.com",
    ),
    dict(
        title="Hospital Wait Time Optimizer",
        abstract="Public health facilities in Eswatini frequently face long, unpredictable patient queues that strain staff and frustrate patients. This project builds a queue-management and forecasting system that models arrival patterns per department and predicts expected wait times in real time, allowing clinics to redistribute staff to the busiest service points. A lightweight web dashboard displays live queue length and projected wait per department, deployed for demonstration at a partner clinic.",
        student_name="Nomvula Simelane",
        supervisor="Mr. M. Zwane",
        keywords="queueing systems, healthcare operations, forecasting, real-time dashboard, public health",
        department="Computer Science",
        institution=EMCU,
        topic_area="Health",
        year=2026,
        status="Ongoing",
        live_demo_url="https://hospital-wait-optimizer.onrender.com",
        github_url="https://github.com/nsimelane/hospital-wait-optimizer",
        owner_email="student@demo.com",
        approval_status="pending",  # demonstrates the review queue
    ),
    dict(
        title="Understanding Vulvovaginal Candidiasis Co-infections with Sexually Transmitted Infections in Sub-Saharan Africa",
        abstract="Vulvovaginal candidiasis (VVC) is a common mucosal infection of the lower female reproductive tract, with an estimated 75% of women experiencing at least one episode in their lifetime. Drawing on PLOS, PubMed, and open-access articles published between 2008 and 2023, this review examines the incidence of VVC alongside co-infections and genital-tract infections across Sub-Saharan Africa. A pooled co-infection prevalence ranging from 9.6% in South Africa to 34.1% in Namibia was found, with Chlamydia trachomatis, Trichomonas vaginalis, and Neisseria gonorrhoeae the most commonly observed co-infecting organisms.",
        student_name="Latembe Tembe",
        supervisor="Etando Ayuk (MSc)",
        keywords="vulvovaginal candidiasis, sexually transmitted infections, sub-saharan africa, co-infections, genital tract infection",
        department="Medical Laboratory",
        institution=EMCU,
        topic_area="Health",
        year=2023,
        status="Completed",
        owner_email="researcher@demo.com",
    ),
    dict(
        title="Medication Adherence Among Chronic Illness Patients in Rural Nursing Clinics",
        abstract="Poor medication adherence remains a major driver of avoidable complications among chronic-illness patients in rural Eswatini, where clinic distance and irregular follow-up compound the problem. This study surveyed nursing staff and patients across three rural clinics to identify the leading barriers to adherence, ranging from transport cost to inconsistent stock availability, and proposes a structured follow-up protocol nurses can apply during routine visits to improve long-term outcomes.",
        student_name="Precious Motsa",
        supervisor="Mrs. T. Nkambule",
        keywords="medication adherence, chronic illness, rural healthcare, nursing intervention, patient follow-up",
        department="Nursing Sciences",
        institution=EMCU,
        topic_area="Health",
        year=2025,
        status="Completed",
        owner_email="student@demo.com",
    ),
    dict(
        title="Academic Stress and Coping Mechanisms Among Final-Year University Students",
        abstract="Final-year students face compounding academic and financial pressure that can affect mental wellbeing and performance. This study surveyed final-year students across three faculties to identify the most common stressors and the coping strategies students report using, comparing adaptive strategies such as peer support and time-management against maladaptive strategies such as avoidance, and recommends targeted interventions the university counselling office could adopt.",
        student_name="Sipho Khumalo",
        supervisor="Dr. N. Vilakati",
        keywords="academic stress, coping mechanisms, student mental health, higher education, counselling",
        department="Psychology",
        institution=EMCU,
        topic_area="Higher Education",
        year=2024,
        status="Completed",
        owner_email="researcher@demo.com",
    ),
    dict(
        title="Radiographer Exposure to Ionising Radiation in Rural District Hospitals",
        abstract="Radiographers in rural district hospitals often work with ageing equipment and inconsistent access to dosimetry monitoring, raising concerns about cumulative occupational radiation exposure. This study measured recorded dosimeter readings across four district hospitals over a six-month period, compared them against national safety thresholds, and surveyed radiographers on protective-equipment availability and shielding practices, finding gaps concentrated in the two hospitals with the oldest fluoroscopy units.",
        student_name="Andile Simelane",
        supervisor="Mr. S. Fakudze",
        keywords="radiography, radiation safety, occupational exposure, dosimetry, medical imaging",
        department="Radiography",
        institution=EMCU,
        topic_area="Health",
        year=2025,
        status="Completed",
        owner_email="student@demo.com",
    ),
    dict(
        title="Case Management Outcomes for Youth Aging Out of Foster Care",
        abstract="Youth transitioning out of foster care at the age of majority often lose structured support at the exact point they need it most. This study reviewed case files and conducted interviews with 22 care-leavers to assess how case-management intensity in the final year of care correlates with housing stability, continued education, and employment outcomes 12 months after exit, and proposes a step-down support model for social workers to phase out involvement more gradually.",
        student_name="Fikile Mamba",
        supervisor="Mrs. P. Shongwe",
        keywords="social work, foster care, case management, youth transition, care leavers",
        department="Social Work",
        institution=EMCU,
        topic_area="Community",
        year=2025,
        status="Completed",
        owner_email="researcher@demo.com",
    ),
    dict(
        title="Crop Disease Detection from Leaf Images Using Convolutional Neural Networks",
        abstract="Smallholder farmers often lack timely access to agricultural extension officers who can diagnose crop disease early enough to prevent yield loss. This project trains a convolutional neural network on a labelled dataset of maize leaf images to classify common regional diseases, packaged behind a simple mobile-friendly interface where a farmer can photograph a leaf and receive an instant diagnosis with a recommended treatment, evaluated against extension-officer diagnoses on a held-out field sample.",
        student_name="Sanele Dube",
        supervisor="Mr. M. Zwane",
        keywords="convolutional neural networks, computer vision, crop disease, precision agriculture, mobile app",
        department="Computer Science",
        institution=EMCU,
        topic_area="Agriculture",
        year=2026,
        status="Ongoing",
        live_demo_url="https://crop-disease-detector.onrender.com",
        github_url="https://github.com/sdube/crop-disease-cnn",
        owner_email="student@demo.com",
    ),
    dict(
        title="Blockchain-Based Voting System for Student Government Elections",
        abstract="Student government elections run on paper ballots or unverifiable online forms, leaving no way to audit results after the fact. This project proposes a permissioned blockchain ledger for recording votes, where each cast ballot is hashed and chained, allowing any student to independently verify the final tally without revealing individual votes. A pilot was run in parallel with a real student election to compare turnout and result consistency against the official paper count.",
        student_name="Thabo Ndlangamandla",
        supervisor="Mr. M. Zwane",
        keywords="blockchain, e-voting, distributed ledger, election integrity, campus technology",
        department="Computer Science",
        institution=EMCU,
        topic_area="Higher Education",
        year=2026,
        status="Ongoing",
        github_url="https://github.com/tndlangamandla/campus-vote-chain",
        owner_email="student@demo.com",
        approval_status="rejected",
        review_note="The abstract doesn't address how ballot secrecy is preserved once votes are hashed on a public-facing chain — please revise the privacy/anonymity section before resubmitting.",
        reviewer_email="admin@demo.com",
    ),
]

# (question, answer) pairs keyed by project title, seeded to demonstrate the Q&A feature
SAMPLE_DISCUSSION = {
    "Job Recruitment System": [
        ("What dataset did you train the classifier on, and how did you handle class imbalance?",
         "We used a public resume/job-description dataset supplemented with anonymised local postings, and applied class weighting rather than oversampling since the qualified/not-qualified split was only mildly imbalanced."),
    ],
    "Crop Disease Detection from Leaf Images Using Convolutional Neural Networks": [
        ("How does accuracy hold up on leaf images taken in poor lighting, since that's common in the field?",
         "Accuracy drops noticeably under low light — around a 12-point drop in our test set. We're experimenting with a lighting-normalisation preprocessing step to close that gap."),
    ],
}


def build_corpus_for_seed(rows):
    if not rows:
        return [], None, None
    documents = [f"{r['title']} {r['abstract']} {r['keywords']}" for r in rows]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return rows, None, None
    return rows, matrix, vectorizer


def seed():
    init_db()
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    user_ids = {}
    for u in USERS:
        existing = db.execute("SELECT id FROM users WHERE email = ?", (u["email"],)).fetchone()
        if existing:
            user_ids[u["email"]] = existing["id"]
            continue
        cur = db.execute(
            "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (u["name"], u["email"], generate_password_hash(DEMO_PASSWORD), u["role"], datetime.now().isoformat()),
        )
        user_ids[u["email"]] = cur.lastrowid
    db.commit()

    inserted_rows = []
    for p in PROJECTS:
        exists = db.execute("SELECT id FROM projects WHERE title = ?", (p["title"],)).fetchone()
        if exists:
            continue
        summary = auto_summary(p["abstract"])
        owner_id = user_ids.get(p.get("owner_email"))
        approval_status = p.get("approval_status", "approved")
        reviewed_by = user_ids.get(p.get("reviewer_email", "admin@demo.com")) if approval_status != "pending" else None
        reviewed_at = datetime.now().isoformat() if approval_status != "pending" else None
        cur = db.execute(
            """INSERT INTO projects
               (title, abstract, student_name, supervisor, keywords, topic_area, department,
                institution, year, file_path, summary, builds_on_id, owner_id, live_demo_url,
                github_url, status, approval_status, review_note, reviewed_by, reviewed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (p["title"], p["abstract"], p["student_name"], p["supervisor"], p["keywords"],
             p["topic_area"], p["department"], p["institution"], p["year"], summary, owner_id,
             p.get("live_demo_url"), p.get("github_url"), p.get("status", "Ongoing"),
             approval_status, p.get("review_note"), reviewed_by, reviewed_at,
             datetime.now().isoformat()),
        )
        inserted_rows.append((cur.lastrowid, p["title"]))
    db.commit()

    # Overlap flags across the freshly seeded corpus
    rows = db.execute("SELECT * FROM projects").fetchall()
    rows, matrix, vectorizer = build_corpus_for_seed(rows)
    if matrix is not None:
        for i, row in enumerate(rows):
            query_vec = matrix[i]
            scores = cosine_similarity(query_vec, matrix).flatten()
            for j, row2 in enumerate(rows):
                if i >= j:
                    continue
                if scores[j] >= SIMILARITY_FLAG_THRESHOLD:
                    already = db.execute(
                        "SELECT id FROM overlap_flags WHERE project_id = ? AND similar_project_id = ?",
                        (row["id"], row2["id"]),
                    ).fetchone()
                    if not already:
                        db.execute(
                            "INSERT INTO overlap_flags (project_id, similar_project_id, similarity_score) VALUES (?, ?, ?)",
                            (row["id"], row2["id"], float(scores[j])),
                        )
    db.commit()

    # Sample discussion threads
    title_to_id = {r["title"]: r["id"] for r in rows}
    for title, thread in SAMPLE_DISCUSSION.items():
        project_id = title_to_id.get(title)
        if not project_id:
            continue
        for question, answer in thread:
            existing_q = db.execute(
                "SELECT id FROM comments WHERE project_id = ? AND body = ?", (project_id, question)
            ).fetchone()
            if existing_q:
                continue
            asker_id = user_ids.get("student@demo.com")
            answerer_id = user_ids.get("admin@demo.com")
            q_cur = db.execute(
                "INSERT INTO comments (project_id, user_id, parent_id, body, created_at) VALUES (?, ?, NULL, ?, ?)",
                (project_id, asker_id, question, datetime.now().isoformat()),
            )
            db.execute(
                "INSERT INTO comments (project_id, user_id, parent_id, body, created_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, answerer_id, q_cur.lastrowid, answer, datetime.now().isoformat()),
            )
    db.commit()
    db.close()

    print(f"Seeded {len(inserted_rows)} new project(s) and {len(USERS)} demo account(s).")
    print(f"Demo login password for all seeded accounts: {DEMO_PASSWORD}")
    for u in USERS:
        print(f"  {u['role']:<11} -> {u['email']}")


if __name__ == "__main__":
    seed()
