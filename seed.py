"""Seeds the archive with:
  - 6 real final-year theses transcribed from library photos (Pharmacy, Computer
    Science x2, Medical Laboratory Sciences, Nursing Science, Psychology)
  - 16 MCS306 sprint-prototype projects (all Computer Science) from the original
    "Addressing National Needs Through Technology" brief, used to demo lineage,
    similarity flagging, and topic-gap dashboards with realistic clusters
"""

import os
import sqlite3
from datetime import datetime

from app import init_db, DB_PATH, auto_summary, SIMILARITY_FLAG_THRESHOLD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMCU = "Eswatini Medical Christian University"
SCU = "Swaziland Christian University"

REAL_THESES = [
    dict(
        title="Assessment of Quality and Evaluation of the Pharmaceutical Equivalence of Selected Brands of Ciprofloxacin Tablets Marketed in Eswatini",
        abstract="Ciprofloxacin is a quinolone derivative antibiotic with a broad spectrum of activity used to treat urinary tract, respiratory tract, skin, soft tissue, and intra-abdominal infections. The increase in generic drug products from multiple sources has raised the influx of counterfeit and substandard products, creating a challenge for prescribers substituting one brand for another. This study assessed and evaluated the quality and purity of three different ciprofloxacin tablet brands marketed in Eswatini, purchased from wholesalers, subjecting them to standard physicochemical tests of uniformity of weight, hardness, disintegration, dissolution, and friability to determine their pharmaceutical equivalence and interchangeability.",
        student_name="Harmony-Sarah N'senda",
        supervisor="Department of Pharmacy Faculty",
        keywords="ciprofloxacin, pharmaceutical equivalence, quality control, quinolone antibiotics, dissolution testing",
        department="Pharmacy",
        institution=EMCU,
        topic_area="Health",
        year=2018,
    ),
    dict(
        title="Job Recruitment System",
        abstract="Machine learning plays a significant role in modern human resource procurement. This project uses supervised learning, a type of machine learning that trains a model on labelled data, to classify resumes and job descriptions as 'qualified' or 'not qualified' for a given role. Natural language processing extracts information from resume and job description text using artificial neural networks. The system aims to improve efficiency and effectiveness in the hiring process, reduce cost, and help organisations identify and filter the most suitable candidates for vacant positions based on their skills and attributes.",
        student_name="Olwethu Sakhele Dlamini",
        supervisor="Mr. M. Zwane",
        keywords="machine learning, natural language processing, recruitment, hr technology, resume classification",
        department="Computer Science",
        institution=EMCU,
        topic_area="Higher Education",
        year=2024,
    ),
    dict(
        title="Secure Biometric Authentication on ATM's",
        abstract="Traditional password-based authentication methods for ATMs are prone to security breaches including hacking, phishing, and unauthorized access. This project presents a biometric authentication system that uses face recognition and fingerprint scanning to authenticate users. The system scans the face of the user and compares it with a stored image; if it matches, the user proceeds to fingerprint verification against a database representing citizens of Eswatini. Only when both biometric checks succeed is the user authenticated, providing a more secure alternative to PIN-based ATM access that mitigates brute force and social engineering attacks.",
        student_name="James Baker",
        supervisor="Mr. Makhubu",
        keywords="biometric authentication, facial recognition, fingerprint scanning, atm security, banking technology",
        department="Computer Science",
        institution=EMCU,
        topic_area="Finance",
        year=2024,
    ),
    dict(
        title="Understanding Vulvovaginal Candidiasis Co-infections with Sexually Transmitted Infections in Sub-Saharan Africa",
        abstract="Vulvovaginal candidiasis (VVC) is an exceedingly common mucosal infection of the lower female reproductive tract, with an estimated 75% of women experiencing at least one episode in their lifetime. Using databases such as PLOS, PUBMED, and open access articles published between 2008 and 2023, this review investigated the incidence of VVC in association with co-infections and genital tract infections across Sub-Saharan Africa. A pooled prevalence of VVC co-infection with STIs ranging from 9.6% in South Africa to 34.1% in Namibia was found, with Chlamydia trachomatis, Trichomonas vaginalis, and Neisseria gonorrhoeae among the most commonly observed co-infecting organisms, highlighting a need for more robust regional research on susceptibility profiles.",
        student_name="Latembe Eliza-Rose Mapule Malaika Tembe",
        supervisor="Etando Ayuk (MSc)",
        keywords="vulvovaginal candidiasis, sexually transmitted infections, sub-saharan africa, co-infections, genital tract infection",
        department="Medical Laboratory Sciences",
        institution=EMCU,
        topic_area="Health",
        year=2023,
    ),
    dict(
        title="Practice and Knowledge as Factors Influencing Nursing Documentation Practice Among Registered Nurses in Eswatini Public Hospitals",
        abstract="Nursing documentation is the record of nursing care that is planned and delivered to individual patients by qualified nurses, and is a vital component of safe, ethical, and effective nursing practice. Poor or incomplete documentation undermines patient care and leaves nurses vulnerable to professional misconduct proceedings. This quantitative study measured the level of knowledge and practice around nursing care documentation among 96 registered nurses at Mbabane Government, Raleigh Fitkin Memorial, Hlatikhulu Government, and Good Shepherd Mission hospitals using self-administered questionnaires. Results showed good knowledge and practice on nursing care documentation overall (84% and 83.1% respectively), though gaps remain especially among non-specialized registered nurses.",
        student_name="Dlamini Thando M & Lukhele Nolwazi Z",
        supervisor="Prof. Heeyoung So",
        keywords="nursing documentation, patient care records, nurse knowledge, clinical practice, eswatini public hospitals",
        department="Nursing Science",
        institution=EMCU,
        topic_area="Health",
        year=2020,
    ),
    dict(
        title="Understanding the Lived Experiences of Pupils with Disability at St. Joseph's High School: School Implications for Policy",
        abstract="There is little empirical and policy attention on disability in Swaziland, which makes disability issues difficult to deal with and leaves people with disabilities marginalized and their perspectives overlooked. This phenomenological, qualitative study explored the disability experience from the standpoint of learners with actual disabilities at St. Joseph's High School through interviews with seven learners, analysed using interpretive phenomenological analysis. Findings revealed that learners with disabilities feel lonely and uncared for because they cannot participate in sports with peers, that school infrastructure does not accommodate some disabilities, and that teachers sometimes appear not to understand their conditions. The study recommends government support for inclusive education and community education to reduce stigmatization.",
        student_name="Nicole Johnston",
        supervisor="Mr. Baffour Boahen-Boahen",
        keywords="disability, lived experience, inclusive education, phenomenological study, school policy",
        department="Psychology",
        institution=SCU,
        topic_area="Education",
        year=2022,
    ),
]

MCS306_PROJECTS = [
    dict(
        title="Rural Clinic Queue Tracker",
        abstract="A digital check-in and queue display system for rural clinics that reduces undocumented waiting and speeds up referrals to regional hospitals. Patients check in via a tablet kiosk, staff see a live queue on a shared screen, and cases needing escalation are flagged with notes for the receiving hospital.",
        student_name="Nomcebo Dlamini",
        supervisor="Dr. T. Simelane",
        keywords="healthcare, queue management, flask, referral system",
        topic_area="Health",
        year=2025,
        builds_on_title=None,
    ),
    dict(
        title="Hospital Wait Time Optimizer",
        abstract="An extension of clinic check-in systems that predicts expected wait time per patient using historical queue data, and reorders low-urgency cases to smooth peak-hour congestion at understaffed rural facilities.",
        student_name="Sipho Mabuza",
        supervisor="Dr. T. Simelane",
        keywords="healthcare, wait time prediction, queue optimization, scheduling",
        topic_area="Health",
        year=2026,
        builds_on_title="Rural Clinic Queue Tracker",
    ),
    dict(
        title="Maternal Health Reminder Platform",
        abstract="A reminder system for antenatal visits and child immunisations, generating a personalised schedule from a mother's due date or child's birth date and notifying her by SMS ahead of each appointment.",
        student_name="Nokuthula Shongwe",
        supervisor="Dr. T. Simelane",
        keywords="maternal health, sms reminders, immunisation, healthcare",
        topic_area="Health",
        year=2025,
        builds_on_title=None,
    ),
    dict(
        title="Pharmacy Medicine Availability Checker",
        abstract="A crowd-updated directory where pharmacies mark medicine stock levels and patients search nearby pharmacies before travelling, cutting down repeat trips looking for prescribed medication.",
        student_name="Andile Nkambule",
        supervisor="Mrs. P. Khumalo",
        keywords="pharmacy, healthcare, inventory, geolocation",
        topic_area="Health",
        year=2026,
        builds_on_title=None,
    ),
    dict(
        title="Water Point Status Reporter",
        abstract="A crowd-reporting map application where community members tap a pin to mark a borehole or tap as working or broken, feeding a status dashboard that helps oversight bodies prioritise repairs.",
        student_name="Lindiwe Zwane",
        supervisor="Mr. B. Nxumalo",
        keywords="water infrastructure, crowd reporting, maps, oversight dashboard",
        topic_area="Water & Sanitation",
        year=2025,
        builds_on_title=None,
    ),
    dict(
        title="EEC Outage Reporter",
        abstract="A crowd-reporting map for power outages where residents log location and time of an outage, generating a heatmap that distinguishes isolated faults from wider grid failures.",
        student_name="Mzwandile Fakudze",
        supervisor="Mr. B. Nxumalo",
        keywords="power outage, crowd reporting, heatmap, energy",
        topic_area="Energy",
        year=2025,
        builds_on_title=None,
    ),
    dict(
        title="Kombi Route & Fare Finder",
        abstract="A searchable, user-editable directory of kombi routes, ranks, and approximate fares between towns such as Manzini, Mbabane, and Nhlangano, keeping transport information current through community edits.",
        student_name="Thabo Dlamini",
        supervisor="Mrs. P. Khumalo",
        keywords="public transport, routes, fares, community editing",
        topic_area="Transport",
        year=2026,
        builds_on_title=None,
    ),
    dict(
        title="Digital Stokvel Tracker",
        abstract="A ledger application for informal rotating savings groups that records members, contribution amounts, payout rotation order, and running balance, replacing paper and WhatsApp-based tracking.",
        student_name="Zanele Motsa",
        supervisor="Dr. T. Simelane",
        keywords="fintech, savings groups, ledger, informal finance",
        topic_area="Finance",
        year=2025,
        builds_on_title=None,
    ),
    dict(
        title="Youth Skills & Gig Board",
        abstract="A listing platform where out-of-school youth post skill profiles in trades like tiling, hairdressing, and IT repair, and small businesses post short-term gigs, matching local labour supply to demand.",
        student_name="Mandla Simelane",
        supervisor="Mr. B. Nxumalo",
        keywords="gig economy, youth employment, skills matching",
        topic_area="Youth Employment",
        year=2026,
        builds_on_title=None,
    ),
    dict(
        title="School Dropout Early-Warning Dashboard",
        abstract="An attendance logging tool where three or more consecutive absences automatically flags a learner for teacher follow-up, replacing paper registers with a system that surfaces at-risk students early.",
        student_name="Ayanda Vilakati",
        supervisor="Mrs. P. Khumalo",
        keywords="education, attendance tracking, early warning, dropout prevention",
        topic_area="Education",
        year=2025,
        builds_on_title=None,
    ),
    dict(
        title="SiSwati Literacy Tutor",
        abstract="A gamified quiz application covering common SiSwati lesson sets with word-matching and pronunciation-style exercises, aimed at strengthening early-grade literacy support in a local language.",
        student_name="Precious Dube",
        supervisor="Mrs. P. Khumalo",
        keywords="education, literacy, siswati, gamification",
        topic_area="Education",
        year=2026,
        builds_on_title=None,
    ),
    dict(
        title="Exam Revision Hub",
        abstract="An upload-and-browse platform for past examination papers organised by subject, with a discussion thread on each paper so Matric and IGCSE students can share peer study notes.",
        student_name="Sibusiso Maseko",
        supervisor="Dr. T. Simelane",
        keywords="education, past papers, peer learning, revision",
        topic_area="Education",
        year=2025,
        builds_on_title=None,
    ),
    dict(
        title="Student Final-Year Project Repository",
        abstract="A searchable archive for final-year project titles, abstracts, supervisors, keywords, and downloadable reports, using semantic search and similarity detection to surface conceptually related past work, organised by academic department.",
        student_name="Buccho Nkosi",
        supervisor="Dr. T. Simelane",
        keywords="repository, semantic search, tf-idf, similarity detection, knowledge management",
        topic_area="Higher Education",
        year=2026,
        builds_on_title=None,
    ),
    dict(
        title="Lost & Found Portal for Universities",
        abstract="A campus service where students report lost or found items with photos, descriptions, and pickup information, replacing informal WhatsApp group postings with a searchable central listing.",
        student_name="Nomfundo Simelane",
        supervisor="Mr. B. Nxumalo",
        keywords="campus services, lost and found, higher education",
        topic_area="Higher Education",
        year=2025,
        builds_on_title=None,
    ),
    dict(
        title="Campus Facility Booking System",
        abstract="A booking system where staff view lecture room and laboratory availability, submit booking requests, and receive approval notifications, reducing double-booking of shared campus spaces.",
        student_name="Sanele Ndlangamandla",
        supervisor="Mrs. P. Khumalo",
        keywords="facility booking, resource management, higher education",
        topic_area="Higher Education",
        year=2026,
        builds_on_title=None,
    ),
    dict(
        title="Eswatini Heritage & Tourism Guide",
        abstract="A mobile-friendly guide documenting cultural sites and events such as Incwala, Umhlanga, and Ezulwini landmarks, with site descriptions, photos, and a basic event calendar for visitors and local youth.",
        student_name="Gugu Ndzimandze",
        supervisor="Dr. T. Simelane",
        keywords="tourism, heritage, culture, event calendar",
        topic_area="Tourism & Culture",
        year=2025,
        builds_on_title=None,
    ),
]


def seed():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    title_to_id = {}

    # Real theses first (PRJ-<year>-0001 through 0006)
    for t in REAL_THESES:
        summary = auto_summary(t["abstract"])
        cur = db.execute(
            """INSERT INTO projects
               (title, abstract, student_name, supervisor, keywords, topic_area, department,
                institution, year, file_path, summary, builds_on_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (t["title"], t["abstract"], t["student_name"], t["supervisor"], t["keywords"],
             t["topic_area"], t["department"], t["institution"], t["year"], None, summary,
             None, datetime.now().isoformat()),
        )
        title_to_id[t["title"]] = cur.lastrowid

    # MCS306 cohort — all Computer Science, all EMCU
    for p in MCS306_PROJECTS:
        summary = auto_summary(p["abstract"])
        builds_on_id = title_to_id.get(p["builds_on_title"]) if p.get("builds_on_title") else None
        cur = db.execute(
            """INSERT INTO projects
               (title, abstract, student_name, supervisor, keywords, topic_area, department,
                institution, year, file_path, summary, builds_on_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (p["title"], p["abstract"], p["student_name"], p["supervisor"], p["keywords"],
             p["topic_area"], "Computer Science", EMCU, p["year"], None, summary,
             builds_on_id, datetime.now().isoformat()),
        )
        title_to_id[p["title"]] = cur.lastrowid

    db.commit()

    # Compute overlap flags across the whole seeded corpus
    rows = db.execute("SELECT * FROM projects").fetchall()
    documents = [f"{r['title']} {r['abstract']} {r['keywords']}" for r in rows]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(documents)
    sim_matrix = cosine_similarity(matrix)

    seen_pairs = set()
    for i, row_i in enumerate(rows):
        for j, row_j in enumerate(rows):
            if i >= j:
                continue
            score = sim_matrix[i][j]
            if score >= SIMILARITY_FLAG_THRESHOLD:
                pair_key = tuple(sorted([row_i["id"], row_j["id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                db.execute(
                    """INSERT INTO overlap_flags (project_id, similar_project_id, similarity_score)
                       VALUES (?, ?, ?)""",
                    (row_i["id"], row_j["id"], round(score * 100, 1)),
                )
    db.commit()
    db.close()
    print(f"Seeded {len(REAL_THESES)} real theses + {len(MCS306_PROJECTS)} MCS306 projects, "
          f"{len(seen_pairs)} overlap flags.")


if __name__ == "__main__":
    seed()
