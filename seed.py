"""Seeds the archive with realistic sample projects so the prototype demos
well immediately — real similarity clusters, real topic distribution,
real lineage chain, no empty-state screens."""

import os
import sqlite3
from datetime import datetime

from app import init_db, DB_PATH, auto_summary, SIMILARITY_FLAG_THRESHOLD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAMPLE_PROJECTS = [
    dict(
        title="Rural Clinic Queue Tracker",
        abstract="A digital check-in and queue display system for rural clinics that reduces undocumented waiting and speeds up referrals to regional hospitals. Patients check in via a tablet kiosk, staff see a live queue on a shared screen, and cases needing escalation are flagged with notes for the receiving hospital.",
        student_name="Nomcebo Dlamini",
        supervisor="Dr. T. Simelane",
        keywords="healthcare, queue management, flask, referral system",
        topic_area="Health",
        year=2025,
        builds_on_id=None,
    ),
    dict(
        title="Hospital Wait Time Optimizer",
        abstract="An extension of clinic check-in systems that predicts expected wait time per patient using historical queue data, and reorders low-urgency cases to smooth peak-hour congestion at understaffed rural facilities.",
        student_name="Sipho Mabuza",
        supervisor="Dr. T. Simelane",
        keywords="healthcare, wait time prediction, queue optimization, scheduling",
        topic_area="Health",
        year=2026,
        builds_on_id=1,
    ),
    dict(
        title="Maternal Health Reminder Platform",
        abstract="A reminder system for antenatal visits and child immunisations, generating a personalised schedule from a mother's due date or child's birth date and notifying her by SMS ahead of each appointment.",
        student_name="Nokuthula Shongwe",
        supervisor="Dr. T. Simelane",
        keywords="maternal health, sms reminders, immunisation, healthcare",
        topic_area="Health",
        year=2025,
        builds_on_id=None,
    ),
    dict(
        title="Pharmacy Medicine Availability Checker",
        abstract="A crowd-updated directory where pharmacies mark medicine stock levels and patients search nearby pharmacies before travelling, cutting down repeat trips looking for prescribed medication.",
        student_name="Andile Nkambule",
        supervisor="Mrs. P. Khumalo",
        keywords="pharmacy, healthcare, inventory, geolocation",
        topic_area="Health",
        year=2026,
        builds_on_id=None,
    ),
    dict(
        title="Water Point Status Reporter",
        abstract="A crowd-reporting map application where community members tap a pin to mark a borehole or tap as working or broken, feeding a status dashboard that helps oversight bodies prioritise repairs.",
        student_name="Lindiwe Zwane",
        supervisor="Mr. B. Nxumalo",
        keywords="water infrastructure, crowd reporting, maps, oversight dashboard",
        topic_area="Water & Sanitation",
        year=2025,
        builds_on_id=None,
    ),
    dict(
        title="EEC Outage Reporter",
        abstract="A crowd-reporting map for power outages where residents log location and time of an outage, generating a heatmap that distinguishes isolated faults from wider grid failures.",
        student_name="Mzwandile Fakudze",
        supervisor="Mr. B. Nxumalo",
        keywords="power outage, crowd reporting, heatmap, energy",
        topic_area="Energy",
        year=2025,
        builds_on_id=None,
    ),
    dict(
        title="Kombi Route & Fare Finder",
        abstract="A searchable, user-editable directory of kombi routes, ranks, and approximate fares between towns such as Manzini, Mbabane, and Nhlangano, keeping transport information current through community edits.",
        student_name="Thabo Dlamini",
        supervisor="Mrs. P. Khumalo",
        keywords="public transport, routes, fares, community editing",
        topic_area="Transport",
        year=2026,
        builds_on_id=None,
    ),
    dict(
        title="Digital Stokvel Tracker",
        abstract="A ledger application for informal rotating savings groups that records members, contribution amounts, payout rotation order, and running balance, replacing paper and WhatsApp-based tracking.",
        student_name="Zanele Motsa",
        supervisor="Dr. T. Simelane",
        keywords="fintech, savings groups, ledger, informal finance",
        topic_area="Finance",
        year=2025,
        builds_on_id=None,
    ),
    dict(
        title="Youth Skills & Gig Board",
        abstract="A listing platform where out-of-school youth post skill profiles in trades like tiling, hairdressing, and IT repair, and small businesses post short-term gigs, matching local labour supply to demand.",
        student_name="Mandla Simelane",
        supervisor="Mr. B. Nxumalo",
        keywords="gig economy, youth employment, skills matching",
        topic_area="Youth Employment",
        year=2026,
        builds_on_id=None,
    ),
    dict(
        title="School Dropout Early-Warning Dashboard",
        abstract="An attendance logging tool where three or more consecutive absences automatically flags a learner for teacher follow-up, replacing paper registers with a system that surfaces at-risk students early.",
        student_name="Ayanda Vilakati",
        supervisor="Mrs. P. Khumalo",
        keywords="education, attendance tracking, early warning, dropout prevention",
        topic_area="Education",
        year=2025,
        builds_on_id=None,
    ),
    dict(
        title="SiSwati Literacy Tutor",
        abstract="A gamified quiz application covering common SiSwati lesson sets with word-matching and pronunciation-style exercises, aimed at strengthening early-grade literacy support in a local language.",
        student_name="Precious Dube",
        supervisor="Mrs. P. Khumalo",
        keywords="education, literacy, siswati, gamification",
        topic_area="Education",
        year=2026,
        builds_on_id=None,
    ),
    dict(
        title="Exam Revision Hub",
        abstract="An upload-and-browse platform for past examination papers organised by subject, with a discussion thread on each paper so Matric and IGCSE students can share peer study notes.",
        student_name="Sibusiso Maseko",
        supervisor="Dr. T. Simelane",
        keywords="education, past papers, peer learning, revision",
        topic_area="Education",
        year=2025,
        builds_on_id=None,
    ),
    dict(
        title="Student Final-Year Project Repository",
        abstract="A searchable archive for final-year project titles, abstracts, supervisors, keywords, and downloadable reports, using semantic search and similarity detection to surface conceptually related past work.",
        student_name="Buccho Nkosi",
        supervisor="Dr. T. Simelane",
        keywords="repository, semantic search, tf-idf, similarity detection, knowledge management",
        topic_area="Higher Education",
        year=2026,
        builds_on_id=None,
    ),
    dict(
        title="Lost & Found Portal for Universities",
        abstract="A campus service where students report lost or found items with photos, descriptions, and pickup information, replacing informal WhatsApp group postings with a searchable central listing.",
        student_name="Nomfundo Simelane",
        supervisor="Mr. B. Nxumalo",
        keywords="campus services, lost and found, higher education",
        topic_area="Higher Education",
        year=2025,
        builds_on_id=None,
    ),
    dict(
        title="Campus Facility Booking System",
        abstract="A booking system where staff view lecture room and laboratory availability, submit booking requests, and receive approval notifications, reducing double-booking of shared campus spaces.",
        student_name="Sanele Ndlangamandla",
        supervisor="Mrs. P. Khumalo",
        keywords="facility booking, resource management, higher education",
        topic_area="Higher Education",
        year=2026,
        builds_on_id=None,
    ),
    dict(
        title="Eswatini Heritage & Tourism Guide",
        abstract="A mobile-friendly guide documenting cultural sites and events such as Incwala, Umhlanga, and Ezulwini landmarks, with site descriptions, photos, and a basic event calendar for visitors and local youth.",
        student_name="Gugu Ndzimandze",
        supervisor="Dr. T. Simelane",
        keywords="tourism, heritage, culture, event calendar",
        topic_area="Tourism & Culture",
        year=2025,
        builds_on_id=None,
    ),
]


def seed():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    for p in SAMPLE_PROJECTS:
        summary = auto_summary(p["abstract"])
        db.execute(
            """INSERT INTO projects
               (title, abstract, student_name, supervisor, keywords, topic_area, year,
                file_path, summary, builds_on_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (p["title"], p["abstract"], p["student_name"], p["supervisor"], p["keywords"],
             p["topic_area"], p["year"], None, summary, p["builds_on_id"], datetime.now().isoformat()),
        )
    db.commit()

    # Compute overlap flags across the seeded corpus so the dashboard has real data
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
    print(f"Seeded {len(SAMPLE_PROJECTS)} projects and {len(seen_pairs)} overlap flags.")


if __name__ == "__main__":
    seed()
