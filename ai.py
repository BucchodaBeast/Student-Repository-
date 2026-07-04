"""
AI intelligence layer — Groq (Llama 3.3-70b) calls that sit on top of the
static archive. Every function degrades gracefully to None if GROQ_API_KEY
isn't set or the request fails, so the rest of the app never breaks because
of this layer.

Set GROQ_API_KEY as an environment variable (in Render: Environment tab).
"""

import os
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def ai_enabled():
    return bool(GROQ_API_KEY)


def _call_groq(messages, max_tokens=500, temperature=0.4):
    if not GROQ_API_KEY:
        return None
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=25,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def ai_summary(title, abstract):
    """One-sentence plain-language summary. Returns None on failure —
    caller should fall back to the extractive summarizer."""
    return _call_groq(
        [
            {
                "role": "system",
                "content": "You are an academic archivist writing single-sentence, plain-language "
                "summaries of research abstracts for a searchable catalog. Output ONLY the "
                "summary sentence — no preamble, no quotation marks.",
            },
            {
                "role": "user",
                "content": f"Title: {title}\n\nAbstract: {abstract}\n\n"
                "Write one clear sentence (max 28 words) summarizing what this project does and why it matters.",
            },
        ],
        max_tokens=90,
    )


def ai_overlap_analysis(new_title, new_abstract, flagged):
    """flagged: list of (sqlite row, similarity score 0-1). Returns a short
    reasoned judgment on genuine overlap risk vs. shared vocabulary, or None."""
    context = "\n\n".join(
        f'- "{r["title"]}" ({round(s * 100)}% textual similarity): {r["abstract"][:280]}'
        for r, s in flagged
    )
    return _call_groq(
        [
            {
                "role": "system",
                "content": "You are an academic integrity reviewer. Given a new project abstract "
                "and similar existing projects, assess in 2-3 direct sentences whether this looks "
                "like genuine overlap/plagiarism risk or just shared subject-matter vocabulary, and "
                "name what specifically matches or differs (methodology, scope, population, etc).",
            },
            {
                "role": "user",
                "content": f"NEW SUBMISSION\nTitle: {new_title}\nAbstract: {new_abstract}\n\n"
                f"SIMILAR EXISTING PROJECTS:\n{context}",
            },
        ],
        max_tokens=220,
    )


def ai_defense_questions(title, abstract, department):
    """Returns numbered list of 5 panel questions as raw text, or None."""
    return _call_groq(
        [
            {
                "role": "system",
                "content": "You are a strict final-year project defense panelist. Generate exactly "
                "5 tough, specific questions this panel would ask about the project below. Number "
                "them 1-5, one per line. No preamble, no closing remarks.",
            },
            {
                "role": "user",
                "content": f"Department: {department}\nTitle: {title}\nAbstract: {abstract}",
            },
        ],
        max_tokens=400,
    )


def ai_research_assistant(question, context_projects):
    """context_projects: list of (sqlite row, similarity score). Returns an
    answer citing project IDs inline as [N], or None."""
    context = "\n\n".join(
        f'[{r["id"]}] "{r["title"]}" ({r["department"]}, {r["year"]}) by {r["student_name"]}: {r["abstract"]}'
        for r, s in context_projects
    )
    return _call_groq(
        [
            {
                "role": "system",
                "content": "You are a research assistant for a university final-year project "
                "archive. Answer the user's question using ONLY the provided project excerpts. "
                "Cite projects inline using their bracketed ID, e.g. [3]. If the archive genuinely "
                "has nothing relevant, say so plainly instead of inventing anything. Be concise — "
                "4-6 sentences.",
            },
            {
                "role": "user",
                "content": f"ARCHIVE EXCERPTS:\n{context}\n\nQUESTION: {question}",
            },
        ],
        max_tokens=550,
    )


def ai_gap_suggestions(department, existing_titles):
    """Suggests 3 novel, unaddressed project ideas for a department given
    what's already been done, grounded in real-world problems."""
    titles_list = "\n".join(f"- {t}" for t in existing_titles) or "(none on file yet)"
    return _call_groq(
        [
            {
                "role": "system",
                "content": "You are a final-year project coordinator at a university in Eswatini. "
                "Given the projects already completed in a department, propose exactly 3 NEW final-year "
                "project ideas that are not redundant with existing work, are technically scoped for a "
                "one-semester student project, and address a genuine real-world problem relevant to "
                "Eswatini or a similar Sub-Saharan African context. Format as:\n"
                "1. [Title] — [1-sentence problem it solves]\n2. ...\n3. ...\n"
                "No preamble, no closing remarks.",
            },
            {
                "role": "user",
                "content": f"Department: {department}\n\nExisting projects on file:\n{titles_list}",
            },
        ],
        max_tokens=350,
    )
