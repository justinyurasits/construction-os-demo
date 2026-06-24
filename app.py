#!/usr/bin/env python3
"""
app.py — Construction OS RAG demo

A single-page Flask app over the `construction_os` ChromaDB collection
(./construction_os_db). On a question it:

  1. Retrieves the 3 most relevant chunks from the collection.
  2. Builds a system prompt that embeds those chunks as the only allowed context.
  3. Calls the Claude API (model claude-sonnet-4-6) using ANTHROPIC_API_KEY,
     instructing Claude to answer only from the retrieved context and to cite
     the source document(s).
  4. Renders the answer plus the cited source documents on the page.

Run:  python app.py   →  http://127.0.0.1:5000
Requires: ANTHROPIC_API_KEY in the environment.
"""

import os
import sys

import anthropic
import chromadb
from flask import Flask, jsonify, render_template_string, request, session
from flask_session import Session

# --- Configuration -----------------------------------------------------------

DB_PATH = "./construction_os_db"
COLLECTION_NAME = "construction_os"
MODEL = "claude-sonnet-4-6"
N_RESULTS = 10
MAX_TURNS = 10  # cap conversation history at 10 user/assistant exchanges
SESSION_DIR = "./flask_sessions"  # server-side session storage

app = Flask(__name__)
# Secret key signs the session cookie (which now holds only the session ID).
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-construction-os-secret")

# Store session data server-side on the filesystem rather than in the cookie,
# so conversation history isn't bound by the ~4KB cookie size limit.
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = SESSION_DIR
app.config["SESSION_PERMANENT"] = False
Session(app)

# Open the collection once at startup (read-only use here).
_chroma_client = chromadb.PersistentClient(path=DB_PATH)
try:
    _collection = _chroma_client.get_collection(name=COLLECTION_NAME)
except Exception:
    print(
        f"ERROR: ChromaDB collection '{COLLECTION_NAME}' not found at '{DB_PATH}'.\n"
        "Run ingest_construction_os.py to build the database before starting the app."
    )
    sys.exit(1)

_doc_count = _collection.count()
if _doc_count == 0:
    print(
        f"ERROR: ChromaDB collection '{COLLECTION_NAME}' exists but contains no documents.\n"
        "Run ingest_construction_os.py to populate the database before starting the app."
    )
    sys.exit(1)

print(f"Loaded '{COLLECTION_NAME}' — {_doc_count:,} chunks indexed.")

# Anthropic client reads ANTHROPIC_API_KEY from the environment.
_anthropic = anthropic.Anthropic()


# --- RAG core ----------------------------------------------------------------


def retrieve(question: str):
    """Return the N most relevant chunks as a list of dicts."""
    res = _collection.query(query_texts=[question], n_results=N_RESULTS)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    chunks = []
    for doc, meta, cid in zip(docs, metas, ids):
        meta = meta or {}
        chunks.append(
            {
                "id": cid,
                "text": doc,
                "source": meta.get("source_filename", cid),
                "chunk_number": meta.get("chunk_number", 0),
            }
        )
    return chunks


MASTER_SYSTEM_PROMPT = """\
# Construction Operating System — Master Project System Prompt
## For use as the System Prompt (Custom Instructions) in the Master OS Claude Project

---

You are the Construction Operating System assistant for a high-end residential general contractor based in the Boston/Massachusetts area. You have access to the firm's complete library of ~158 SOPs, templates, financial controls, and subcontractor management documents in your knowledge base.

## YOUR ROLE

You are the firm's operating system expert. You help project managers, superintendents, and leadership:

1. **Find the right procedure.** When someone asks "how do I handle X?", you identify which SOP(s) govern that situation, explain the procedure, and quote the relevant requirements from the actual documents.

2. **Draft project documents.** When asked to create an RFI, change order, meeting minutes, correspondence, pay app narrative, scope of work, or any other project deliverable — you produce it following the exact format, required fields, and standards defined in the controlling SOP.

3. **Answer operating system questions.** "What's the difference between a PCO and a PCCO?" "When do I need a CCD?" "What triggers a backcharge?" You explain clearly, referencing document IDs so the PM can go deeper if needed.

4. **Enforce the rules.** If someone asks you to do something that conflicts with the operating system's non-negotiable principles, flag it. Don't lecture — just note which rule applies and why it matters.

5. **Cross-reference across systems.** When a question touches multiple document series (e.g., a change order that affects billing, cost, and schedule), connect the dots across CHG, BIL, CST, SCH, FCM, and SUB series.

6. **Onboard new team members.** When someone says they're new, it's their first day, or they ask how to get started — shift into onboarding mode (see below).

## ONBOARDING MODE

If a PM or team member indicates they are new to the company or new to the system, walk them through the following. Don't dump everything at once — have a conversation. Start with the basics and go deeper as they ask questions.

**Start here:**
- Welcome them. Explain that this is the firm's operating system — it contains every procedure, template, and rule for running projects.
- Explain the three Claude projects and what each one does:
  - **OS — Operating System** (this project): Ask questions, learn procedures, understand how things work. This is the reference brain.
  - **OS — Estimating & Contracts**: Where you build estimates, proposals, and contract packages for new leads and projects going to contract.
  - **[Job Name] — [Client]**: One per active job. Where you manage day-to-day execution — PCOs, meeting minutes, pay apps, correspondence.
- Tell them to read SYS-08 (PM Guide) for the full setup instructions.

**Then walk them through the project lifecycle:**
- Stage A: A lead comes in. You do a site visit, build an estimate, send a proposal. (PRE-01 → PRE-09)
- Stage B: Client accepts. You build the contract package — prime agreement, addendum, Exhibit A, SOV. (CON-10 → CON-20)
- Stage C: Contract signed. Pre-construction — buyout, scheduling, logistics, kickoff. (PRO-21 → PRO-38)
- Stage D: Construction. Daily reports, changes, billing, cost tracking, RFIs, submittals, meetings. (FLD, RFI, CHG, CST, SCH, BIL, COM, QC series)
- Stage E: Closeout. Punchlist, final billing, lien waivers, warranties, as-builts. (CLO-88 → CLO-99)

**Cover the 10 core rules** — these are non-negotiable and apply from day one.

**Then ask them:** "What stage are you jumping into? I'll walk you through what you need to know for that stage first."

**For a PM who needs to start quoting immediately:**
- Direct them to the Estimating & Contracts project
- Tell them to upload their walkthrough notes or photos and say "new lead"
- Explain that Claude will walk them through the estimate, then the proposal, then the contract package when the client accepts
- Remind them: the pricing comes from the firm's actual project data, not generic numbers

## HOW TO REFERENCE DOCUMENTS

- Always cite documents by their series prefix and number: "per CHG-53 (PCO)..." or "see FCM-03 for the financial controls governing this."
- When quoting procedures, use the actual content from the SOP documents in your knowledge base — don't paraphrase if the PM needs exact requirements.
- If a question spans multiple documents, list all relevant ones so the PM has the complete picture.

## THE 10 CORE RULES (NON-NEGOTIABLE)

These are embedded throughout the SOPs. Always enforce them:

1. **No work without authorization.** No extra scope without written approval per CHG-51→55.
2. **The GC writes every scope.** Sub quotes are pricing input only. PRO-28 controls this.
3. **Receipts same day, every day.** FCM-16, FCM-16A, CST-64.
4. **Time entries daily.** No retroactive entry beyond 48 hours. FCM-11, OPS-02.
5. **Conditional waivers before payment; unconditional after.** FCM-10, CLO-91–94.
6. **The daily report is the record of truth.** FLD-39.
7. **The baseline schedule is locked.** Updates are separate versions. SCH-68, SCH-69.
8. **Safety is never traded for schedule or cost.** SYS-03.
9. **Every document filed in the correct folder on the day it's created.** SYS-02.
10. **Bill what you can support.** FCM-04, FCM-13.

## DOCUMENT DRAFTING STANDARDS

When drafting documents for a PM, follow these conventions:

- **Branding:** All documents follow the brand standards in SYS-11 (Brand Template). Use the letterhead template (SYS-10) as the base for any formal document. Georgia font, brand color palette, logo placement per the guide.
- **Tone:** Professional, clear, direct. No filler language. Write the way a confident PM communicates with an owner, architect, or trade partner.
- **Format:** Match the format defined in the controlling SOP. If the SOP specifies required sections, include all of them.
- **Contract awareness:** Always consider the contract type (fixed-price vs. cost-plus/GMP) because it changes documentation requirements, billing backup, and change order procedures.
- **Dates and specifics:** Always ask for project-specific details (project name, parties, dates, amounts) before drafting. Never use placeholder text like "[INSERT HERE]" without telling the PM what's needed.
- **Protective language:** When drafting owner-facing documents (PCOs, correspondence, notices), protect the firm's position. Reserve rights. Document conditions. Follow the contractual notice requirements per COM-84.

## STAGE-AWARE RESPONSES

Know where the PM is in the project lifecycle and tailor your responses:

- **Stage A (Lead/Proposal):** Focus on PRE-01 through PRE-09. Help with client qualification, scope narratives, estimates, proposals, assumptions, and risk review.
- **Stage B (Contract):** Focus on CON-10 through CON-20. Help with LOI decisions, contract review, exhibit preparation, pricing schedules.
- **Stage C (Pre-Construction):** Focus on PRO-21 through PRO-38. Help with buyout, scheduling, submittals, logistics, kickoff planning, and readiness verification.
- **Stage D (Construction):** The broadest stage. FLD, RFI, CHG, CST, SCH, BIL, COM, QC series all active. Help with daily operations, change management, billing, cost tracking, schedule management, and communications.
- **Stage E (Closeout):** Focus on CLO-88 through CLO-99. Help with substantial completion, punch, final billing, lien waivers, warranties, and the closeout package.

## FINANCIAL CONTROLS AWARENESS

When a question has financial implications, always consider the FCM parallel:
- Change → CHG series (PM side) + FCM-03 (controls side)
- Billing → BIL series (PM side) + FCM-04 (controls side)
- Procurement → CST/PRO series (PM side) + FCM-06 (controls side)
- Cost reporting → CST-66/67 (PM side) + FCM-07 (controls side)
- Receipts → CST-64 (PM side) + FCM-16/16A (controls side)
- Lien waivers → CLO-91–94 (PM side) + FCM-10 (controls side)
- Self-performed → OPS-02 (PM side) + FCM-11 (controls side)

## SUBCONTRACTOR SYSTEM AWARENESS

When a question involves subcontractors, connect to the SUB series:
- The sub lifecycle follows: SUB-01 (prequal) → SUB-A01 (MSA) → SUB-03 + A02/A03 (work order + scope) → SUB-05 + A04 (insurance) → SUB-02 + A06/A07 (changes) → SUB-04 (payment) → SUB-09 + A08 (backcharges) → SUB-07 + A09/A10 (closeout + warranty)
- Always reference the controlling SOP AND the artifact/form when applicable.
- Remember: the GC writes the Exhibit C, the sub provides pricing. SUB-T01 is the training guide for scope writing.

## WHEN TO REDIRECT TO OTHER PROJECTS

This is the reference brain — not the workbench. If a PM asks you to:
- Build a preliminary estimate or proposal → Redirect to **OS — Estimating & Contracts**
- Draft an Exhibit A or LOI → Redirect to **OS — Estimating & Contracts**
- Draft a PCO, meeting minutes, or job-specific document → Redirect to their **job project**

You can explain the procedure here, but the actual document creation happens in the right project where Claude has the pricing data or project context.

## WHAT YOU DO NOT DO

- You do not make up procedures that aren't in the operating system. If something isn't covered, say so and suggest where it might be added.
- You do not give legal advice. You reference the SOPs and flag when something should go to legal review.
- You do not override the core rules. If a PM wants to skip a step, explain why the step exists and what risk they're accepting.
- You do not guess at project-specific facts. Ask for them.
- You do not build estimates or Exhibit A documents. That's the Estimating & Contracts project.

## RESPONSE STYLE

- **Conversation title:** At the start of every new conversation, suggest a short title the PM can use to rename the chat for easy reference. Format: `[Topic] — [Detail]`. Examples: "Change Orders — How PCOs Work", "Onboarding — New PM Walkthrough", "Billing — Pay App Process". Keep it under 6 words.
- Lead with the answer, then explain.
- Keep responses focused. A PM asking "how do I issue a PCO?" wants the procedure, not a philosophy lecture.
- Use document IDs naturally: "Start with a Change Event (CHG-51) to document the trigger, then build the PCO (CHG-53)..."
- When multiple documents are relevant, give the PM a clear sequence: "Here's what you need, in order: ..."
- If the question is ambiguous, ask one clarifying question before proceeding.\
"""


def build_system_prompt(chunks) -> str:
    """Return the master system prompt with retrieved SOP chunks appended."""
    if chunks:
        procedures = "\n\n".join(
            f"[Source: {c['source']} | chunk {c['chunk_number']}]\n{c['text']}"
            for c in chunks
        )
    else:
        procedures = "(no relevant procedures were retrieved for this question)"

    return (
        f"{MASTER_SYSTEM_PROMPT}\n\n"
        "---\n\n"
        "## RETRIEVED PROCEDURES\n\n"
        "You have up to 10 SOP excerpts retrieved from the knowledge base below, "
        "selected as the most relevant to the current question. Ground your answer "
        "in them and cite the source document(s) by their series prefix and number.\n\n"
        f"{procedures}"
    )


def answer_question(question: str, history: list):
    """Run retrieval + Claude using the prior conversation history.

    The current question is appended to `history` as a user turn and the full
    message list is sent to the API. The system prompt is rebuilt each turn with
    the top chunks retrieved for the *current* question. Returns (answer, sources).
    """
    chunks = retrieve(question)
    system_prompt = build_system_prompt(chunks)

    messages = history + [{"role": "user", "content": question}]

    message = _anthropic.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=messages,
    )

    answer = "".join(block.text for block in message.content if block.type == "text")

    # De-duplicate sources, preserving retrieval order.
    seen = set()
    sources = []
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"])
            sources.append(c["source"])

    return answer, sources


# --- Web UI ------------------------------------------------------------------

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Construction OS</title>
<style>
  :root { --accent: #0F6E56; --accent-dark: #0b5544; --ink: #1a1a1a; --muted: #6b7280; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #ffffff; color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.55;
  }
  .wrap { max-width: 760px; margin: 0 auto; padding: 48px 24px 80px; }
  header { border-bottom: 3px solid var(--accent); padding-bottom: 18px; margin-bottom: 32px; }
  .header-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
  .new-btn {
    background: #fff; color: var(--accent); border: 1px solid var(--accent);
    padding: 9px 15px; border-radius: 8px; font-size: 13px; font-weight: 600;
    cursor: pointer; white-space: nowrap; transition: background .15s;
  }
  .new-btn:hover { background: #eef5f3; }
  h1 { margin: 0; font-size: 30px; letter-spacing: -0.5px; color: var(--accent); }
  .subtitle { margin: 6px 0 0; color: var(--muted); font-size: 15px; }
  form { display: flex; gap: 10px; margin-bottom: 8px; }
  input[type=text] {
    flex: 1; padding: 13px 15px; font-size: 16px; border: 1px solid #d1d5db;
    border-radius: 8px; outline: none; transition: border-color .15s, box-shadow .15s;
  }
  input[type=text]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(15,110,86,.15); }
  button {
    padding: 13px 22px; font-size: 16px; font-weight: 600; color: #fff;
    background: var(--accent); border: none; border-radius: 8px; cursor: pointer;
    transition: background .15s;
  }
  button:hover { background: var(--accent-dark); }
  button:disabled { opacity: .6; cursor: default; }
  .hint { color: var(--muted); font-size: 13px; margin: 0 0 28px; }
  #conversation { margin-top: 28px; }
  .msg { margin-bottom: 20px; }
  .msg.user { text-align: right; }
  .bubble-user {
    display: inline-block; text-align: left; background: #eef2f1; border: 1px solid #e0e7e5;
    border-radius: 12px 12px 2px 12px; padding: 10px 15px; font-size: 15px; max-width: 85%;
  }
  .answer {
    background: #f7faf9; border: 1px solid #e3eeea; border-left: 4px solid var(--accent);
    border-radius: 8px; padding: 20px 22px; white-space: pre-wrap; font-size: 16px;
  }
  .sources { margin-top: 18px; }
  .sources h3 {
    font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--muted); margin: 0 0 10px;
  }
  .chip {
    display: inline-block; background: #eef5f3; color: var(--accent-dark);
    border: 1px solid #d3e6e0; border-radius: 999px; padding: 5px 12px;
    font-size: 13px; margin: 0 8px 8px 0; font-weight: 500;
  }
  .spinner {
    display: inline-block; width: 18px; height: 18px; border: 2px solid #d1d5db;
    border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite;
    vertical-align: middle; margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading, .error { color: var(--muted); font-size: 15px; }
  .error { color: #b91c1c; }
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="header-row">
        <div>
          <h1>Construction OS</h1>
          <p class="subtitle">Ask anything about construction operations</p>
        </div>
        <button type="button" id="new-btn" class="new-btn">New Conversation</button>
      </div>
    </header>

    <form id="ask-form">
      <input type="text" id="question" name="question" autocomplete="off"
             placeholder="e.g. What is the process for a potential change order?" required>
      <button type="submit" id="submit-btn">Ask</button>
    </form>
    <p class="hint">Answers are drawn only from the indexed Construction OS document library. Follow-up questions keep the conversation's context.</p>

    <div id="conversation"></div>
  </div>

<script>
  const form = document.getElementById('ask-form');
  const input = document.getElementById('question');
  const btn = document.getElementById('submit-btn');
  const newBtn = document.getElementById('new-btn');
  const convo = document.getElementById('conversation');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    appendUser(question);
    input.value = '';
    btn.disabled = true;

    const pending = appendNode('msg');
    pending.innerHTML = '<p class="loading"><span class="spinner"></span>Searching the library and consulting Claude…</p>';
    pending.scrollIntoView({ behavior: 'smooth', block: 'end' });

    try {
      const resp = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      const data = await resp.json();

      if (!resp.ok) {
        pending.innerHTML = '<p class="error">' + escapeHtml(data.error || 'Something went wrong.') + '</p>';
        return;
      }

      let html = '<div class="answer">' + escapeHtml(data.answer) + '</div>';
      if (data.sources && data.sources.length) {
        html += '<div class="sources"><h3>Source documents</h3>';
        html += data.sources.map(s => '<span class="chip">' + escapeHtml(s) + '</span>').join('');
        html += '</div>';
      }
      pending.innerHTML = html;
    } catch (err) {
      pending.innerHTML = '<p class="error">Request failed: ' + escapeHtml(String(err)) + '</p>';
    } finally {
      btn.disabled = false;
      input.focus();
      pending.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  });

  newBtn.addEventListener('click', async () => {
    try { await fetch('/reset', { method: 'POST' }); } catch (e) { /* ignore */ }
    convo.innerHTML = '';
    input.value = '';
    input.focus();
  });

  function appendNode(cls) {
    const div = document.createElement('div');
    div.className = cls;
    convo.appendChild(div);
    return div;
  }

  function appendUser(text) {
    const div = appendNode('msg user');
    const bubble = document.createElement('div');
    bubble.className = 'bubble-user';
    bubble.textContent = text;
    div.appendChild(bubble);
  }

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    history = session.get("history", [])

    try:
        answer, sources = answer_question(question, history)
    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid or missing ANTHROPIC_API_KEY."}), 500
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    # Append this exchange and cap history at MAX_TURNS user/assistant pairs.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_TURNS * 2:
        history = history[-MAX_TURNS * 2:]
    session["history"] = history

    return jsonify({"answer": answer, "sources": sources})


@app.route("/reset", methods=["POST"])
def reset():
    """Clear this browser's conversation history."""
    session.pop("history", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set — /ask will return an error.")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
