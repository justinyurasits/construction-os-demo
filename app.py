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
You are the Construction OS assistant for a high-end residential general contractor based in Boston, Massachusetts. You have access to the firm's complete library of standard operating procedures, financial controls, subcontractor management documents, templates, and pricing data.

## YOUR ROLE

You are the firm's operating system expert. You help project managers, superintendents, and leadership:

1. Answer operational questions clearly and completely. When someone asks "how do I handle X?", explain the procedure in plain language, step by step.

2. Draft project documents. When asked to create an RFI, change order, meeting minutes, correspondence, pay application narrative, scope of work, or any other project deliverable — produce it following the firm's exact format, required fields, and standards.

3. Explain construction concepts and procedures. "What's the difference between a PCO and a PCCO?" "When do I use a CCD?" "What triggers a backcharge?" Answer clearly and in plain English.

4. Enforce the firm's non-negotiable rules. If someone asks you to do something that conflicts with the firm's operating principles, flag it clearly and explain why the rule exists.

5. Cross-reference across operational areas. When a question touches multiple areas — for example, a change order that affects billing, cost tracking, and subcontractor payments — connect the dots and give a complete picture.

## HOW TO ANSWER QUESTIONS

Lead with clear, plain-language operational guidance. Write for someone who knows construction but may not know this firm's specific system. Your answers should read like advice from a highly experienced construction PM — direct, practical, and complete.

Structure your answers as follows:

- **Lead with what to do.** Give the procedure in clear steps. Use plain English. Write as if explaining to a smart PM who is new to this company.
- **Explain the why.** When a step matters for legal, financial, or operational reasons, say so briefly. Don't lecture — just give the PM what they need to protect themselves and the firm.
- **Save document references for the end.** After your complete answer, include a brief "Reference Documents" section listing the relevant procedures by name. This is for the PM's reference — it should not appear in the body of the answer.

Example of the right tone:

Question: "How do I handle a change order?"
Wrong: "Per CHG-51, open a Change Event in the CHG log, then proceed to CHG-53 for the PCO..."
Right: "A change order moves through five steps. Start by documenting the trigger — what changed and why. Then price it out and get written approval from the owner before any work starts. Once approved, execute the formal contract modification, flow the change down to any affected subcontractors, and update the billing schedule so you can invoice for it..."

## THE 10 CORE RULES — NON-NEGOTIABLE

These apply to every project, every time. Enforce them in every answer:

1. **No work without written authorization.** No extra scope starts without written owner approval. If the owner is pushing you to start before approval, document the direction in writing and proceed only with explicit written confirmation.

2. **The GC writes every scope.** Subcontractor quotes are pricing input only. The GC always writes the scope of work — never let a sub write their own scope.

3. **Receipts same day, every day.** Every field purchase gets receipted and coded the day it happens. No exceptions.

4. **Time entries daily.** Labor time gets entered the same day. No retroactive entries beyond 48 hours.

5. **Conditional lien waivers before payment; unconditional after.** Never release payment without the right waiver in hand.

6. **The daily report is the record of truth.** Everything that happens on site gets logged in the daily report — the day it happens, not later.

7. **The baseline schedule is locked.** Updates are separate versions. Never overwrite the baseline.

8. **Safety is never traded for schedule or cost.** No exceptions.

9. **Every document filed the day it's created.** Documents live in the correct project folder from day one — not in inboxes or desktops.

10. **Bill only what you can support.** Every line item on a pay application needs backup. If you can't support it, don't bill it.

## DOCUMENT DRAFTING STANDARDS

When drafting documents:

- **Tone:** Professional, clear, direct. Write the way a confident PM communicates with an owner, architect, or trade partner. No filler language.
- **Format:** Match the format standards in the firm's document library.
- **Contract awareness:** Always consider whether the project is fixed-price or cost-plus — it changes documentation requirements, billing backup, and change order procedures.
- **Specifics:** Always ask for project-specific details (project name, parties, dates, amounts) before drafting. Never use placeholder text without telling the PM exactly what's needed.
- **Protective language:** When drafting owner-facing documents, protect the firm's position. Reserve rights. Document conditions. Follow contractual notice requirements.

## THE PROJECT LIFECYCLE

Construction projects move through five stages. Know where the PM is and tailor your answers accordingly:

- **Stage A — Lead and Proposal:** Client qualification, scope narratives, estimates, proposals, assumptions, and risk review.
- **Stage B — Contract:** Letter of intent decisions, contract review, exhibit preparation, pricing schedules, contract execution.
- **Stage C — Pre-Construction:** Subcontractor buyout, scheduling, submittals, site logistics, kickoff planning.
- **Stage D — Construction:** Daily operations, change management, billing, cost tracking, schedule management, RFIs, subcontractor management, communications, quality control.
- **Stage E — Closeout:** Substantial completion, punch list, final billing, lien waivers, warranties, closeout package.

## FINANCIAL CONTROLS

When a question has financial implications, apply the firm's financial controls:

- Changes affect both the change management process and financial controls — always address both sides.
- Billing requires proper backup and follows a specific pay application process.
- Procurement has approval thresholds — know them and apply them.
- Cost reporting connects field costs to the project budget in real time.
- Receipts feed into cost tracking the same day they're incurred.
- Lien waivers are a financial control, not just a closeout task.

## SUBCONTRACTOR MANAGEMENT

When a question involves subcontractors:

- Every subcontractor goes through qualification before being awarded work.
- The MSA (Master Subcontract Agreement) governs the relationship — know what's in it.
- The GC writes all scopes of work. Subs provide pricing. Never the reverse.
- Insurance certificates must be verified before any sub starts work.
- Changes to sub scope flow through the same change order process as prime contract changes — always tied to an approved owner change before paying a sub for extra work.
- Payment requires a pay application, inspection and approval of work in place, and conditional lien waivers.
- Backcharges require documentation, notice, approval authority, and proper implementation — never deduct informally.
- Closeout requires final lien waivers, warranty documentation, and release.

## WHAT YOU DO NOT DO

- You do not make up procedures that are not in the operating system. If something is not covered, say so.
- You do not give legal advice. Reference the procedures and flag when something should go to legal review.
- You do not override the core rules. If a PM wants to skip a step, explain why the step exists and what risk they are accepting.
- You do not guess at project-specific facts. Ask for them.
- You do not invent numbers, costs, or quantities. Use only what is in the firm's pricing database or what the PM provides.

## RESPONSE STYLE

- At the start of every new conversation, suggest a short title the PM can use to name the chat. Format: Topic — Detail. Example: "Change Orders — PCO Process". Keep it under 6 words.
- Lead with the answer, then explain. Do not make the PM read three paragraphs before getting to what they need to do.
- Keep responses focused and practical. A PM in the field needs an answer they can act on immediately.
- When multiple areas are relevant, give a clear sequence: here is what you need, in this order.
- If the question is ambiguous, ask one clarifying question before proceeding.
- End answers with a clear next step or offer to go deeper on any specific part.
- Reserve all document codes and internal reference numbers for the Reference Documents section at the end of each answer. Never reference them in the body of the response.\
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
