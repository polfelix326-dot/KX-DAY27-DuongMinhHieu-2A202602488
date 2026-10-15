"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")

    async def recent_sessions() -> list[dict]:
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       CASE MAX(CASE risk_level
                                WHEN 'high' THEN 3
                                WHEN 'med' THEN 2
                                ELSE 1
                           END)
                           WHEN 3 THEN 'high'
                           WHEN 2 THEN 'med'
                           ELSE 'low'
                       END AS worst_risk,
                       MAX(timestamp) AS last_event
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 25
                """
            ) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    sessions = asyncio.run(recent_sessions())
    if sessions:
        st.caption("Select a session to reopen it.")

        def select_recent_session() -> None:
            selected_rows = st.session_state.recent_sessions_table.selection.rows
            if not selected_rows:
                return
            selected = sessions[selected_rows[0]]
            st.session_state.thread_id = selected["thread_id"]
            st.session_state.pr_url = selected["pr_url"]
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.session_state.restore_thread_id = selected["thread_id"]

        st.dataframe(
            sessions,
            hide_index=True,
            on_select=select_recent_session,
            selection_mode="single-row",
            key="recent_sessions_table",
            column_config={
                "thread_id": st.column_config.TextColumn("thread_id"),
                "pr_url": st.column_config.LinkColumn("pr_url"),
                "worst_risk": st.column_config.TextColumn("worst_risk"),
                "last_event": st.column_config.TextColumn("last_event"),
            },
        )
    else:
        st.caption("No review sessions yet.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input(
        "Feedback (optional)",
        key=f"approval_feedback_{st.session_state.thread_id}",
    )
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for index, question in enumerate(payload["questions"]):
            answers[question] = st.text_input(
                question,
                key=f"escalation_answer_{st.session_state.thread_id}_{index}",
            )
        submitted = st.form_submit_button("Submit answers")
    if submitted:
        return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg,
            )
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


async def load_saved_session(thread_id: str) -> tuple[dict, dict | None]:
    """Load a checkpoint's current values and any pending interrupt payload."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
        payload = next(
            (
                interrupt.value
                for task in snapshot.tasks
                for interrupt in task.interrupts
            ),
            None,
        )
        return snapshot.values, payload


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.pop("restore_thread_id", None)

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Restore an existing durable graph checkpoint selected from Recent sessions.
restore_thread_id = st.session_state.get("restore_thread_id")
if restore_thread_id:
    with st.spinner("Restoring saved review session..."):
        values, payload = asyncio.run(load_saved_session(restore_thread_id))
    st.session_state.interrupt_payload = payload
    st.session_state.final = values if payload is None and values.get("final_action") else None
    st.session_state.pop("restore_thread_id", None)

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action in {"auto_committed", "committed", "committed_after_escalation"}:
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
        if action == "auto_committed":
            analysis = final.get("analysis")
            if analysis is not None:
                st.metric("Confidence", f"{analysis.confidence:.0%}")
                st.markdown(analysis.summary)
            st.link_button("View PR and review comment", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    elif action.endswith("_commit_failed") or action == "commit_failed":
        st.error(f"Could not post the review comment ({action}).")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
