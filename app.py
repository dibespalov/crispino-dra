"""
app.py — Streamlit web interface for Crispino.DRA.

This is the user-facing entry point. It wraps the orchestrator with:
  - File upload widgets for contract and claim PDFs
  - Live progress streaming as the pipeline runs
  - Rendered Markdown briefs displayed in expandable sections
  - Download buttons for the generated reports

Run from the project root with:
    streamlit run app.py
"""

import shutil
import tempfile
import threading
import time
from pathlib import Path
from queue import Queue, Empty

import streamlit as st

from crispino_dra.orchestrator import run_crispino, CrispinoResult


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Crispino.DRA — Dispute Resolution Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS for a cleaner look
st.markdown("""
<style>
  .main-header {
    font-size: 2.4rem;
    color: #1F3864;
    margin-bottom: 0;
    font-weight: 700;
  }
  .subtitle {
    color: #5B6B85;
    font-size: 1.05rem;
    margin-top: 0;
    margin-bottom: 1rem;
  }
  .stage-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    background: #EEF2F8;
    color: #2E5599;
    font-size: 0.85rem;
    font-weight: 500;
    margin-right: 6px;
  }
  .confidence-high   { color: #2E7D32; font-weight: 700; }
  .confidence-medium { color: #B07810; font-weight: 700; }
  .confidence-low    { color: #C0392B; font-weight: 700; }
  .timebar-passed    { color: #2E7D32; font-weight: 700; }
  .timebar-failed    { color: #C0392B; font-weight: 700; }
  .timebar-unclear   { color: #B07810; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================

st.markdown('<p class="main-header">⚖️ Crispino.DRA</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Dispute Resolution Agent — preliminary structured analysis of '
    'construction claims. Crispino assesses; humans decide.</p>',
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar — about / how it works
# ============================================================

with st.sidebar:
    st.markdown("### How Crispino works")
    st.markdown(
        "1. **Upload** the governing contract and the claim submission. \n"
        "2. **Decompose** — Crispino identifies discrete claim items. \n"
        "3. **Analyse** — each item is assessed in parallel against the contract. \n"
        "4. **Review** — Crispino flags items requiring human judgment."
    )
    st.markdown("---")
    st.markdown("### What Crispino does **not** do")
    st.markdown(
        "Crispino does **not issue verdicts**. It surfaces entitlement analysis, "
        "procedural compliance, evidence assessment, and adversarial counterargument. "
        "The Head of Contracts makes the resolution decision."
    )
    st.markdown("---")
    st.markdown("### Demo files")
    st.markdown(
        "If you want to test without uploading, the project repository includes synthetic "
        "test files at `data/contracts/sample_contract.pdf` and `data/claims/sample_claim.pdf`."
    )


# ============================================================
# Helpers
# ============================================================

def _save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    """Save a Streamlit UploadedFile to disk and return the path."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / uploaded_file.name
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def _confidence_pill(value: str) -> str:
    cls = {"HIGH": "confidence-high", "MEDIUM": "confidence-medium",
           "LOW": "confidence-low"}.get(value, "")
    return f'<span class="{cls}">{value}</span>'


def _timebar_pill(value: str) -> str:
    cls = {"PASSED": "timebar-passed", "FAILED": "timebar-failed",
           "UNCLEAR": "timebar-unclear"}.get(value, "")
    return f'<span class="{cls}">{value}</span>'


def _run_pipeline_in_thread(
    contract_path: Path, claim_path: Path, queue: Queue
):
    """Run the orchestrator in a worker thread, pushing progress events to a queue."""
    def progress(stage: str, detail: str = ""):
        queue.put(("progress", stage, detail))

    try:
        result = run_crispino(
            contract_path=str(contract_path),
            claim_path=str(claim_path),
            progress_callback=progress,
        )
        queue.put(("complete", result, None))
    except Exception as e:
        queue.put(("error", type(e).__name__, str(e)))


# ============================================================
# Main input section
# ============================================================

st.markdown("### 1. Upload your documents")

col1, col2 = st.columns(2)
with col1:
    contract_file = st.file_uploader(
        "Governing contract (PDF)",
        type=["pdf"],
        key="contract_upload",
        help="The contract under which the claim is brought. For the demo, "
             "use data/contracts/sample_contract.pdf from the project repo.",
    )
with col2:
    claim_file = st.file_uploader(
        "Claim submission (PDF)",
        type=["pdf"],
        key="claim_upload",
        help="The claim being assessed. For the demo, use "
             "data/claims/sample_claim.pdf from the project repo.",
    )

# Demo-mode shortcut: load synthetic files directly from project paths
use_demo = st.checkbox(
    "Use built-in synthetic test files (skip upload)",
    value=False,
    help="Loads data/contracts/sample_contract.pdf and data/claims/sample_claim.pdf "
         "directly. Useful for live demonstrations.",
)


# ============================================================
# Run button
# ============================================================

ready_to_run = use_demo or (contract_file is not None and claim_file is not None)

st.markdown("### 2. Run Crispino")
run_clicked = st.button(
    "🚀 Analyse the claim",
    type="primary",
    disabled=not ready_to_run,
    use_container_width=False,
)

if not ready_to_run:
    st.caption("⤴️ Upload both documents (or tick the demo checkbox) to enable the run button.")


# ============================================================
# Execute the pipeline
# ============================================================

if run_clicked:
    # Resolve input paths
    if use_demo:
        contract_path = Path("data/contracts/sample_contract.pdf")
        claim_path = Path("data/claims/sample_claim.pdf")
        if not contract_path.exists() or not claim_path.exists():
            st.error("Demo files not found in `data/contracts/` or `data/claims/`. "
                     "Either upload your own or check the project setup.")
            st.stop()
    else:
        # Save uploaded files to a temp directory the orchestrator can read
        upload_dir = Path("data/_uploads")
        contract_path = _save_uploaded_file(contract_file, upload_dir)
        claim_path = _save_uploaded_file(claim_file, upload_dir)

    st.markdown("### 3. Live execution trace")
    st.caption("Watch Crispino decompose the claim and analyse items in parallel.")

    # Containers for streaming output
    progress_container = st.empty()
    log_container = st.empty()
    log_lines = []

    queue: Queue = Queue()
    worker = threading.Thread(
        target=_run_pipeline_in_thread,
        args=(contract_path, claim_path, queue),
        daemon=True,
    )
    worker.start()

    result: CrispinoResult | None = None
    error_info: tuple | None = None
    start_time = time.time()

    # Per-item live status — populated as item_start events arrive,
    # updated as item_complete events arrive
    items_in_flight: dict[int, dict] = {}

    # Two separate containers so each can refresh independently
    items_container = st.empty()       # Top: the live items panel
    log_container_text = st.empty()    # Bottom: the running event log

    def render_items_panel():
        """Refresh the per-item live status table."""
        if not items_in_flight:
            return

        md = "#### Claim items being analysed\n\n"
        md += "| Item | Title | Status |\n"
        md += "|------|-------|--------|\n"
        for num in sorted(items_in_flight.keys()):
            info = items_in_flight[num]
            status_text = info["status"]
            md += f"| **{num}** | {info['title']} | {status_text} |\n"
        items_container.markdown(md)

    # Poll the queue and update the UI as events arrive
    with st.spinner("Crispino is reading your documents..."):
        while worker.is_alive() or not queue.empty():
            try:
                event = queue.get(timeout=0.5)
            except Empty:
                elapsed = time.time() - start_time
                progress_container.markdown(f"⏱️ Elapsed: **{elapsed:.0f}s**")
                continue

            if event[0] == "progress":
                _, stage, detail = event

                # Special handling for per-item events
                if stage == "item_start":
                    # detail format: "Item N: title"
                    try:
                        prefix, title = detail.split(":", 1)
                        item_num = int(prefix.replace("Item", "").strip())
                        title = title.strip()
                        items_in_flight[item_num] = {
                            "title": title,
                            "status": "🔄 Running...",
                        }
                        render_items_panel()
                    except (ValueError, IndexError):
                        log_lines.append(f"**{stage}** — {detail}")

                elif stage == "item_complete":
                    # detail format: "Item N done — confidence: X, time bar: Y"
                    try:
                        prefix = detail.split("done")[0]
                        item_num = int(prefix.replace("Item", "").strip())
                        # Extract confidence and time bar from the detail string
                        conf = "?"
                        tb = "?"
                        if "confidence:" in detail:
                            conf_part = detail.split("confidence:")[1].split(",")[0].strip()
                            conf = conf_part
                        if "time bar:" in detail:
                            tb = detail.split("time bar:")[1].strip()
                        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡",
                                      "LOW": "🔴"}.get(conf, "⚪")
                        tb_emoji = {"PASSED": "✅", "FAILED": "❌",
                                    "UNCLEAR": "⚠️"}.get(tb, "❓")
                        if item_num in items_in_flight:
                            items_in_flight[item_num]["status"] = (
                                f"✅ **Complete** · {conf_emoji} {conf} · {tb_emoji} {tb}"
                            )
                            render_items_panel()
                    except (ValueError, IndexError):
                        log_lines.append(f"**{stage}** — {detail}")

                elif stage == "item_error":
                    try:
                        prefix = detail.split("FAILED")[0]
                        item_num = int(prefix.replace("Item", "").strip())
                        if item_num in items_in_flight:
                            items_in_flight[item_num]["status"] = "❌ **Failed**"
                            render_items_panel()
                    except (ValueError, IndexError):
                        log_lines.append(f"**{stage}** — {detail}")

                else:
                    # General stage event — append to the running log
                    log_lines.append(f"**{stage}** — {detail}")
                    log_container_text.markdown(
                        "\n\n".join(f"• {line}" for line in log_lines)
                    )

            elif event[0] == "complete":
                _, result, _ = event
                break
            elif event[0] == "error":
                error_info = event
                break

    worker.join(timeout=2)
    progress_container.empty()

    # ============================================================
    # Handle results
    # ============================================================

    if error_info:
        st.error(
            f"**Crispino encountered an error:** "
            f"`{error_info[1]}: {error_info[2]}`. "
            f"Check the terminal for the full traceback."
        )
        st.stop()

    if result is None:
        st.error("Crispino finished but no result was produced. "
                 "Check the terminal logs.")
        st.stop()

    # ----------- Success header -----------
    st.success(
        f"✅ Analysis complete in **{result.duration_seconds:.0f}s**. "
        f"Crispino identified **{len(result.items)}** discrete claim items and "
        f"flagged **{len(result.items_needing_review)}** for human review."
    )

    # ----------- Summary table -----------
    st.markdown("### 4. Summary of findings")

    table_md = "| # | Title | Type | Confidence | Time Bar |\n"
    table_md += "|---|-------|------|------------|----------|\n"
    for brief in result.briefs:
        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(brief.confidence, "⚪")
        tb_emoji = {"PASSED": "✅", "FAILED": "❌", "UNCLEAR": "⚠️"}.get(
            brief.procedural.time_bar_status, "❓"
        )
        table_md += (
            f"| {brief.item_number} "
            f"| {brief.item_title} "
            f"| {brief.item_type} "
            f"| {conf_emoji} {brief.confidence} "
            f"| {tb_emoji} {brief.procedural.time_bar_status} |\n"
        )
    st.markdown(table_md)

    # ----------- Items flagged for review -----------
    if result.items_needing_review:
        with st.expander(
            f"⚠️ {len(result.items_needing_review)} item(s) flagged for "
            f"explicit human review",
            expanded=True,
        ):
            for brief in result.items_needing_review:
                reasons = []
                if brief.confidence in ("LOW", "MEDIUM"):
                    reasons.append(f"confidence: **{brief.confidence}**")
                if brief.procedural.time_bar_status in ("FAILED", "UNCLEAR"):
                    reasons.append(f"time bar: **{brief.procedural.time_bar_status}**")
                st.markdown(
                    f"- **Item {brief.item_number}** — {brief.item_title} "
                    f"({'; '.join(reasons)})"
                )

    # ----------- Per-item briefs (expandable) -----------
    st.markdown("### 5. Detailed assessment briefs")
    st.caption("Click each item to expand its full structured analysis.")

    for brief in result.briefs:
        conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(brief.confidence, "⚪")
        tb_emoji = {"PASSED": "✅", "FAILED": "❌", "UNCLEAR": "⚠️"}.get(
            brief.procedural.time_bar_status, "❓"
        )

        # Read the rendered Markdown file
        brief_md_path = result.output_dir / f"brief_item_{brief.item_number:02d}.md"
        if brief_md_path.exists():
            brief_md = brief_md_path.read_text(encoding="utf-8")
        else:
            brief_md = "*Brief file not found.*"

        with st.expander(
            f"Item {brief.item_number} — {brief.item_title}  "
            f"·  {conf_emoji} {brief.confidence}  ·  {tb_emoji} "
            f"{brief.procedural.time_bar_status}",
            expanded=False,
        ):
            st.markdown(brief_md)

    # ----------- Downloads -----------
    st.markdown("### 6. Downloads")
    dl_col1, dl_col2 = st.columns(2)

    # Summary memo download
    summary_path = result.output_dir / "00_summary_memo.md"
    if summary_path.exists():
        with dl_col1:
            st.download_button(
                "📄 Download summary memo (Markdown)",
                data=summary_path.read_text(encoding="utf-8"),
                file_name=f"{result.run_id}_summary.md",
                mime="text/markdown",
            )

    # Audit log download
    log_path = result.output_dir / "audit_log.jsonl"
    if log_path.exists():
        with dl_col2:
            st.download_button(
                "📊 Download audit log (JSONL)",
                data=log_path.read_text(encoding="utf-8"),
                file_name=f"{result.run_id}_audit.jsonl",
                mime="application/json",
            )

    st.caption(
        f"All output files are also written to disk at "
        f"`{result.output_dir}` for downstream use."
    )

    # ----------- Footer note -----------
    st.markdown("---")
    st.caption(
        "**Crispino.DRA Disclaimer.** This output is preliminary structured analysis "
        "intended to support — not replace — professional judgment. Crispino does not "
        "issue verdicts. Resolution posture decisions and accountability rest with "
        "the Head of Contracts and authorised legal counsel."
    )

else:
    # Show a placeholder until a run is triggered
    st.markdown("---")
    st.info(
        "👆 Upload your contract and claim, or enable the demo files, then click "
        "**Analyse the claim** to run Crispino."
    )