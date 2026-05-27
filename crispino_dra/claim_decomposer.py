"""
claim_decomposer.py — Identifies discrete claim items within a claim submission.

A single claim submission letter (like the one BuildPro sent to ACME) typically
contains MULTIPLE distinct claim items, each with its own contractual basis,
factual narrative, and quantum. These must be analyzed separately, because they
may have wildly different merits.

This module uses Claude to read the full claim text and extract a structured list
of claim items. Each item becomes a unit of work for the downstream analyzer.

Design choices:
  - We use Claude's JSON output mode (structured_output via system prompt) rather
    than parsing free text. This is more reliable for downstream consumption.
  - We keep this module FAST and CHEAP — it's just classification, not analysis.
    We use Sonnet for now; could downgrade to Haiku later for cost optimization.
  - The decomposer DOES NOT pass the contract. It only sees the claim text.
    Cross-referencing claims against contract clauses happens in the analyzer.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv


# Load environment variables (ANTHROPIC_API_KEY) once at module load
load_dotenv()


# Anthropic client. We instantiate once at module level; the client is thread-safe
# and reusable across calls. Saves a few ms per call.
_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    """Lazy-instantiate the Anthropic client. Avoids errors at import time
    if the API key is missing (lets `python -c "import crispino_dra"` succeed)."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found. Ensure .env exists in the project root "
                "and contains: ANTHROPIC_API_KEY=sk-ant-api03-..."
            )
        _client = Anthropic(api_key=api_key)
    return _client


@dataclass
class ClaimItem:
    """
    A single discrete claim item identified within a larger claim submission.

    Attributes:
        item_number: Sequential number assigned by the decomposer (1, 2, 3, ...).
        title: Short descriptive title (e.g., "Foundation Design Delays — EOT").
        claim_type: Category (e.g., "Extension of Time", "Variation", "Force Majeure").
        contractual_basis_cited: Clauses the claimant relies on (e.g., "Clause 8.2(c), 20.1").
        summary: 2-3 sentence summary of what is being claimed and why.
        relief_sought: What the claimant wants (e.g., "18 Days EOT and £67,400").
        source_section: Where in the claim doc this item is found (e.g., "Section 2").
    """
    item_number: int
    title: str
    claim_type: str
    contractual_basis_cited: str
    summary: str
    relief_sought: str
    source_section: str

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Prompt construction
# ============================================================

DECOMPOSER_SYSTEM_PROMPT = """\
You are a legal claim analyst specialised in construction contracts. Your single \
job is to read a claim submission document and identify the discrete claim items \
it contains.

A "claim item" is a self-contained matter for which the claimant seeks specific \
relief (extension of time, additional payment, force majeure relief, etc.) on a \
specific contractual or factual basis. A single submission letter may contain \
ONE claim item or MANY. Each item must be analysed separately downstream, so \
your decomposition determines the entire analysis pipeline.

Rules:
1. Identify EVERY discrete claim item, even those clearly weak or defective. Do not \
filter on merit — that is the analyst's job, not yours.
2. If two adjacent paragraphs argue the same matter with the same contractual basis, \
they are ONE item. If two paragraphs argue different matters or different bases, they \
are TWO items.
3. Do not invent items. If a paragraph is purely contextual narrative without seeking \
specific relief, it is not a claim item.
4. Be faithful to the claimant's framing. Use the claimant's own language for titles \
and contractual basis, even if you disagree with it.

You will return your output as a single JSON object with this exact structure:

{
  "items": [
    {
      "item_number": 1,
      "title": "<short descriptive title>",
      "claim_type": "<e.g., Extension of Time, Variation, Force Majeure, Disruption, etc.>",
      "contractual_basis_cited": "<clauses cited by the claimant, e.g., 'Clause 8.2(c), 20.1'>",
      "summary": "<2-3 sentence summary of what is claimed and why>",
      "relief_sought": "<what the claimant wants, e.g., '18 Days EOT and £67,400'>",
      "source_section": "<where in the document, e.g., 'Section 2' or 'pages 3-4'>"
    },
    ...
  ]
}

Return ONLY the JSON. No preamble, no commentary, no markdown fences."""


DECOMPOSER_USER_TEMPLATE = """\
Below is the full text of a claim submission. Identify and decompose the discrete \
claim items it contains, following the rules in the system prompt.

=== CLAIM SUBMISSION TEXT ===

{claim_text}

=== END OF CLAIM SUBMISSION TEXT ===

Return the JSON object now."""


# ============================================================
# Public API
# ============================================================

def decompose_claim(claim_text: str, model: str = "claude-sonnet-4-5") -> list[ClaimItem]:
    """
    Analyse a claim submission and return a structured list of discrete claim items.

    Args:
        claim_text: The full text of the claim submission (from document_loader).
        model: Anthropic model identifier. Default is Sonnet 4.5.

    Returns:
        List of ClaimItem instances, one per discrete matter in the submission.

    Raises:
        RuntimeError: If the API call fails or returns malformed JSON.
    """
    if not claim_text or len(claim_text.strip()) < 100:
        raise ValueError("claim_text is empty or too short to decompose.")

    client = _get_client()

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=DECOMPOSER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": DECOMPOSER_USER_TEMPLATE.format(claim_text=claim_text),
            }
        ],
    )

    # Extract the text content
    raw_output = response.content[0].text.strip()

    # Strip markdown fences if Claude wrapped its JSON (defensive — system prompt
    # says not to, but models occasionally do anyway)
    if raw_output.startswith("```"):
        # Drop the first line (e.g., "```json") and the last line ("```")
        lines = raw_output.split("\n")
        raw_output = "\n".join(lines[1:-1]).strip()

    # Parse the JSON
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Decomposer returned malformed JSON: {e}\n"
            f"First 500 chars of output:\n{raw_output[:500]}"
        ) from e

    if "items" not in parsed:
        raise RuntimeError(
            f"Decomposer output missing 'items' key. Got keys: {list(parsed.keys())}"
        )

    # Convert each dict into a ClaimItem dataclass
    items: list[ClaimItem] = []
    for raw_item in parsed["items"]:
        try:
            items.append(ClaimItem(**raw_item))
        except TypeError as e:
            raise RuntimeError(
                f"Decomposer returned a claim item with unexpected fields: {raw_item}. "
                f"Error: {e}"
            ) from e

    return items


# ============================================================
# CLI test — run this file directly
# ============================================================
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    # We import document_loader at runtime so this module can be imported standalone
    from crispino_dra.document_loader import load_pdf

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]claim_decomposer.py — Smoke Test[/bold cyan]",
        border_style="cyan",
    ))

    # Load the synthetic claim
    console.print("\n[bold]Step 1:[/bold] Loading the synthetic claim...")
    claim_doc = load_pdf("data/claims/sample_claim.pdf")
    console.print(f"  Loaded {claim_doc.page_count} pages, "
                  f"{claim_doc.approximate_tokens:,} approximate tokens.")

    # Decompose it
    console.print("\n[bold]Step 2:[/bold] Calling Claude to decompose the claim...")
    console.print("  [dim](This will make one API call, ~$0.01-0.03)[/dim]")
    items = decompose_claim(claim_doc.text)
    console.print(f"  Decomposer identified [bold green]{len(items)}[/bold green] "
                  f"discrete claim items.")

    # Display each item in a nicely formatted panel
    console.print("\n[bold]Step 3:[/bold] Identified claim items\n")
    for item in items:
        body = (
            f"[bold]Type:[/bold] {item.claim_type}\n"
            f"[bold]Contractual basis:[/bold] {item.contractual_basis_cited}\n"
            f"[bold]Relief sought:[/bold] {item.relief_sought}\n"
            f"[bold]Source:[/bold] {item.source_section}\n\n"
            f"[bold]Summary:[/bold]\n{item.summary}"
        )
        console.print(Panel(
            body,
            title=f"[bold cyan]Claim Item {item.item_number} — {item.title}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        ))

    # Sanity check — we expect exactly 3 items from the synthetic claim
    console.print(f"\n[dim]Expected: 3 items (designed scenarios for strong / defective / weak claims).[/dim]")
    if len(items) == 3:
        console.print("[bold green]✓ Item count matches expectation.[/bold green]")
    else:
        console.print(f"[yellow]⚠ Got {len(items)} items; expected 3. "
                      f"Inspect the decomposer output above.[/yellow]")

    console.print("")