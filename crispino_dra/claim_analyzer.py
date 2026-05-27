"""
claim_analyzer.py — The core analytical module of Crispino.DRA.

For each discrete claim item identified by the decomposer, this module asks Claude
to produce a structured legal assessment that a human reviewer can rely on.

CRISPINO DOES NOT ISSUE VERDICTS. The brief surfaces:
  - Contractual entitlement analysis with clause citations
  - Procedural compliance check (notice obligations, time bars)
  - Causation and evidence assessment
  - Counterargument simulation (the opposing party's likely position)
  - Identified areas of ambiguity requiring human judgment
  - A confidence score for the analysis itself

The human reads this brief and decides resolution posture. Crispino is a rigorous
analytical partner, not an oracle.

Architectural note on structured output:
  We use Anthropic's tool_use API to force structured output. Claude returns a
  tool_use block whose input matches a strict JSON schema. This is significantly
  more reliable than asking Claude to "return JSON" in the prompt — long, complex
  outputs occasionally produced malformed JSON with the free-text approach. The
  schema-enforced approach essentially eliminates that failure mode.
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from crispino_dra.claim_decomposer import ClaimItem


load_dotenv()

_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    """Lazy-instantiate the Anthropic client."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found. Ensure .env exists in the project root."
            )
        _client = Anthropic(api_key=api_key)
    return _client


# ============================================================
# Data structures for the structured analysis output
# ============================================================

@dataclass
class EntitlementAnalysis:
    """Substantive merits of the claim under the contract (separate from procedure)."""
    contractual_basis_assessment: str
    relevant_clauses: list[str]
    supporting_factors: list[str]
    contra_indicators: list[str]


@dataclass
class ProceduralComplianceAnalysis:
    """Notice and procedural requirements — distinct from substantive merits."""
    notice_requirements_summary: str
    compliance_assessment: str
    time_bar_status: str
    other_procedural_issues: list[str]


@dataclass
class EvidenceAnalysis:
    """Evidentiary support and quantum substantiation assessment."""
    evidence_strength_summary: str
    evidentiary_gaps: list[str]
    quantum_observations: list[str]


@dataclass
class CounterArgument:
    """The opposing party's strongest realistic counter-position."""
    main_counter_position: str
    supporting_points: list[str]


@dataclass
class ClaimAssessmentBrief:
    """The full structured assessment brief for ONE claim item."""
    item_number: int
    item_title: str
    item_type: str

    entitlement: EntitlementAnalysis
    procedural: ProceduralComplianceAnalysis
    evidence: EvidenceAnalysis
    counterargument: CounterArgument

    areas_of_ambiguity: list[str]
    confidence: str
    confidence_reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Tool schema — this is what makes structured output reliable
# ============================================================

# This JSON schema is registered as a "tool" with Claude. Claude is required by
# the API to return data matching this schema. No free-text JSON parsing.
ASSESSMENT_TOOL = {
    "name": "submit_claim_assessment",
    "description": (
        "Submit the structured analytical brief for the claim item. "
        "All fields are required. The brief surfaces analysis only — it does NOT "
        "issue verdicts (valid / invalid)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entitlement": {
                "type": "object",
                "description": "Substantive contractual entitlement analysis.",
                "properties": {
                    "contractual_basis_assessment": {
                        "type": "string",
                        "description": "Whether the cited clauses actually support the relief sought, with reasoning. Specific, clause-referenced.",
                    },
                    "relevant_clauses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific clauses you relied on, each with reference number and brief description.",
                    },
                    "supporting_factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Factors that strengthen the entitlement argument.",
                    },
                    "contra_indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Factors that weaken or defeat the entitlement argument.",
                    },
                },
                "required": ["contractual_basis_assessment", "relevant_clauses",
                             "supporting_factors", "contra_indicators"],
            },
            "procedural": {
                "type": "object",
                "description": "Notice and procedural compliance analysis (separate from substantive merit).",
                "properties": {
                    "notice_requirements_summary": {
                        "type": "string",
                        "description": "What the contract requires for notice and procedure.",
                    },
                    "compliance_assessment": {
                        "type": "string",
                        "description": "Whether the claimant complied, with specific reference to dates and clauses.",
                    },
                    "time_bar_status": {
                        "type": "string",
                        "enum": ["PASSED", "FAILED", "UNCLEAR"],
                        "description": "Whether the time bar was passed, failed, or unclear.",
                    },
                    "other_procedural_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Procedural issues other than the primary time bar.",
                    },
                },
                "required": ["notice_requirements_summary", "compliance_assessment",
                             "time_bar_status", "other_procedural_issues"],
            },
            "evidence": {
                "type": "object",
                "description": "Evidentiary and quantum assessment.",
                "properties": {
                    "evidence_strength_summary": {
                        "type": "string",
                        "description": "Overall evidentiary assessment.",
                    },
                    "evidentiary_gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What the claimant did not substantiate.",
                    },
                    "quantum_observations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific observations about the cost build-up.",
                    },
                },
                "required": ["evidence_strength_summary", "evidentiary_gaps",
                             "quantum_observations"],
            },
            "counterargument": {
                "type": "object",
                "description": "Simulation of the opposing party's strongest realistic position.",
                "properties": {
                    "main_counter_position": {
                        "type": "string",
                        "description": "The opposing party's strongest realistic position in one paragraph.",
                    },
                    "supporting_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific supporting points the opposing party would raise.",
                    },
                },
                "required": ["main_counter_position", "supporting_points"],
            },
            "areas_of_ambiguity": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Genuine judgment calls flagged for the human reviewer.",
            },
            "confidence": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH"],
                "description": "Crispino's confidence in this analysis.",
            },
            "confidence_reasoning": {
                "type": "string",
                "description": "Why this confidence level (one sentence).",
            },
        },
        "required": ["entitlement", "procedural", "evidence", "counterargument",
                     "areas_of_ambiguity", "confidence", "confidence_reasoning"],
    },
}


# ============================================================
# Prompts
# ============================================================

ANALYZER_SYSTEM_PROMPT = """\
You are a senior construction-claims analyst. You assess a single claim item against \
the governing contract and produce a structured analytical brief for a human Head of \
Contracts to review.

CRITICAL CONSTRAINTS:

1. You DO NOT issue verdicts. You do not write "the claim is valid" or "should be \
rejected." You produce structured analysis; the human decides.

2. You DO cite specific contract clauses with reference numbers when relying on them. \
Uncited assertions are not analysis. If you cannot find a relevant clause, say so.

3. You DO surface counterarguments. For every claim, simulate the opposing party's \
strongest realistic position. This is not optional.

4. You DO distinguish substantive entitlement from procedural compliance. A claim can \
be substantively meritorious but fail on procedure (e.g., time bars). Both must be \
assessed separately.

5. You DO flag genuine ambiguity. If a clause is susceptible to multiple reasonable \
interpretations, do not pick one — present both and flag it for the human reviewer.

6. You DO NOT fabricate clause numbers, contract terms, dates, or facts not present \
in the documents you were given.

You submit your analysis by calling the submit_claim_assessment tool. Do not produce \
any other output — no preamble, no commentary, no final summary. Just the tool call."""


ANALYZER_USER_TEMPLATE = """\
Analyse Claim Item {item_number} from the claimant's submission, against the governing \
subcontract.

=== CLAIM ITEM UNDER ANALYSIS ===

Item number: {item_number}
Title: {item_title}
Claim type: {item_type}
Contractual basis cited by claimant: {item_basis}
Relief sought: {item_relief}

Claimant's narrative for this item:
{item_summary}

=== GOVERNING SUBCONTRACT (FULL TEXT) ===

{contract_text}

=== FULL CLAIM SUBMISSION CONTEXT ===

{claim_text}

=== END OF DOCUMENTS ===

Submit your structured analytical brief by calling submit_claim_assessment."""


# ============================================================
# Public API
# ============================================================

def analyze_claim_item(
    item: ClaimItem,
    contract_text: str,
    claim_text: str,
    model: str = "claude-sonnet-4-5",
) -> ClaimAssessmentBrief:
    """
    Analyse a single claim item against the contract.

    Args:
        item: The ClaimItem to analyse (output from the decomposer).
        contract_text: Full text of the governing contract.
        claim_text: Full text of the claim submission (for context).
        model: Anthropic model identifier.

    Returns:
        ClaimAssessmentBrief — structured analysis for the human reviewer.
    """
    client = _get_client()

    user_message = ANALYZER_USER_TEMPLATE.format(
        item_number=item.item_number,
        item_title=item.title,
        item_type=item.claim_type,
        item_basis=item.contractual_basis_cited,
        item_relief=item.relief_sought,
        item_summary=item.summary,
        contract_text=contract_text,
        claim_text=claim_text,
    )

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=ANALYZER_SYSTEM_PROMPT,
        tools=[ASSESSMENT_TOOL],
        tool_choice={"type": "tool", "name": "submit_claim_assessment"},
        messages=[{"role": "user", "content": user_message}],
    )

    # Find the tool_use block in the response
    tool_use_block = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_claim_assessment":
            tool_use_block = block
            break

    if tool_use_block is None:
        raise RuntimeError(
            f"Analyzer did not return a tool_use block for item {item.item_number}. "
            f"stop_reason: {response.stop_reason}"
        )

    parsed = tool_use_block.input  # Already a Python dict — no JSON parsing needed

    # Construct the typed brief
    try:
        return ClaimAssessmentBrief(
            item_number=item.item_number,
            item_title=item.title,
            item_type=item.claim_type,
            entitlement=EntitlementAnalysis(**parsed["entitlement"]),
            procedural=ProceduralComplianceAnalysis(**parsed["procedural"]),
            evidence=EvidenceAnalysis(**parsed["evidence"]),
            counterargument=CounterArgument(**parsed["counterargument"]),
            areas_of_ambiguity=parsed["areas_of_ambiguity"],
            confidence=parsed["confidence"],
            confidence_reasoning=parsed["confidence_reasoning"],
        )
    except (KeyError, TypeError) as e:
        raise RuntimeError(
            f"Analyzer tool output missing or malformed field for item {item.item_number}: {e}\n"
            f"Got keys: {list(parsed.keys())}"
        ) from e


# ============================================================
# CLI test
# ============================================================
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    from crispino_dra.document_loader import load_pdf
    from crispino_dra.claim_decomposer import decompose_claim

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]claim_analyzer.py — Smoke Test[/bold cyan]",
        border_style="cyan",
    ))

    console.print("\n[bold]Step 1:[/bold] Loading documents...")
    contract = load_pdf("data/contracts/sample_contract.pdf")
    claim = load_pdf("data/claims/sample_claim.pdf")
    console.print(f"  Contract: {contract.page_count} pages, "
                  f"{contract.approximate_tokens:,} tokens")
    console.print(f"  Claim: {claim.page_count} pages, "
                  f"{claim.approximate_tokens:,} tokens")

    console.print("\n[bold]Step 2:[/bold] Decomposing claim into items...")
    items = decompose_claim(claim.text)
    console.print(f"  Found {len(items)} claim items.")

    console.print("\n[bold]Step 3:[/bold] Analysing each item against the contract...")
    console.print(f"  [dim](This will make {len(items)} API calls, "
                  f"~$0.05-0.15 total)[/dim]")

    briefs = []
    for item in items:
        console.print(f"\n  → Analysing Item {item.item_number}: {item.title}...")
        brief = analyze_claim_item(item, contract.text, claim.text)
        briefs.append(brief)
        console.print(f"    Done. Confidence: [bold]{brief.confidence}[/bold], "
                      f"Time bar: [bold]{brief.procedural.time_bar_status}[/bold]")

    console.print("\n[bold]Step 4:[/bold] Assessment Briefs\n")

    for brief in briefs:
        colour = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(brief.confidence, "cyan")

        body_lines = [
            f"[bold]Type:[/bold] {brief.item_type}",
            f"[bold]Confidence:[/bold] [{colour}]{brief.confidence}[/{colour}] — {brief.confidence_reasoning}",
            "",
            "[bold]ENTITLEMENT[/bold]",
            f"  {brief.entitlement.contractual_basis_assessment}",
            f"  [dim]Relevant clauses:[/dim] {'; '.join(brief.entitlement.relevant_clauses[:3])}",
            f"  [dim]Supporting:[/dim] {'; '.join(brief.entitlement.supporting_factors[:2])}",
            f"  [dim]Contra-indicators:[/dim] {'; '.join(brief.entitlement.contra_indicators[:2])}",
            "",
            "[bold]PROCEDURAL COMPLIANCE[/bold]",
            f"  Time bar: [bold]{brief.procedural.time_bar_status}[/bold]",
            f"  {brief.procedural.compliance_assessment}",
            "",
            "[bold]EVIDENCE[/bold]",
            f"  {brief.evidence.evidence_strength_summary}",
            f"  [dim]Gaps:[/dim] {'; '.join(brief.evidence.evidentiary_gaps[:2])}",
            "",
            "[bold]COUNTERARGUMENT (opposing party's likely position)[/bold]",
            f"  {brief.counterargument.main_counter_position}",
            "",
            "[bold]AREAS OF AMBIGUITY (for human judgment)[/bold]",
            *[f"  • {a}" for a in brief.areas_of_ambiguity[:3]],
        ]

        console.print(Panel(
            "\n".join(body_lines),
            title=f"[bold {colour}]Item {brief.item_number} — {brief.item_title}[/bold {colour}]",
            border_style=colour,
            padding=(1, 2),
        ))

    console.print("\n[bold green]✓ Analyser smoke test complete.[/bold green]\n")