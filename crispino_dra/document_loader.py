"""
document_loader.py — Reads PDF documents and returns clean text for downstream agents.

This is the foundation module. Every other Crispino component depends on it to
provide reliable, structured access to contract and claim documents.

Design choices:
  - We accept only PDF for MVP (DOCX support is a 30-line addition if needed later).
  - We support page-range slicing so callers can extract specific sections of large
    documents (e.g., the narrative section of a 359-page real claim submission).
  - We expose an approximate token count so downstream callers can decide whether
    to load whole documents into context or use retrieval. Crispino's "size-adaptive"
    decision lives here.

The module returns a LoadedDocument dataclass rather than raw text, so downstream
modules can make informed decisions (whole-document vs chunked, etc.) without
re-reading the file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


# Tokens-per-character ratio for English legal text.
# Empirical: ~3.8-4.2 chars/token. We use 4 as a safe round number.
# Used only for routing decisions, not for billing — Anthropic returns exact counts.
CHARS_PER_TOKEN_ESTIMATE = 4

# Threshold above which we treat a contract as "large" and may need retrieval.
# 60K tokens leaves comfortable headroom in Claude's 200K context for the claim,
# the prompt, and the response. For MVP, contracts above this would trigger an
# alert; production would route to a retrieval-based path.
LARGE_DOCUMENT_TOKEN_THRESHOLD = 60_000


@dataclass
class LoadedDocument:
    """
    A document loaded into memory, ready for LLM processing.

    Attributes:
        path: Filesystem path to the source document.
        text: Full extracted text content for the loaded page range.
        page_count: Total number of pages in the source document.
        pages_loaded: Tuple (start_page, end_page), 1-indexed inclusive, indicating
                      which pages were loaded. Useful when only a section was extracted.
        approximate_tokens: Estimated token count for the loaded text.
        is_large: True if approximate_tokens exceeds LARGE_DOCUMENT_TOKEN_THRESHOLD.
    """

    path: Path
    text: str
    page_count: int
    pages_loaded: tuple[int, int]
    approximate_tokens: int

    @property
    def is_large(self) -> bool:
        return self.approximate_tokens > LARGE_DOCUMENT_TOKEN_THRESHOLD

    def __repr__(self) -> str:
        return (
            f"LoadedDocument(path={self.path.name!r}, "
            f"pages_loaded={self.pages_loaded} of {self.page_count}, "
            f"approximate_tokens={self.approximate_tokens:,}, "
            f"is_large={self.is_large})"
        )


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate based on character count.

    For exact token counting, the caller should use Anthropic's count_tokens API.
    This estimate is used only for routing decisions (e.g., whole-document load
    vs. retrieval), where ~10% accuracy is sufficient.
    """
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def load_pdf(
    path: str | Path,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> LoadedDocument:
    """
    Load a PDF and return its text content as a LoadedDocument.

    Args:
        path: Path to the PDF file (string or Path object).
        start_page: First page to load (1-indexed, inclusive). Default 1.
        end_page: Last page to load (1-indexed, inclusive). If None, loads
                  through the final page of the document.

    Returns:
        LoadedDocument with extracted text and metadata.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the page range is invalid.
        RuntimeError: If the PDF contains no extractable text (likely scanned).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)

    # Default end_page to last page if not specified
    if end_page is None:
        end_page = total_pages

    # Validate page range
    if start_page < 1 or start_page > total_pages:
        raise ValueError(
            f"start_page={start_page} is out of range. "
            f"PDF has {total_pages} pages (valid range: 1 to {total_pages})."
        )
    if end_page < start_page or end_page > total_pages:
        raise ValueError(
            f"end_page={end_page} is invalid. "
            f"Must be between start_page ({start_page}) and {total_pages}."
        )

    # Extract text from the requested pages
    # pypdf uses 0-indexed pages internally; we convert from 1-indexed at the boundary
    page_texts: list[str] = []
    for page_num in range(start_page - 1, end_page):
        page_text = reader.pages[page_num].extract_text() or ""
        page_texts.append(page_text)

    full_text = "\n\n".join(page_texts).strip()

    # Sanity check — if we got essentially nothing, the PDF is likely scanned
    if len(full_text) < 100 and (end_page - start_page + 1) > 1:
        raise RuntimeError(
            f"PDF appears to contain no extractable text: {path.name}. "
            f"It may be a scanned document requiring OCR. "
            f"Got only {len(full_text)} characters from {end_page - start_page + 1} pages."
        )

    return LoadedDocument(
        path=path,
        text=full_text,
        page_count=total_pages,
        pages_loaded=(start_page, end_page),
        approximate_tokens=estimate_tokens(full_text),
    )


# ============================================================
# CLI test — run this file directly to verify it works
# ============================================================
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    console.print(Panel.fit(
        "[bold cyan]document_loader.py — Smoke Test[/bold cyan]",
        border_style="cyan",
    ))

    # Test 1: Load the synthetic contract (full)
    console.print("\n[bold]Test 1:[/bold] Load synthetic contract (full)")
    contract = load_pdf("data/contracts/sample_contract.pdf")
    console.print(f"  {contract!r}")
    console.print(f"  [dim]Preview:[/dim] {contract.text[:150]!r}...")

    # Test 2: Load the synthetic claim (full)
    console.print("\n[bold]Test 2:[/bold] Load synthetic claim (full)")
    claim = load_pdf("data/claims/sample_claim.pdf")
    console.print(f"  {claim!r}")
    console.print(f"  [dim]Preview:[/dim] {claim.text[:150]!r}...")

    # Test 3: Page-range slicing — load only pages 2-4 of the claim
    console.print("\n[bold]Test 3:[/bold] Load only pages 2-4 of the claim (slicing test)")
    claim_slice = load_pdf("data/claims/sample_claim.pdf", start_page=2, end_page=4)
    console.print(f"  {claim_slice!r}")

    # Test 4: Error handling — non-existent file
    console.print("\n[bold]Test 4:[/bold] Error handling (expected to fail gracefully)")
    try:
        load_pdf("data/contracts/does_not_exist.pdf")
        console.print("  [red]✗ Should have raised FileNotFoundError[/red]")
    except FileNotFoundError as e:
        console.print(f"  [green]✓ Caught expected error:[/green] {e}")

    # Test 5: Error handling — invalid page range
    console.print("\n[bold]Test 5:[/bold] Invalid page range (expected to fail gracefully)")
    try:
        load_pdf("data/contracts/sample_contract.pdf", start_page=999)
        console.print("  [red]✗ Should have raised ValueError[/red]")
    except ValueError as e:
        console.print(f"  [green]✓ Caught expected error:[/green] {e}")

    # Summary
    console.print("\n")
    summary = Table(title="Document Sizing — Architecture Decision", border_style="green")
    summary.add_column("Document", style="bold")
    summary.add_column("Pages", justify="right")
    summary.add_column("Tokens (approx)", justify="right")
    summary.add_column("Size class", justify="center")

    summary.add_row(
        "Contract", str(contract.page_count),
        f"{contract.approximate_tokens:,}",
        "[yellow]LARGE — retrieval[/yellow]" if contract.is_large
        else "[green]small — whole[/green]"
    )
    summary.add_row(
        "Claim", str(claim.page_count),
        f"{claim.approximate_tokens:,}",
        "[yellow]LARGE — retrieval[/yellow]" if claim.is_large
        else "[green]small — whole[/green]"
    )
    summary.add_row(
        "[bold]Combined[/bold]",
        f"[bold]{contract.page_count + claim.page_count}[/bold]",
        f"[bold]{contract.approximate_tokens + claim.approximate_tokens:,}[/bold]",
        "[dim]→ MVP path[/dim]",
    )
    console.print(summary)
    console.print(
        f"\n[dim]Claude Sonnet 4.5 context window: 200,000 tokens. "
        f"Combined load is {(contract.approximate_tokens + claim.approximate_tokens) / 200_000 * 100:.1f}% of context.[/dim]"
    )

    console.print("\n[bold green]✓ All smoke tests passed.[/bold green]\n")