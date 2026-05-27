"""
audit_logger.py — Structured audit logging for Crispino.DRA.

Every meaningful agent action is logged to a JSONL file (one JSON object per line)
plus to the terminal in human-readable form. This serves two purposes:

  1. PRODUCTION-PATTERN ACCOUNTABILITY. In a real legal-AI deployment, every agent
     decision must be traceable — who ran it, when, what inputs, what outputs, how
     long. This module mirrors that pattern even at MVP scale.

  2. DEMO MATERIAL. The audit log is itself a deliverable. Reviewers can see the
     agent's reasoning trail step by step, including the parallel execution pattern.

Design notes:
  - JSONL (JSON Lines) format — one event per line, easy to parse, easy to append.
  - Thread-safe via a lock — important because the orchestrator runs claim analyses
    in parallel and multiple threads will write to the log concurrently.
  - Human-readable terminal output uses `rich` for colour/formatting.
  - One log file per Crispino run, timestamped. Stored in logs/.
"""

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console


# Single global console — Rich is thread-safe by default
_console = Console()

# Project root logs directory
_LOGS_DIR = Path("logs")
_LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class AuditEvent:
    """A single audit log entry."""
    event_id: str                       # UUID for cross-referencing
    timestamp: str                       # ISO 8601 with timezone
    run_id: str                          # Groups events from one Crispino run
    component: str                       # "decomposer", "analyzer", "orchestrator", etc.
    action: str                          # "start", "complete", "error", "decision"
    item_number: Optional[int] = None   # Which claim item (None for top-level events)
    duration_seconds: Optional[float] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLogger:
    """
    Thread-safe audit logger. One instance per Crispino run.

    Usage:
        logger = AuditLogger()                              # opens a new run
        with logger.timed("analyzer", "analyze_item_3"):    # logs start + complete
            do_thing()
        logger.event("decision", "escalating to human", details={"reason": "..."})
        logger.close()                                       # finalises the run
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.log_path = _LOGS_DIR / f"{self.run_id}.jsonl"
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._event_count = 0

        # Open the file in append mode; create if absent
        self._fh = open(self.log_path, "a", encoding="utf-8")

        self.event("orchestrator", "run_start", details={"run_id": self.run_id})

    def event(
        self,
        component: str,
        action: str,
        item_number: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        details: Optional[dict] = None,
    ) -> str:
        """Log a single event. Returns the event_id."""
        evt = AuditEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=self.run_id,
            component=component,
            action=action,
            item_number=item_number,
            duration_seconds=duration_seconds,
            details=details or {},
        )

        with self._lock:
            self._fh.write(json.dumps(evt.to_dict()) + "\n")
            self._fh.flush()
            self._event_count += 1

        self._print_to_terminal(evt)
        return evt.event_id

    def timed(self, component: str, action: str, item_number: Optional[int] = None):
        """Context manager that logs start and complete events with duration."""
        return _TimedBlock(self, component, action, item_number)

    def error(
        self,
        component: str,
        message: str,
        item_number: Optional[int] = None,
        details: Optional[dict] = None,
    ) -> str:
        """Log an error event."""
        combined = {"message": message, **(details or {})}
        return self.event(component, "error", item_number=item_number, details=combined)

    def close(self) -> dict:
        """Finalise the run. Returns a summary."""
        total = time.time() - self._start_time
        summary = {
            "run_id": self.run_id,
            "event_count": self._event_count,
            "total_seconds": round(total, 2),
            "log_path": str(self.log_path),
        }
        self.event("orchestrator", "run_complete", details=summary)
        self._fh.close()
        return summary

    def _print_to_terminal(self, evt: AuditEvent):
        """Pretty-print one event to the terminal."""
        colour = {
            "run_start": "bold cyan",
            "run_complete": "bold cyan",
            "start": "dim",
            "complete": "green",
            "error": "bold red",
            "decision": "yellow",
        }.get(evt.action, "white")

        item_tag = f"[#{evt.item_number}] " if evt.item_number is not None else ""
        duration_tag = f" ({evt.duration_seconds:.2f}s)" if evt.duration_seconds else ""

        _console.print(
            f"[dim]{evt.timestamp.split('T')[1][:8]}[/dim] "
            f"[{colour}]{evt.component}.{evt.action}[/{colour}] "
            f"{item_tag}"
            f"[dim]{duration_tag}[/dim]"
        )


class _TimedBlock:
    """Context manager returned by AuditLogger.timed(). Logs start + complete with duration."""

    def __init__(self, logger: AuditLogger, component: str, action: str,
                 item_number: Optional[int]):
        self.logger = logger
        self.component = component
        self.action = action
        self.item_number = item_number
        self.start: float = 0.0

    def __enter__(self):
        self.start = time.time()
        self.logger.event(self.component, f"{self.action}_start",
                          item_number=self.item_number)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start
        if exc_type is None:
            self.logger.event(
                self.component, f"{self.action}_complete",
                item_number=self.item_number,
                duration_seconds=round(duration, 3),
            )
        else:
            self.logger.error(
                self.component,
                f"{self.action} failed: {exc_val}",
                item_number=self.item_number,
                details={"exception_type": exc_type.__name__ if exc_type else None,
                         "duration_seconds": round(duration, 3)},
            )
        return False  # Don't suppress exceptions


# ============================================================
# CLI test
# ============================================================
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    import time as t

    c = Console()
    c.print(Panel.fit("[bold cyan]audit_logger.py — Smoke Test[/bold cyan]",
                      border_style="cyan"))

    logger = AuditLogger()
    c.print(f"\n[dim]Logging to: {logger.log_path}[/dim]\n")

    # Simulate some agent activity
    logger.event("decomposer", "decompose_claim",
                 details={"claim_pages": 6, "items_found": 3})

    with logger.timed("analyzer", "analyze", item_number=1):
        t.sleep(0.2)

    with logger.timed("analyzer", "analyze", item_number=2):
        t.sleep(0.15)

    logger.event("analyzer", "decision", item_number=2,
                 details={"confidence": "MEDIUM", "time_bar": "FAILED"})

    # Simulate an error
    try:
        with logger.timed("analyzer", "analyze", item_number=3):
            raise ValueError("Simulated error for testing")
    except ValueError:
        pass  # Test expects this

    summary = logger.close()

    c.print("\n[bold green]✓ Audit logger smoke test complete.[/bold green]\n")
    c.print(Panel(
        f"Run ID: {summary['run_id']}\n"
        f"Events logged: {summary['event_count']}\n"
        f"Total duration: {summary['total_seconds']}s\n"
        f"Log file: {summary['log_path']}",
        title="[bold green]Run Summary[/bold green]",
        border_style="green",
    ))

    # Read back the log and print first/last lines to verify
    c.print("\n[dim]Log file content (verification):[/dim]")
    with open(logger.log_path) as f:
        lines = f.readlines()
    for line in lines:
        c.print(f"  [dim]{line.strip()}[/dim]")