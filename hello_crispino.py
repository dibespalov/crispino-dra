"""
hello_crispino.py — Crispino's first words.

Smoke test verifying:
  - Python environment is configured
  - Anthropic SDK is installed
  - API key loads from .env
  - The Anthropic API is reachable and billed correctly

If this prints Crispino's introduction, the foundation is solid
and every later component builds on this same pattern.
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Load environment variables from .env into the running process
load_dotenv()

# Rich console for nicely formatted terminal output
console = Console()

# Pull the API key from the loaded environment
api_key = os.getenv("ANTHROPIC_API_KEY")

# Fail loudly and helpfully if the key didn't load
if not api_key:
    console.print("[bold red]ERROR:[/bold red] ANTHROPIC_API_KEY not found.")
    console.print("Check that .env exists in the project root and contains:")
    console.print("  ANTHROPIC_API_KEY=sk-ant-api03-...")
    raise SystemExit(1)

# Initialize the Anthropic client — this is your handle for every API call
client = Anthropic(api_key=api_key)

console.print(Panel.fit(
    "[bold cyan]Crispino.DRA[/bold cyan] is waking up...",
    border_style="cyan",
))

# The actual API call: send one message, receive one response
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=400,
    messages=[
        {
            "role": "user",
            "content": (
                "You are Crispino.DRA — a Dispute Resolution Agent in development. "
                "Your purpose is to analyze legal contracts and the claims raised against them, "
                "then produce a reasoned resolution memo for each claim, like a junior legal associate would. "
                "Introduce yourself in 3 short sentences. "
                "Be confident but not boastful. "
                "End with one sentence about what you need to start working."
            ),
        }
    ],
)

# Extract the text content from the response object
crispino_message = response.content[0].text

# Print Crispino's reply in a nicely framed box
console.print(Panel(
    crispino_message,
    title="[bold cyan]Crispino speaks[/bold cyan]",
    border_style="cyan",
    padding=(1, 2),
))

# Print token usage — useful for tracking cost
input_tokens = response.usage.input_tokens
output_tokens = response.usage.output_tokens
console.print(
    f"\n[dim]Tokens used — input: {input_tokens}, output: {output_tokens}[/dim]"
)

console.print("\n[bold green]Smoke test passed.[/bold green] Crispino.DRA is alive.\n")