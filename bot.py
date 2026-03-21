"""
AttoSense MK1 — CLI Bot
=======================
Interactive command-line intent classifier.
Runs independently — no browser needed.

Usage:
  python bot.py                    # interactive mode
  python bot.py "book flight"      # single classification
  python bot.py --file input.txt   # classify each line
"""

import sys
import os
import json
import asyncio
import argparse
from pathlib import Path

# ── Path fix (run from project root) ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


# ── Colour helpers (Windows CMD compatible) ────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[31m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    BLUE   = "\033[34m"
    CYAN   = "\033[36m"
    WHITE  = "\033[97m"

    @staticmethod
    def enable_windows():
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )

DOMAIN_COLOUR = {
    "information": C.CYAN,
    "action":      C.GREEN,
    "problem":     C.RED,
    "transaction": C.BLUE,
    "creative":    "\033[35m",   # magenta
    "personal":    "\033[36m",   # cyan
    "technical":   C.YELLOW,
}


def _conf_bar(conf: float, width: int = 20) -> str:
    filled = int(conf * width)
    bar    = "█" * filled + "░" * (width - filled)
    col    = C.GREEN if conf >= 0.82 else C.YELLOW if conf >= 0.70 else C.RED
    return f"{col}{bar}{C.RESET} {int(conf*100)}%"


def _print_result(result: dict, latency_ms: float = 0) -> None:
    intent   = result.get("intent", "—")
    domain   = result.get("intent_domain", "information")
    conf     = result.get("confidence", 0)
    sent     = result.get("sentiment", "neutral")
    esc      = result.get("requires_escalation", False)
    steps    = result.get("reasoning_steps", [])
    entities = result.get("entities", [])
    competing= result.get("competing_intent")
    comp_c   = result.get("competing_confidence", 0)
    lang     = result.get("language_detected")

    domain_col = DOMAIN_COLOUR.get(domain, C.WHITE)

    print()
    print(f"  {C.BOLD}{C.WHITE}┌─ AttoSense Result ──────────────────────────{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  Intent  : {C.BOLD}{C.WHITE}{intent}{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  Domain  : {domain_col}{domain.upper()}{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  Conf    : {_conf_bar(conf)}")
    print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  Sentiment: {sent}  {'  🚨 ESCALATE' if esc else ''}")
    if lang:
        print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  Language : {lang}")
    if competing and comp_c > 0.04:
        print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  Runner-up: {competing} ({int(comp_c*100)}%)")
    if entities:
        ent_str = "  ".join(f"{C.DIM}[{e['label']}]{C.RESET} {e['value']}" for e in entities)
        print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  Entities : {ent_str}")
    if steps:
        print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  Reasoning:")
        for i, step in enumerate(steps, 1):
            col = C.GREEN if i == len(steps) else C.DIM
            print(f"  {C.BOLD}{C.WHITE}│{C.RESET}    {col}{i:02d}. {step}{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}│{C.RESET}  {C.DIM}Latency: {latency_ms:.0f} ms{C.RESET}")
    print(f"  {C.BOLD}{C.WHITE}└─────────────────────────────────────────────{C.RESET}")
    print()


async def _classify_one(text: str) -> dict | None:
    """Run one classification through the full pipeline."""
    try:
        from backend.core.multimodal import classify_text_async
        result, model, latency = await classify_text_async(text)
        return result.model_dump(), latency
    except EnvironmentError as e:
        print(f"\n  {C.RED}Error: {e}{C.RESET}")
        print(f"  {C.DIM}Set GROQ_API_KEY in your .env file.{C.RESET}\n")
        return None, 0
    except Exception as e:
        print(f"\n  {C.RED}Classification failed: {e}{C.RESET}\n")
        return None, 0


def _banner():
    print(f"""
{C.BOLD}{C.WHITE}  ╔═══════════════════════════════════════╗
  ║   AttoSense MK1 · Intent Intelligence  ║
  ║   Universal NLU · 3-stage pipeline     ║
  ╚═══════════════════════════════════════╝{C.RESET}
  {C.DIM}Type a message and press Enter to classify.
  Commands: :quit  :clear  :json  :help{C.RESET}
""")


async def interactive_loop():
    """Interactive REPL mode."""
    C.enable_windows()
    _banner()

    json_mode = False

    while True:
        try:
            user_input = input(f"  {C.CYAN}>{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {C.DIM}Goodbye.{C.RESET}\n")
            break

        if not user_input:
            continue

        if user_input.lower() in (":quit", ":exit", ":q"):
            print(f"\n  {C.DIM}Goodbye.{C.RESET}\n")
            break
        elif user_input.lower() == ":clear":
            os.system("cls" if sys.platform == "win32" else "clear")
            _banner()
            continue
        elif user_input.lower() == ":json":
            json_mode = not json_mode
            print(f"  {C.DIM}JSON mode {'ON' if json_mode else 'OFF'}{C.RESET}\n")
            continue
        elif user_input.lower() == ":help":
            print(f"""
  {C.BOLD}Commands:{C.RESET}
    :quit   — exit
    :clear  — clear screen
    :json   — toggle raw JSON output
    :help   — show this message

  {C.BOLD}Just type any message to classify its intent.{C.RESET}
  Examples:
    What is the capital of France?
    I need to book a flight to Tokyo next Friday
    My app keeps crashing with error 500
    Write me a haiku about autumn
    Debug this Python function
""")
            continue

        print(f"  {C.DIM}Classifying…{C.RESET}", end="\r")
        result, latency = await _classify_one(user_input)
        if result is None:
            continue

        if json_mode:
            print(f"  {json.dumps(result, indent=2)}")
        else:
            _print_result(result, latency)


async def single_classification(text: str) -> int:
    """Classify one message and exit."""
    C.enable_windows()
    result, latency = await _classify_one(text)
    if result is None:
        return 1
    _print_result(result, latency)
    return 0


async def file_classification(path: str) -> int:
    """Classify each non-empty line in a file."""
    C.enable_windows()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        print(f"\n  {C.RED}File not found: {path}{C.RESET}\n")
        return 1

    lines = [l.strip() for l in lines if l.strip()]
    print(f"\n  Classifying {len(lines)} inputs from {path}\n")

    for i, line in enumerate(lines, 1):
        print(f"  {C.DIM}[{i}/{len(lines)}] {line[:60]}{'…' if len(line)>60 else ''}{C.RESET}")
        result, latency = await _classify_one(line)
        if result:
            intent = result.get("intent","—")
            domain = result.get("intent_domain","—")
            conf   = result.get("confidence", 0)
            col    = DOMAIN_COLOUR.get(domain, C.WHITE)
            print(f"  {col}{domain:12}{C.RESET} {C.BOLD}{intent}{C.RESET}  ({int(conf*100)}%)\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="AttoSense MK1 — CLI Intent Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("message", nargs="?", help="Message to classify (omit for interactive mode)")
    parser.add_argument("--file", "-f", help="Classify each line in a text file")
    args = parser.parse_args()

    if args.file:
        sys.exit(asyncio.run(file_classification(args.file)))
    elif args.message:
        sys.exit(asyncio.run(single_classification(args.message)))
    else:
        asyncio.run(interactive_loop())


if __name__ == "__main__":
    main()
