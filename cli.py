"""Command-line interaction using the same dialogue engine as the website."""

from __future__ import annotations

import argparse
import sys

from restaurant_bot import RestaurantBot


def main() -> None:
    # Keep Unicode menu prices readable when output is redirected on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Chat with Olive Table without a browser.")
    parser.add_argument("--mode", choices=["enhanced", "baseline"], default="enhanced")
    parser.add_argument("--show-sources", action="store_true", help="Show evidence and NLP intent after each answer.")
    args = parser.parse_args()
    bot = RestaurantBot()
    state = {}
    print("Olive Table | offline restaurant assistant")
    print("Use English questions. Type /reset for a new conversation or /quit to leave.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if text.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye!")
            break
        if text.lower() == "/reset":
            state.clear()
            print("Assistant: Your conversation has been reset.")
            continue
        if not text:
            continue
        try:
            response = bot.reply(text, state, args.mode)
        except ValueError as exc:
            print(f"Please try again: {exc}")
            continue
        print(f"Assistant: {response['reply']}")
        if args.show_sources:
            print("Intent:", response["intent"], "| Score:", response["confidence"])
            print("Sources:", ", ".join(source["title"] for source in response["sources"]) or "Conversation / clarification")


if __name__ == "__main__":
    main()
