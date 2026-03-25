"""Entry point for the TrimRX Voice Agent.

Usage:
    python run.py
    python run.py --name "Jane Doe" --medication "Semaglutide" --dosage "0.5mg weekly injection"
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from config import DEFAULT_PATIENT, OPENAI_API_KEY
from voice_agent import VoiceAgent


def main():
    parser = argparse.ArgumentParser(description="TrimRX Voice Agent — Medication Refill Check-in")
    parser.add_argument("--name", default=DEFAULT_PATIENT["name"], help="Patient name")
    parser.add_argument("--medication", default=DEFAULT_PATIENT["medication"], help="Medication name")
    parser.add_argument("--dosage", default=DEFAULT_PATIENT["dosage"], help="Dosage info")
    parser.add_argument("--save", default=None, help="Path to save call record JSON")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    print("=" * 60)
    print("TrimRX Voice Agent — Medication Refill Check-in")
    print("=" * 60)
    print(f"  Patient:    {args.name}")
    print(f"  Medication: {args.medication}")
    print(f"  Dosage:     {args.dosage}")
    print()
    print("  The agent will greet you and conduct the check-in.")
    print("  Speak naturally into your microphone.")
    print("  Press Ctrl+C to end the call early.")
    print("=" * 60)
    print()

    agent = VoiceAgent(args.name, args.medication, args.dosage)
    try:
        call_record = asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\n[Agent] Call interrupted.")
        call_record = agent.call_record
        if call_record.outcome == "in_progress":
            call_record.set_outcome("incomplete")

    # Save call record if requested
    if args.save and call_record:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(call_record.to_json())
        print(f"\nCall record saved to {save_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)
