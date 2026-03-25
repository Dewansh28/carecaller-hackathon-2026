"""Text-based version of the TrimRX agent for testing without audio.

Uses OpenAI's chat completions API to simulate the same conversation
flow, allowing rapid iteration on the system prompt and tool logic.

Usage:
    python text_agent.py
    python text_agent.py --name "Jane Doe" --medication "Semaglutide" --dosage "0.5mg weekly injection"
"""
import argparse
import json
import sys
from openai import OpenAI

from config import OPENAI_API_KEY, DEFAULT_PATIENT
from prompts import build_system_prompt
from questionnaire import CallRecord, QUESTIONS


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_response",
            "description": (
                "Record a patient's answer to a health check-in question. "
                "Call this after the patient answers each of the 14 questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question_index": {
                        "type": "integer",
                        "description": (
                            "0-based index of the question (0-13). "
                            "Q0=feeling, Q1=weight, Q2=height, Q3=weight_lost, "
                            "Q4=side_effects, Q5=satisfaction, Q6=goal_weight, "
                            "Q7=dosage_request, Q8=new_meds, Q9=new_conditions, "
                            "Q10=allergies, Q11=surgeries, Q12=doctor_questions, "
                            "Q13=address_changed"
                        ),
                    },
                    "answer": {
                        "type": "string",
                        "description": "The patient's answer, captured accurately.",
                    },
                },
                "required": ["question_index", "answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": (
                "End the call and record the outcome. Call this when the "
                "conversation is finished."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": [
                            "completed", "incomplete", "opted_out",
                            "scheduled", "escalated", "wrong_number",
                            "voicemail",
                        ],
                        "description": "The call outcome.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Brief notes about why the call ended.",
                    },
                },
                "required": ["outcome"],
            },
        },
    },
]


def run_text_agent(patient_name: str, medication: str, dosage: str):
    """Run the agent in text mode using chat completions."""
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    call_record = CallRecord(patient_name, medication, dosage)
    system_prompt = build_system_prompt(patient_name, medication, dosage)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "[System: The call has just connected. Start the conversation.]",
        },
    ]

    print("=" * 60)
    print("TrimRX Voice Agent — Text Simulation")
    print("=" * 60)
    print(f"  Patient: {patient_name}")
    print(f"  Medication: {medication} ({dosage})")
    print("  Type your responses as the patient.")
    print("  Type 'quit' to end the call.")
    print("=" * 60)
    print()

    call_ended = False

    while not call_ended:
        # Get agent response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        # Handle tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if name == "record_response":
                    idx = args["question_index"]
                    answer = args["answer"]
                    call_record.set_answer(idx, answer)
                    q = QUESTIONS[idx] if idx < len(QUESTIONS) else f"Q{idx}"
                    print(f"  [Recorded Q{idx + 1}]: {answer}")
                    result = {"status": "recorded", "question_index": idx}

                elif name == "end_call":
                    outcome = args["outcome"]
                    notes = args.get("notes", "")
                    call_record.set_outcome(outcome)
                    print(f"\n  [Call ended: {outcome}]")
                    if notes:
                        print(f"  [Notes: {notes}]")
                    result = {"status": "call_ended", "outcome": outcome}
                    call_ended = True

                else:
                    result = {"error": f"Unknown function: {name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            # If there was text content alongside tool calls, print it
            if message.content:
                print(f"\n  Jessica: {message.content}")

            # Continue to get the agent's next spoken response
            if not call_ended:
                continue

        # Print agent's spoken response
        if message.content:
            print(f"\n  Jessica: {message.content}")

        if call_ended:
            break

        # Get patient input
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  [Call interrupted]")
            call_record.set_outcome("incomplete")
            break

        if user_input.lower() == "quit":
            call_record.set_outcome("incomplete")
            break

        messages.append({"role": "user", "content": user_input})

    # Print summary
    print("\n" + "=" * 60)
    print("CALL SUMMARY")
    print("=" * 60)
    print(call_record.summary())

    return call_record


def main():
    parser = argparse.ArgumentParser(description="TrimRX Text Agent — Simulation Mode")
    parser.add_argument("--name", default=DEFAULT_PATIENT["name"], help="Patient name")
    parser.add_argument("--medication", default=DEFAULT_PATIENT["medication"], help="Medication")
    parser.add_argument("--dosage", default=DEFAULT_PATIENT["dosage"], help="Dosage")
    parser.add_argument("--save", default=None, help="Save call record to JSON file")
    args = parser.parse_args()

    record = run_text_agent(args.name, args.medication, args.dosage)

    if args.save:
        from pathlib import Path
        Path(args.save).write_text(record.to_json())
        print(f"\nCall record saved to {args.save}")


if __name__ == "__main__":
    main()
