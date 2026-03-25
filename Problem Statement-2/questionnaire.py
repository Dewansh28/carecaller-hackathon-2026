"""14-question health check-in questionnaire and response tracking."""
import json
from dataclasses import dataclass, field
from datetime import datetime


QUESTIONS = [
    "How have you been feeling overall?",
    "What's your current weight in pounds?",
    "What's your height in feet and inches?",
    "How much weight have you lost this past month in pounds?",
    "Any side effects from your medication this month?",
    "Satisfied with your rate of weight loss?",
    "What's your goal weight in pounds?",
    "Any requests about your dosage?",
    "Have you started any new medications or supplements since last month?",
    "Do you have any new medical conditions since your last check-in?",
    "Any new allergies?",
    "Any surgeries since your last check-in?",
    "Any questions for your doctor?",
    "Has your shipping address changed?",
]


@dataclass
class CallRecord:
    """Tracks the state and responses of a single call."""

    patient_name: str
    medication: str
    dosage: str
    outcome: str = "in_progress"
    responses: list = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: str = ""

    def __post_init__(self):
        if not self.responses:
            self.responses = [
                {"question": q, "answer": ""} for q in QUESTIONS
            ]

    def set_answer(self, question_index: int, answer: str) -> None:
        """Record an answer for a specific question."""
        if 0 <= question_index < len(self.responses):
            self.responses[question_index]["answer"] = answer

    def set_outcome(self, outcome: str) -> None:
        """Set the call outcome."""
        valid = {
            "completed", "incomplete", "opted_out",
            "scheduled", "escalated", "wrong_number", "voicemail",
        }
        if outcome in valid:
            self.outcome = outcome
            self.ended_at = datetime.now().isoformat()

    @property
    def answered_count(self) -> int:
        return sum(1 for r in self.responses if r["answer"].strip())

    @property
    def response_completeness(self) -> float:
        return self.answered_count / len(QUESTIONS)

    def to_dict(self) -> dict:
        return {
            "patient_name": self.patient_name,
            "medication": self.medication,
            "dosage": self.dosage,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "answered_count": self.answered_count,
            "response_completeness": self.response_completeness,
            "responses": self.responses,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Print a readable summary of the call."""
        lines = [
            f"Patient: {self.patient_name}",
            f"Medication: {self.medication} ({self.dosage})",
            f"Outcome: {self.outcome}",
            f"Answered: {self.answered_count}/{len(QUESTIONS)}",
            f"Completeness: {self.response_completeness:.0%}",
            "",
            "Responses:",
        ]
        for i, r in enumerate(self.responses, 1):
            answer = r["answer"] if r["answer"] else "(not answered)"
            lines.append(f"  Q{i}: {r['question']}")
            lines.append(f"  A{i}: {answer}")
            lines.append("")
        return "\n".join(lines)
