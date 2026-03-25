# CareCaller Hackathon 2026

**AI Voice Agent Analytics** — Solutions for both problem statements of the CareCaller Hackathon 2026.

CareCaller is an AI-powered healthcare validation call system where AI voice agents make outbound calls to patients for medication refill check-ins, collecting health questionnaire responses via natural conversation.

---

## Problem 1: Call Quality Auto-Flagger

**Goal:** Automatically detect calls that require human review (`has_ticket` prediction).

### Approach: 3-Layer Cascading Ensemble

| Layer | Method | What It Catches |
|-------|--------|----------------|
| **Rule-Based** | 10 deterministic rules with 100% precision | Whisper mismatches, medical advice violations, outcome miscategorizations |
| **LightGBM** | Gradient boosting on 74 engineered features | Subtle data capture errors, edge cases |
| **Ensemble** | Cascading priority logic with threshold tuning | Best of both — high recall + zero false positives |

### Results (Validation Set: 144 calls, 11 tickets)

| Model | F1 | Recall | Precision |
|-------|-----|--------|-----------|
| Rules only | 0.8421 | 0.7273 | 1.0000 |
| LightGBM only | 0.9091 | 0.9091 | 0.9091 |
| **Ensemble** | **0.9524** | **0.9091** | **1.0000** |

### Feature Engineering (74 features)
- **36 Metadata features** — call duration, word counts, turn counts, whisper mismatches, one-hot encoded outcomes
- **24 Text/NLP features** — validation notes keyword signals, transcript medical advice detection, outcome consistency checks
- **14 Response features** — weight outlier detection, answer consistency, outcome-response mismatches

### Run
```bash
cd "Problem Statement-1"
pip install -r requirements.txt

python3 train.py              # Train + evaluate on validation set
python3 predict.py 0.48       # Generate submission CSV
```

---

## Problem 2: AI Voice Agent Simulator

**Goal:** Build a real-time voice agent that conducts medication refill check-in calls.

### Approach: OpenAI Realtime API + Function Calling

- **Real-time voice conversation** via WebSocket (OpenAI Realtime API, PCM16 @ 24kHz)
- **Server-side VAD** — automatic speech detection, no manual silence thresholds
- **Function calling** — `record_response` captures each Q&A answer in real-time, `end_call` records the outcome
- **Echo suppression** — mic muted while agent speaks, 300ms buffer after playback
- **Structured output** — clean JSON with all 14 question-answer pairs, outcome, and timestamps

### Call Flow
1. Greet patient by name, confirm identity
2. Confirm interest in medication refill
3. Time check (2 minutes)
4. Ask all 14 health questions sequentially
5. Close the call

### Edge Cases Handled
- Wrong number → apologize and end call
- Opt-out → confirm and end gracefully
- Reschedule → offer callback time
- Escalation → transfer to care team
- Medical advice requests → redirect to doctor (guardrail)
- Pricing questions → redirect to billing team

### Run
```bash
cd "Problem Statement-2"
pip install -r requirements.txt
export OPENAI_API_KEY='your-key-here'

# Voice mode (mic + speaker)
python3 run.py --name "John Smith" --medication "Tirzepatide" --dosage "2.5mg weekly injection"

# Text simulation mode (for testing)
python3 text_agent.py --name "John Smith"

# Save call record
python3 run.py --name "John Smith" --save output/call_record.json
```

---

## Project Structure

```
├── Datasets/                          # Shared datasets
│   ├── csv/                           # Flat tabular format
│   ├── json/                          # Nested with metadata
│   ├── parquet/                       # Columnar binary format
│   ├── transcript_samples.json        # 35 sample transcripts
│   └── DATA_DICTIONARY.md            # Field descriptions
│
├── Problem Statement-1/               # Call Quality Auto-Flagger
│   ├── config.py                      # Paths, constants, hyperparameters
│   ├── data_loader.py                 # Load JSON data splits
│   ├── features/
│   │   ├── metadata_features.py       # Numeric + categorical features
│   │   ├── text_features.py           # NLP from transcripts + validation notes
│   │   ├── response_features.py       # Q&A analysis features
│   │   └── build_features.py          # Feature orchestrator
│   ├── models/
│   │   ├── rule_based.py              # 10 high-precision deterministic rules
│   │   ├── gradient_boost.py          # LightGBM classifier
│   │   └── ensemble.py               # Cascading ensemble logic
│   ├── evaluation.py                  # F1, recall, precision metrics
│   ├── train.py                       # Training pipeline
│   ├── predict.py                     # Submission CSV generator
│   └── output/
│       └── submission.csv             # Test set predictions
│
├── Problem Statement-2/               # AI Voice Agent Simulator
│   ├── config.py                      # API keys, audio settings
│   ├── prompts.py                     # System prompt for Jessica (TrimRX agent)
│   ├── questionnaire.py              # 14 health questions + CallRecord
│   ├── voice_agent.py                 # Realtime voice agent (WebSocket + audio)
│   ├── text_agent.py                  # Text-based simulation mode
│   └── run.py                         # Entry point
│
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Framework | LightGBM, scikit-learn |
| Voice API | OpenAI Realtime API (WebSocket) |
| Audio I/O | sounddevice (PCM16, 24kHz) |
| NLP | Regex-based feature extraction |
| LLM | GPT-4o (text agent), GPT-4o Realtime (voice agent) |

## Dependencies

```bash
# Problem 1
pip install pandas numpy scikit-learn lightgbm

# Problem 2
pip install openai websockets sounddevice numpy python-dotenv

# macOS only (for LightGBM)
brew install libomp
```

---

**Built for CareCaller Hackathon 2026 | March 19, 2026**
