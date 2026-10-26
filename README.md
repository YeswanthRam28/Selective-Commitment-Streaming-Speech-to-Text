# Selective Commitment Streaming Speech-to-Text

A real-time streaming speech-to-text application implementing **Selective Commitment** to decouple provisional hypotheses from committed text.

## Features
- **Selective Commitment Engine**: Words remain provisional while uncertain, allowing later acoustic updates to revise them. Once stable across $K$ consecutive updates or upon reaching a latency deadline, words commit.
- **Hypothesis Alignment**: Prevents whole-transcript flicker; isolates revisions only to affected provisional spans without corrupting stable prefixes.
- **Provider Adapters**:
  - `DeterministicMockRecognizer`: Scripted incremental playback with intentional acoustic revisions for deterministic testing and reproducible experiments.
  - `AWSTranscribeAdapter`: True cloud real-time streaming recognition via AWS Transcribe.
- **Baselines Included**:
  - `Raw Streaming`: Direct display of partial hypotheses without policy filtering.
  - `Fixed Delay`: Monotonic lag-based emission.
  - `Selective Commitment`: Stability + confidence + deadline-based emission.
- **Evaluation Suite**: Automated Word Error Rate (WER) scoring, first-display vs. commitment latency tracking, and revision/flicker counting.

---

## Quickstart

### 1. Backend (FastAPI + Python 3.13)
```powershell
cd backend
venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- Health Check: `http://127.0.0.1:8000/health`
- WebSocket Streaming: `ws://127.0.0.1:8000/ws/transcribe`
- Replay Evaluation: `POST http://127.0.0.1:8000/api/replay`

### 2. Frontend (Vite + React + TypeScript)
```powershell
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Running Automated Tests
```powershell
backend\venv\Scripts\python -m pytest backend\tests
```
