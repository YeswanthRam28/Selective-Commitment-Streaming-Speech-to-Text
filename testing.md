# Testing Guide: Selective Commitment Streaming ASR

This document provides a comprehensive test plan and step-by-step verification procedures for the **TWO 26 Selective Commitment Speech Recognition** system, covering automated unit tests, offline reproducible benchmarks, live streaming playback, and microphone edge cases.

---

## 1. System & Service Architecture Under Test

- **Backend API**: FastAPI (`http://127.0.0.1:8000`)
  - Health Endpoint: `GET /health`
  - Offline Replay Scorer: `POST /api/replay`
  - Real-Time WebSocket Streaming: `ws://127.0.0.1:8000/ws/transcribe`
- **Frontend Client**: Vite + React + TypeScript (`http://localhost:5173/`)
- **Adapters**:
  - `DeterministicMockRecognizer`: Scripted incremental playback with simulated acoustic revisions (`test` $\rightarrow$ `testing`).
  - `AWSTranscribeAdapter`: Real cloud streaming recognition via AWS Transcribe.

---

## 2. Prerequisites & Environment Check

Before executing the tests, verify both the backend and frontend services are running:

### A. Verify Backend
Open a terminal and run:
```powershell
backend\venv\Scripts\python -c "import httpx; r = httpx.get('http://127.0.0.1:8000/health'); print(r.json())"
```
**Expected Output:**
```json
{"status": "healthy", "aws_streaming_available": true, "supported_policies": ["raw_streaming", "fixed_delay", "selective_commitment"]}
```

### B. Verify Frontend
Open your browser and navigate to:
```
http://localhost:5173/
```
Verify the top status pill displays: **"Backend Engine Online"**.

---

## 3. Automated Test Suite (Pytest)

Run the backend unit tests for text normalization, Levenshtein WER computation, hypothesis alignment, and policy transitions:

```powershell
backend\venv\Scripts\python -m pytest backend\tests -v
```

### What This Tests:
1. `test_normalize_text`: Punctuation stripping, case folding, and whitespace normalization.
2. `test_compute_wer_exact_match`: Identical strings produce $0.0\%$ WER.
3. `test_compute_wer_substitution_and_deletion`: Confirms dynamic programming edit distance for substitutions, deletions, and insertions.
4. `test_alignment_incremental_and_revision`:
   - Hypothesis 1: `"We are test"`
   - Hypothesis 2: `"We are testing"`
   - Asserts word `"test"` is revised to `"testing"` with `revised_count = 1` and history `["test"]` without corrupting unaffected spans.
5. `test_selective_commitment_policy`:
   - Spans with `update_count < threshold` remain **Provisional**.
   - Spans with `update_count >= threshold` transition to **Committed**.
6. `test_raw_streaming_baseline`: Confirms tokens remain provisional until final utterance reconciliation.

**Expected Result:**
```
backend/tests/test_policy_and_alignment.py::test_normalize_text PASSED
backend/tests/test_policy_and_alignment.py::test_compute_wer_exact_match PASSED
backend/tests/test_policy_and_alignment.py::test_compute_wer_substitution_and_deletion PASSED
backend/tests/test_policy_and_alignment.py::test_alignment_incremental_and_revision PASSED
backend/tests/test_policy_and_alignment.py::test_selective_commitment_policy PASSED
backend/tests/test_policy_and_alignment.py::test_raw_streaming_baseline PASSED
============================== 6 passed in 1.29s ==============================
```

---

## 4. End-to-End Test Scenarios

### Test Scenario 1: Offline Replay Benchmark (Zero Hardware Dependency)
**Goal:** Verify repeatable evaluation metrics, WER computation, and revision tracking without needing microphone input.

1. Navigate to `http://localhost:5173/`.
2. Under **Commitment Policy**, select **Selective Commitment (Proposed)**.
3. Click the **"Run Replay Evaluation"** button.
4. **Expected Observations in UI**:
   - The **Offline Benchmark Results** card appears below the metrics grid.
   - **WER Badge**: Displays `WER: 0.0%`.
   - **Edits**: Displays `0 / 8 ref tokens`.
   - **Total Revisions (Flicker)**: Displays `1` (detecting the acoustic revision from `test` to `testing`).
   - **Final Transcript**: `"We are testing selective commitment in speech recognition."`

#### Testing via CLI (cURL / PowerShell):
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/replay" -Method Post -ContentType "application/json" -Body '{"policy_config": {"mode": "selective_commitment", "required_stability_updates": 2, "max_provisional_duration_sec": 2.5, "fixed_delay_updates": 3}}'
```

---

### Test Scenario 2: Visual Selective Commitment & Revision Animation
**Goal:** Visually verify the distinct states of provisional spans, committed spans, and revisions in real-time.

1. Keep **Recognizer Adapter** set to **Deterministic Synthetic Mock**.
2. Set **Commitment Policy** to **Selective Commitment (Proposed)**.
3. Set **Stability Threshold** to **2 updates**.
4. Click **"Start Streaming"** (or top-right **"Start Stream ↗"**).
5. **Observe the Live Transcript Area**:
   - Newly arrived words appear as **Provisional**:
     - Electric orange text with warm peach pillow background (`#ffebe3`).
     - Dashed orange underline.
     - Badge displaying update count (e.g. `1u`).
   - Words that remain stable across 2 partial hypotheses turn **Committed**:
     - Bold ink-black text.
     - Solid dark underline.
   - When the mock recognizer updates `"test"` to `"testing"`:
     - Word briefly flashes purple (`#7c3aed`) with a micro-bounce animation.
     - Revision pill badge appears: `↺1`.
     - Hovering over `"testing"` displays a tooltip showing the revision history: `(Prev: test)`.
   - Upon final hypothesis:
     - All remaining provisional tokens transition to committed/finalized.
     - Top pill updates from "Listening" to completed.

---

### Test Scenario 3: Baseline Policy Comparisons
**Goal:** Validate that the proposed policy achieves lower display latency than fixed-delay and lower visual churn than raw streaming.

| Policy | Configuration | Expected Visual Behavior | Expected Revision Churn |
| :--- | :--- | :--- | :--- |
| **Selective Commitment** | `Stability = 2`, `Deadline = 2.5s` | Spans commit individually as each stabilizes. Stable prefix remains locked. | Minimal (1 revision on true acoustic correction). |
| **Raw Streaming** | `Raw Streaming Baseline` | All tokens remain provisional (orange dashed) throughout the entire utterance until the very end. High visual instability. | High potential flicker. |
| **Fixed Delay** | `Lag Frontier = 3 tokens` | Monotonically commits tokens strictly 3 words behind the frontier, regardless of token confidence or stability. | Rigid delay; slow first commitment. |

**Steps to test**:
1. Switch **Commitment Policy** dropdown to **Raw Streaming**.
2. Click **Start Streaming** and observe that all tokens stay provisional until the end.
3. Switch **Commitment Policy** dropdown to **Fixed Delay**.
4. Click **Start Streaming** and observe tokens committing only once they lag 3 tokens behind.

---

### Test Scenario 4: Live Microphone Audio Capture
**Goal:** Verify browser microphone audio streaming through 16-bit 16 kHz linear PCM.

1. Ensure a working microphone is connected to your computer.
2. Under **Recognizer Adapter**, select **AWS Transcribe Streaming** (or Synthetic Mock if AWS credentials are not set).
3. Click **"Start Streaming"**.
4. When prompted by your browser, click **"Allow"** for microphone permissions.
5. Speak clearly into the microphone.
6. Verify:
   - Red/Orange "Listening" pulsing indicator appears in the transcript header.
   - Audio chunks are processed without UI stutter or memory leaks.
7. Click **"Stop Streaming"**.
8. Verify the audio tracks are released and the session gracefully finalizes.

---

### Test Scenario 5: Edge Cases & Error Handling

1. **Microphone Denial**:
   - In browser settings, block microphone access for `http://localhost:5173/`.
   - Click "Start Streaming".
   - **Expected**: A clear, non-crashing alert informs the user: *"Microphone access error: Permission denied"*. The UI state resets cleanly to Idle.
2. **Backend Unavailability**:
   - Stop the backend process (Ctrl+C on port 8000).
   - In the frontend, the top badge will change to **"Connecting to Backend..."** / Offline.
   - Clicking "Run Replay Evaluation" displays a friendly prompt to start the backend.
3. **Rapid Start/Stop**:
   - Rapidly click "Start Streaming" and immediately "Stop Streaming".
   - **Expected**: WebSockets and audio context clean up without orphaned tracks or memory leaks.

---

## 5. Test Verification Matrix

| # | Feature / Test Case | Tool / Method | Pass Criteria |
| :---: | :--- | :--- | :---: |
| **1** | Unit Tests | `pytest backend/tests` | 6/6 tests passing (100%) |
| **2** | Backend Health | `GET /health` | HTTP 200, status `healthy` |
| **3** | Replay Benchmark API | `POST /api/replay` | WER: 0.0%, 1 revision tracked |
| **4** | UI Theme & Layout | Browser inspection | Warm cream palette, giant orange title, stickers, retro cards |
| **5** | Token State Distinction | Visual test | Committed (black/solid), Provisional (orange/dashed), Revised (purple/badge) |
| **6** | Policy Switching | Controls dropdown | Live switching between Selective, Raw, and Fixed Delay |
| **7** | Web Audio Capture | Microphone test | 16kHz PCM audio stream via WebSockets |
| **8** | Session Finalization | Replay / Stop stream | All provisional tokens reconcile to final transcript |