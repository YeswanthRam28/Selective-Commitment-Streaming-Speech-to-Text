# Testing Guide: Selective Commitment Streaming ASR

This document provides a comprehensive test plan and step-by-step verification procedures for the **TWO 26 Selective Commitment Speech Recognition** system, covering on-device PyTorch deep learning models (NVIDIA RTX 3050 GPU), model training routines, automated unit tests, offline reproducible benchmarks, live streaming playback, and microphone edge cases.

---

## 1. System & Service Architecture Under Test

- **Backend API**: FastAPI (`http://127.0.0.1:8000`)
  - Health Endpoint: `GET /health` (reports CUDA GPU status, PyTorch model availability, and active policies)
  - Offline Replay Scorer: `POST /api/replay`
  - Real-Time WebSocket Streaming: `ws://127.0.0.1:8000/ws/transcribe`
- **Frontend Client**: Vite + React + TypeScript (`http://localhost:5173/`)
- **Acoustic Recognizer Adapters**:
  - `DeepLearningRecognizer` (**Default / On-Device**): Runs a PyTorch CTC Acoustic Neural Network (Mel Spectrogram $\rightarrow$ 1D Conv $\rightarrow$ BiLSTM $\rightarrow$ CTC Projection) locally on the **NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.4)** without external cloud APIs.
  - `DeterministicMockRecognizer`: Scripted incremental playback with simulated acoustic revisions (`test` $\rightarrow$ `testing`) for reproducible offline evaluation.
  - `AWSTranscribeAdapter`: Optional cloud streaming fallback via AWS Transcribe.

---

## 2. Prerequisites & Environment Check

Before executing the tests, verify both the backend and frontend services are running:

### A. Verify Backend & GPU Acceleration
Open a terminal and run:
```powershell
backend\venv\Scripts\python -c "import httpx; r = httpx.get('http://127.0.0.1:8000/health'); print(r.json())"
```
**Expected Output:**
```json
{
  "status": "healthy",
  "deep_learning_available": true,
  "whisper_available": true,
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3050 A Laptop GPU",
  "aws_streaming_available": true,
  "supported_policies": ["raw_streaming", "fixed_delay", "selective_commitment"]
}
```

### B. Verify Frontend
Open your browser and navigate to:
```
http://localhost:5173/
```
Verify the top status pills display:
- **`● CUDA Active (NVIDIA GeForce RTX 3050 A Laptop GPU)`**
- **`OpenAI Whisper (GPU)`** (or CTC Wav2Vec2)

---

## 3. Deep Learning Model Training & Automated Unit Tests

### A. Test On-Device PyTorch CTC Acoustic Model Training
Verify that the custom neural network can train directly on your RTX 3050 GPU using pure PyTorch `nn.CTCLoss`:

```powershell
cd d:\Projects\College\Cloud\backend
venv\Scripts\python.exe -m app.ml.train
```

**Expected Training Output:**
```text
 Starting CTC Acoustic Model Training on device: CUDA
Epoch 01/05 | CTC Loss: 5.9738 | Time: 0.76s
Epoch 02/05 | CTC Loss: 3.2603 | Time: 0.19s
Epoch 03/05 | CTC Loss: 2.9375 | Time: 0.18s
Epoch 04/05 | CTC Loss: 2.9097 | Time: 0.18s
Epoch 05/05 | CTC Loss: 2.6144 | Time: 0.22s
 Training complete. Model checkpoint saved to: backend/app/ml/checkpoints/model.pt
```

### B. Run Backend Pytest Suite
Run the unit test suite verifying text normalization, Levenshtein WER computation, hypothesis alignment, and policy transitions:

```powershell
cd d:\Projects\College\Cloud
backend\venv\Scripts\python -m pytest backend\tests -v
```

**What This Tests:**
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

### Test Scenario 1: Real-Time Deep Learning Streaming (RTX 3050 GPU)
**Goal:** Verify on-device neural acoustic inference from live microphone audio without third-party cloud APIs.

1. Open `http://localhost:5173/` in your browser.
2. Confirm **Recognizer Adapter** is set to **Local Deep Learning Model (GPU)**.
3. Set **Commitment Policy** to **Selective Commitment (Proposed)**.
4. Click **"Start Streaming"** (or top-right **"Start Stream ↗"**).
5. When prompted by your browser, grant microphone access.
6. Speak a sentence clearly (e.g., *"We are demonstrating real-time streaming speech recognition"*).
7. **Expected Observations**:
   - Audio chunks (16 kHz 16-bit linear PCM) are sent over the WebSocket.
   - The backend runs GPU forward inference every ~300ms using the PyTorch neural network.
   - Live partial hypotheses appear in the transcript area:
     - **Provisional tokens**: Highlighted in electric orange with dashed underlines and update counts (e.g., `1u`).
     - **Committed tokens**: Stable words turn ink-black with solid underlines once they persist for $\ge K$ updates.
8. Click **"Stop Streaming"**.
9. The final utterance reconciliation executes, and all remaining words transition to finalized.

---

### Test Scenario 2: Offline Replay Benchmark (Zero Hardware Dependency)
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

### Test Scenario 3: Visual Selective Commitment & Revision Animation
**Goal:** Visually verify the distinct states of provisional spans, committed spans, and revisions in real-time.

1. Switch **Recognizer Adapter** to **Deterministic Synthetic Mock**.
2. Set **Commitment Policy** to **Selective Commitment (Proposed)**.
3. Set **Stability Threshold** to **2 updates**.
4. Click **"Start Streaming"**.
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

---

### Test Scenario 4: Baseline Policy Comparisons
**Goal:** Validate that the proposed policy achieves lower display latency than fixed-delay and lower visual churn than raw streaming.

| Policy | Configuration | Expected Visual Behavior | Expected Revision Churn |
| :--- | :--- | :--- | :--- |
| **Selective Commitment** | `Stability = 2`, `Deadline = 2.5s` | Spans commit individually as each stabilizes. Stable prefix remains locked. | Minimal (1 revision on true acoustic correction). |
| **Raw Streaming** | `Raw Streaming Baseline` | All tokens remain provisional (orange dashed) throughout the entire utterance until the very end. High visual instability. | High potential flicker. |
| **Fixed Delay** | `Lag Frontier = 3 tokens` | Monotonically commits tokens strictly 3 words behind the frontier, regardless of token confidence or stability. | Rigid delay; slow first commitment. |

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
| **1** | PyTorch CUDA GPU Detection | `backend/app/main.py /health` | `cuda_available: true`, `gpu_name: NVIDIA GeForce RTX 3050 A Laptop GPU` |
| **2** | CTC Acoustic Model Training | `python -m app.ml.train` | Model trains on CUDA, loss decreases, checkpoint saved |
| **3** | Unit Tests | `pytest backend/tests` | 6/6 tests passing (100%) |
| **4** | Replay Benchmark API | `POST /api/replay` | WER: 0.0%, 1 revision tracked |
| **5** | Real-time Deep Learning Mic ASR | Live audio streaming | Neural inference decodes speech on GPU into selective commitment spans |
| **6** | UI Theme & Layout | Browser inspection | TWO 26 warm cream palette, giant orange title, stickers, retro cards |
| **7** | Token State Distinction | Visual test | Committed (black/solid), Provisional (orange/dashed), Revised (purple/badge) |
| **8** | Policy Switching | Controls dropdown | Live switching between Selective, Raw, and Fixed Delay |
| **9** | Web Audio Capture | Microphone test | 16kHz PCM audio stream via WebSockets |
| **10** | Session Finalization | Replay / Stop stream | All provisional tokens reconcile to final transcript |