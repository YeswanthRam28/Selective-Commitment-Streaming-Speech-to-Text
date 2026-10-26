Project mission
- Build a working real-time streaming speech-to-text application centered on selective commitment.
- The system must process speech incrementally instead of waiting for a complete sentence before showing text.
- The defining behavior is to distinguish provisional text from text the policy considers committed.
- Reliable regions should become stable without forcing unrelated uncertain regions to block all future output.
- Uncertain regions must remain editable by later recognition updates until the policy commits them or the utterance ends.
- Treat selective commitment as a policy layer over a streaming recognizer, not as a claim of a new foundation model.
- Keep the architecture modular so the recognizer can be replaced without rewriting the commitment engine.
- Prioritize correctness, measurable behavior, and clear state transitions over decorative features.
- Do not claim the method is novel or superior before testing it against meaningful baselines.
Core user workflow
- Let the user start and stop microphone capture explicitly.
- Request microphone permission only when the user initiates capture.
- Show a clear recording state while audio is being captured.
- Send audio incrementally to the recognition backend using a persistent streaming connection where supported.
- Render partial hypotheses as soon as the backend produces them.
- Mark provisional spans visually differently from committed spans.
- Continue accepting new audio while earlier spans remain uncertain.
- Apply transcript revisions to the smallest safely identifiable region.
- At utterance completion, reconcile the displayed text with the recognizer's final result.
Implementation assumptions
- Assume the project is a new implementation and do not assume an existing repository or model is available.
- Inspect the environment and current project files before choosing or replacing frameworks.
- Prefer a maintainable TypeScript frontend and Python backend unless the existing environment dictates otherwise.
- Keep frontend and backend responsibilities explicit.
- Use a streaming-capable recognizer whose incremental behavior has been verified in practice.
- Do not simulate streaming by splitting a completed transcript into fake chunks.
- Provide a mock recognizer only as a development fallback, clearly separated from real recognition.
- Make provider-specific logic implement a common internal interface.
- Document decisions that depend on the selected recognition provider.
Architecture boundaries
- Separate microphone capture, audio transport, recognition, hypothesis alignment, commitment policy, transcript rendering, and metrics.
- Keep recognition-provider adapters isolated from policy logic.
- Keep the commitment engine independent from UI framework components.
- Represent transcript updates as typed events rather than loosely structured strings.
- Represent words or spans with stable internal identifiers where alignment permits.
- Use timestamps and source offsets whenever the recognizer exposes them.
- Do not assume word indexes remain stable across changing hypotheses.
- Handle a whole-span rewrite without corrupting unaffected transcript regions.
- Design modules so each can be tested with deterministic synthetic events.
Audio capture
- Capture microphone audio using browser-supported APIs with explicit permission handling.
- Choose a sample rate and encoding supported by the selected recognizer.
- Resample audio only when required and do so consistently.
- Avoid recording large amounts of audio in memory when streaming chunks is possible.
- Use bounded audio buffers to prevent unbounded memory growth.
- Handle microphone denial, device removal, muted input, and empty input gracefully.
- Track monotonic timestamps for audio chunks and capture events.
- Preserve chunk ordering and prevent duplicate chunk submission.
- Stop tracks and release audio resources when recording ends.
Audio chunking and transport
- Use configurable chunk sizes and send intervals appropriate to the recognizer.
- Balance small chunks for responsiveness against transport and processing overhead.
- Preserve ordering across all audio messages.
- Include a session identifier and sequence number with each chunk.
- Detect missing, repeated, late, or out-of-order chunks where applicable.
- Use WebSocket or the provider's supported streaming transport for incremental updates.
- Implement connection-open, connection-closing, and reconnect states explicitly.
- Do not silently restart a session in a way that loses transcript context.
- Expose chunking settings for experiments without requiring code edits.
Recognizer adapter
- Define a provider-neutral adapter for starting a session, sending audio, receiving hypotheses, and ending a session.
- Normalize provider events into a common internal event format.
- Preserve raw provider metadata needed for debugging and evaluation.
- Capture partial and final hypotheses separately.
- Record provider timestamps or sequence markers when available.
- Treat missing confidence values as missing rather than inventing confidence scores.
- Do not interpret sentence-level stability as token-level confidence.
- Handle provider errors, rate limits, session expiry, and unsupported audio formats.
- Keep credentials on the backend and never expose secret keys in browser code.
Provider selection and fallback
- Choose a real recognizer that supports incremental transcription suitable for the experiment.
- Verify whether it returns partial hypotheses, word timing, confidence, stability, and finalization signals.
- Do not assume every provider exposes all of those signals.
- Keep provider configuration in environment variables or server-side configuration.
- Provide a clearly labelled mock mode for UI development when credentials are absent.
- Make mock events deterministic so tests can reproduce revisions.
- Do not present mock output as actual speech recognition.
- Make fallback behavior visible in application status.
- Keep provider-specific limitations documented alongside the adapter.
Event model
- Define typed events for session start, audio chunk accepted, partial hypothesis, final hypothesis, warning, error, and session end.
- Define transcript events for provisional span creation, provisional span update, commitment, revision, and finalization.
- Include session ID and event timestamps in every relevant event.
- Include a monotonic event sequence number for debugging and replay.
- Validate incoming event shapes before processing them.
- Reject malformed events without crashing the entire session.
- Keep event processing deterministic for identical event sequences.
- Ensure event handlers do not mutate unrelated session state.
- Make event logs useful for reproducing difficult transcript cases.
Transcript representation
- Represent the current hypothesis as ordered spans rather than a single mutable text string.
- Each span should include text, state, timing when available, and confidence metadata when available.
- Use the states provisional, committed, revised, and finalized where appropriate.
- Keep committed status distinct from objectively correct status.
- Track revisions to provisional spans.
- Permit a span to contain multiple words when word-level alignment is unavailable or unreliable.
- Normalize whitespace and punctuation without changing the recognized lexical content unexpectedly.
- Preserve punctuation and capitalization from the recognizer when possible.
- Support rebuilding the displayed transcript from the current ordered span state.
Hypothesis alignment
- Compare new hypotheses against the current hypothesis to identify stable and changed regions.
- Use timestamps, provider token IDs, or alignment metadata when available.
- When such metadata is absent, use a tested text-alignment strategy.
- Do not align solely by fixed word index because insertions and deletions shift indexes.
- Handle repeated words such as 'that that' without collapsing valid repetitions.
- Handle punctuation-only changes separately from lexical substitutions where useful.
- Limit revisions to the affected provisional region when alignment is reliable.
- If alignment is ambiguous, prefer a conservative span-level update over corrupting stable text.
- Log alignment uncertainty for later evaluation.
Selective commitment policy
- Implement the commitment policy as a separate module with configurable parameters.
- Accept the latest hypothesis, recent hypothesis history, and available model signals as inputs.
- Return a decision for each eligible word or span without directly rendering UI.
- Support commit, remain provisional, and revise decisions.
- Do not block later audio processing merely because an earlier span is uncertain.
- Allow provisional spans to be updated when new evidence changes the recognizer hypothesis.
- Prevent routine updates from silently changing committed text.
- Define an explicit and testable end-of-utterance finalization rule.
- Expose policy decisions and reasons to the metrics and debugging layer.
Baseline policies
- Implement a raw-streaming baseline that displays provider partial hypotheses without the selective policy.
- Implement a confidence-threshold baseline only when meaningful confidence signals exist.
- Implement a fixed-delay or fixed-update-count baseline for comparison.
- Implement the proposed selective commitment policy using confidence and hypothesis stability when available.
- Use the same recognizer and audio inputs across comparable policies.
- Keep baseline behavior selectable through configuration.
- Do not make baseline results artificially weak through different audio or model settings.
- Record the parameters used for every run.
- Use baseline comparisons to determine whether policy complexity provides measurable value.
Confidence handling
- Use confidence only according to the semantics documented by the chosen provider or model.
- Distinguish token confidence, word confidence, utterance confidence, and provider stability metadata.
- Do not treat a raw logit or probability as calibrated correctness probability without validation.
- Allow the policy to operate without confidence by using hypothesis stability and elapsed context.
- Make confidence thresholds configurable.
- Evaluate threshold choices on a validation set rather than the final test set.
- Record missing confidence values and how the policy handles them.
- Consider confidence calibration as a separate research question rather than an assumed feature.
- Ensure confidence is never displayed as a guarantee that a word is correct.
Hypothesis stability
- Track whether a span remains unchanged across successive partial hypotheses.
- Make the number of required agreeing updates configurable.
- Do not assume consecutive updates are independent evidence.
- Account for providers that emit duplicate partial results.
- Ignore duplicate events when calculating update-based stability.
- Track elapsed audio context as well as update count where possible.
- Permit a span to remain provisional if evidence remains insufficient.
- Prevent a span from remaining provisional forever without a defined timeout or finalization behavior.
- Measure how stability settings affect latency, accuracy, and revisions.
Commitment deadlines
- Support an optional maximum provisional duration for a span.
- Treat the deadline as a latency constraint, not proof of correctness.
- When the deadline expires, apply a documented fallback decision.
- Allow fallback behavior to remain provisional or commit with an uncertainty marker, depending on configuration.
- Never silently fabricate a more confident hypothesis when the deadline expires.
- Record deadline-triggered decisions separately from evidence-triggered commitments.
- Make deadline behavior deterministic under a controlled event timeline.
- Test long pauses and slow speech to ensure timers do not create false commitments.
- Compare deadline settings experimentally rather than selecting arbitrary values as optimal.
Word and span granularity
- Support word-level commitment when the recognizer provides reliable word alignment.
- Support span-level commitment when word alignment is unavailable or unstable.
- Keep granularity configurable for experiments.
- Do not split languages or scripts into words using English-only assumptions.
- Handle punctuation attached to words consistently.
- Consider whether a word boundary is trustworthy before committing an individual word.
- Keep a span provisional if its internal alignment cannot be trusted.
- Compare word-level and span-level strategies in evaluation.
- Document the trade-offs between responsiveness and alignment complexity.
Transcript UI
- Create a clean live transcript view with readable typography and sufficient spacing.
- Use a subtle, accessible visual distinction for provisional text.
- Use a separate style for committed text without implying perfect correctness.
- Provide a concise legend explaining the transcript states.
- Keep the newest text visible while preserving a way to inspect earlier transcript content.
- Do not make the transcript jump or scroll aggressively on every token update.
- Animate only when it improves comprehension and does not obscure rapid updates.
- Provide a clear indication when the application is listening, reconnecting, processing, or stopped.
- Ensure the interface remains usable on desktop and narrow screens.
Transcript revision UX
- Update changed provisional regions without unnecessarily replacing the whole transcript view.
- Optionally show a brief visual revision indicator for changed text.
- Do not use alarming colors for ordinary recognition corrections.
- Do not hide meaningful revisions from the user.
- Keep a bounded revision history for debugging or optional inspection.
- Distinguish a revised provisional span from a previously committed span.
- If a committed span must be rolled back due to provider finalization, record the exceptional event.
- Make rollback behavior explicit and testable.
- Prevent stale asynchronous events from overwriting a newer transcript state.
Session controls and states
- Model the session lifecycle with explicit idle, requesting-permission, connecting, listening, stopping, completed, and error states.
- Disable incompatible controls while a session is transitioning.
- Allow the user to stop capture promptly.
- Send an orderly end-of-audio signal when the provider requires it.
- Wait for a bounded finalization period after capture stops.
- Handle a user starting a new session before an old session has fully closed.
- Isolate state and events by session ID.
- Clear session-specific UI state when starting a fresh session while preserving optional prior history.
- Never allow late events from an old session to contaminate the active session.
Backend API
- Create clear endpoints or WebSocket message contracts for session setup and streaming.
- Validate all incoming session settings and audio metadata.
- Use a session identifier to associate audio and transcript events.
- Apply message-size and session-duration limits.
- Return structured errors with safe, actionable descriptions.
- Do not expose stack traces, secret values, or provider credentials to clients.
- Use asynchronous I/O where appropriate for concurrent sessions.
- Add health and readiness checks for deployment and diagnostics.
- Keep the API small and focused on streaming transcription and evaluation.
Security and secrets
- Load cloud credentials from environment variables or an approved secret-management mechanism.
- Never hardcode credentials in source files or frontend configuration.
- Do not log API keys, authorization headers, or raw secrets.
- Validate origins and authentication settings appropriate to the deployment context.
- Apply rate limits and session limits to prevent accidental resource exhaustion.
- Use encrypted transport for remote connections.
- Restrict cloud permissions to the minimum required services and actions.
- Ensure logs avoid unnecessary personal information.
- Document safe handling of environment files and local credentials.
Privacy and audio retention
- Request clear permission before capturing microphone audio.
- Make it clear when audio capture is active.
- Do not retain raw audio by default after processing.
- Keep transcript persistence optional and transparent.
- Do not send audio to an unconfigured external service.
- Explain which recognition provider processes audio in the running application.
- Use consented or public datasets for evaluation.
- Provide a clear way to end capture and release microphone resources.
- Minimize sensitive content in telemetry and error reports.
Metrics and instrumentation
- Record the time the audio chunk was captured and the time it was submitted.
- Record the arrival time of each partial and final hypothesis.
- Record the first time each span was displayed.
- Record the time each span was committed.
- Count revisions to provisional spans.
- Count any post-commit changes separately.
- Record final transcript text or evaluation references only when privacy settings permit.
- Measure backend processing time separately from network and UI delay where possible.
- Use a monotonic clock for elapsed-time measurements.
Evaluation definitions
- Use Word Error Rate as a principal measure of final transcript accuracy.
- Compute WER using a documented text normalization procedure.
- Keep normalization consistent across every policy and baseline.
- Measure time to first displayed word.
- Measure per-word first-display latency using reference timing or a documented alignment method.
- Measure commitment latency separately from first-display latency.
- Measure visible revisions per utterance and the fraction of displayed words later changed.
- Report median and high-percentile latency, not just average latency.
- Report compute usage and cloud cost when those measurements are available.
Experimental design
- Use the same recordings for every policy comparison.
- Separate training or tuning data, validation data, and held-out test data when applicable.
- Choose thresholds using validation data only.
- Keep the held-out test set untouched until policy choices are finalized.
- Include clean speech and more challenging audio conditions.
- Document language, speaker diversity, recording setup, and dataset limitations.
- Run ablations to isolate the effect of confidence, stability, and deadline rules.
- Report results by condition when aggregate metrics conceal important failures.
- Do not invent benchmark results or claim improvement before experiments are run.
Failure cases
- Test fast speech and slow speech.
- Test long pauses within a sentence.
- Test repeated words and self-corrections by the speaker.
- Test names, acronyms, and domain-specific terminology.
- Test low-volume speech and background noise.
- Test network delays, dropped connections, and duplicate events.
- Test provider revisions that insert or delete words earlier in the hypothesis.
- Test missing confidence data and inconsistent timing metadata.
- Test the finalization path when the provider's final transcript differs from its last partial transcript.
Testing strategy
- Write unit tests for transcript alignment and state transitions.
- Write deterministic tests for each commitment policy.
- Test that provisional spans can be revised.
- Test that committed spans are not casually overwritten.
- Test that later words can appear while an earlier span remains provisional.
- Test end-of-utterance finalization with and without a provider final event.
- Test duplicate and out-of-order event handling.
- Test session isolation and stale-event rejection.
- Add integration tests for the recognizer adapter using a mock provider and, when available, a real provider.
Performance and resource control
- Bound memory used for audio buffers, event histories, and revision histories.
- Bound the number of concurrent sessions according to available resources.
- Use backpressure when audio arrives faster than it can be processed.
- Do not drop audio silently when backpressure occurs.
- Expose safe configuration for chunk size, buffer limits, and policy thresholds.
- Measure CPU, memory, and network overhead for the policy layer.
- Keep transcript rendering efficient during frequent updates.
- Avoid rerendering unrelated UI components for every token event.
- Make overload behavior explicit and observable.
AWS deployment
- Keep AWS as the intended deployment environment while allowing local development.
- Select the compute service only after determining model runtime, memory, and latency needs.
- Consider a managed streaming speech service if its metadata supports the required policy.
- Consider a self-hosted model when control over incremental hypotheses is necessary.
- Keep provider credentials and cloud configuration server-side.
- Use secure transport for browser-to-backend and backend-to-provider connections.
- Add health checks and structured operational logs.
- Measure end-to-end latency in the deployed environment rather than extrapolating from local tests.
- Estimate cost per audio minute from actual usage and the selected provider's pricing.
Configuration and reproducibility
- Store policy settings in a typed configuration object.
- Allow selection of baseline and proposed policies through configuration.
- Record the configuration used for each evaluation run.
- Use fixed random seeds where applicable to reproducible components.
- Version the commitment policy and evaluation methodology.
- Keep test dataset identifiers separate from personal transcript content.
- Make experiment runs repeatable with a documented command or script.
- Do not mix development defaults with production secrets.
- Validate configuration values at startup and report invalid settings clearly.
Accessibility and usability
- Ensure provisional and committed text are distinguishable without relying on color alone.
- Use adequate contrast and readable font sizes.
- Provide keyboard-accessible session controls.
- Expose listening and connection state to assistive technologies.
- Respect reduced-motion preferences.
- Make microphone permission errors understandable.
- Do not rely only on animation to communicate a transcript revision.
- Keep controls usable on touch devices.
- Test the application with long transcripts and narrow viewports.
Error handling and recovery
- Handle microphone initialization failures without leaving the UI stuck.
- Handle recognizer connection failures with a visible status and safe recovery path.
- Use bounded retries with backoff where retrying is appropriate.
- Do not replay audio chunks blindly if the provider may already have processed them.
- Distinguish recoverable transport errors from unrecoverable session errors.
- Preserve the last valid transcript state after a recoverable error.
- Make the user aware if any audio was not processed.
- Close stale sockets and release server-side session resources.
- Log diagnostic context without including secrets or unnecessary raw audio.
Code quality and maintainability
- Use clear names for hypotheses, spans, commitment decisions, and final transcripts.
- Keep policy logic free of direct DOM or framework dependencies.
- Use strict types for event payloads and transcript state.
- Separate pure transformation functions from I/O.
- Keep provider-specific assumptions inside adapters.
- Document non-obvious alignment and commitment decisions.
- Prefer small modules with focused responsibilities.
- Remove dead code and placeholder behavior before considering the prototype complete.
- Keep dependency versions controlled and avoid unnecessary libraries.
Research integrity
- Describe the system as an implementation of a proposed selective commitment policy.
- Do not claim that streaming transcription itself is novel.
- Review prior work on partial hypotheses, adaptive emission, confidence calibration, and transcript stability.
- Identify the closest methods before claiming a research contribution.
- Compare against strong, fair baselines using identical audio.
- Report negative or mixed results honestly.
- Separate measured results from hypotheses and planned work.
- Do not tune on the test set or cherry-pick only favorable examples.
- Make claims proportional to the evidence collected.
Implementation sequence
- First verify the environment and establish the chosen framework and provider constraints.
- Build a minimal streaming recognizer connection and confirm that partial hypotheses arrive incrementally.
- Display raw partial hypotheses before adding the selective commitment layer.
- Create the typed transcript model and deterministic event replay harness.
- Implement alignment and revision tracking with unit tests.
- Implement the simplest confidence-and-stability policy supported by available signals.
- Add the raw-streaming and fixed-delay baselines for comparison.
- Instrument latency, revisions, final WER, and resource use.
- Only then refine the policy based on reproducible evaluation evidence.
Definition of functional completeness
- The application must capture live microphone speech and receive genuine incremental recognition events in real mode.
- The transcript must visibly distinguish provisional from committed regions.
- The system must continue processing new speech while an earlier region remains provisional.
- Provisional regions must be revisable when new hypotheses arrive.
- Commitment behavior must be controlled by a testable policy rather than arbitrary UI timing.
- The final transcript must reconcile with the recognizer's final result.
- Metrics must be recorded consistently enough to compare policies.
- Provider errors and microphone errors must not crash the whole application.
- The implementation must remain explicit about limitations and avoid fabricated confidence or evaluation results.
Research literature and prior-art review
- Review recent papers on streaming automatic speech recognition and incremental decoding before selecting the final commitment policy.
- Search specifically for adaptive token emission, partial-hypothesis stability, word-level commitment, and confidence-aware streaming.
- Record the core method, assumptions, latency metrics, and limitations of each closest related approach.
- Compare the proposed policy with the closest prior methods rather than only with a naive baseline.
- Separate implementation novelty from evaluation novelty and engineering contribution.
- Do not describe a known method as new merely because it is implemented in this application.
- Keep references and paper notes in a maintainable research-notes file.
- Use prior-art findings to narrow the hypothesis before committing to a complex design.
- Update the novelty assessment whenever a close method is discovered.
Data and benchmark management
- Choose a speech corpus appropriate to the initial language and domain.
- Check dataset licensing and usage conditions before downloading or redistributing recordings.
- Keep training, validation, and test splits separate where the experiment requires them.
- Store dataset manifests and stable sample identifiers rather than copying unnecessary raw audio.
- Document sampling rate, channel count, duration, speaker metadata, and any filtering applied.
- Ensure every baseline receives the same test utterances and equivalent audio preprocessing.
- Use a representative mix of short utterances, long utterances, pauses, and difficult vocabulary.
- Do not tune the commitment policy on the final test split.
- Record exclusions and explain why any sample was removed from evaluation.
Text normalization and scoring
- Define how case, punctuation, numbers, contractions, and whitespace are normalized before WER calculation.
- Use the same normalization implementation for reference and hypothesis text.
- Preserve a raw transcript for inspection alongside any normalized scoring form.
- Handle empty reference or hypothesis strings explicitly in scoring code.
- Test the scorer with known substitutions, insertions, and deletions.
- Document tokenization rules for languages where whitespace does not reliably define words.
- Do not normalize away meaningful distinctions that would conceal recognition errors.
- Keep live transcript display normalization separate from offline evaluation normalization.
- Include scorer version and normalization configuration in experiment records.
Concurrency and session isolation
- Give each active transcription session independent buffers, transcript state, and policy history.
- Prevent concurrent sessions from sharing a mutable hypothesis tracker.
- Associate every incoming event with its originating session and reject stale session events.
- Apply backpressure and concurrency limits at the backend boundary.
- Clean up server-side state when a client disconnects or a session expires.
- Make cancellation safe even when a provider operation is still awaiting a response.
- Test multiple simultaneous sessions if the chosen deployment is intended to support them.
- Keep session identifiers non-secret and avoid placing sensitive transcript content in them.
- Ensure one session's error does not stop unrelated sessions.
Observability and debugging tools
- Provide a development-only view of incoming partial hypotheses and commitment decisions.
- Show why a span was committed, retained provisionally, or revised using available evidence.
- Include policy version, threshold settings, and event sequence in diagnostic records.
- Allow deterministic replay of recorded event sequences without microphone input.
- Keep debug instrumentation disabled or minimized in production where it could expose sensitive content.
- Differentiate provider errors from alignment errors and policy errors.
- Expose basic session health and event-processing delay.
- Make diagnostics useful without requiring developers to inspect raw audio by default.
- Use debug traces to understand failures rather than to cherry-pick favorable demonstrations.
Deployment configuration and operations
- Keep local development and deployed environment configuration separate.
- Use a documented configuration mechanism for provider selection and endpoint URLs.
- Validate required environment variables at startup without printing their secret values.
- Provide a health check that verifies application readiness without exposing sensitive information.
- Handle graceful shutdown by stopping new sessions and cleaning up active resources.
- Pin compatible dependency versions and review security updates before deployment.
- Keep AWS permissions limited to the services and actions the application actually uses.
- Track resource use and costs during realistic streaming sessions.
- Document how to rotate credentials and shut down billable resources.
Cost-aware engineering
- Estimate the expected cost of recognition, compute, network transfer, and logging for typical sessions.
- Measure actual cost from the selected service configuration instead of relying on an unverified estimate.
- Do not assume that selective commitment reduces recognition-provider charges if the same audio is still processed.
- Measure the additional compute cost of alignment and policy decisions.
- Compare local or self-hosted inference only when it is technically and economically relevant.
- Keep costly debug logging and redundant audio copies out of the normal path.
- Expose safeguards for session duration and concurrent sessions.
- Report cost alongside latency and accuracy so one metric is not optimized in isolation.
- Document the cost assumptions used in any project claims.
Frontend state management
- Keep active recording state, connection state, transcript spans, and experiment settings in explicit state containers.
- Use a reducer or similarly predictable state transition mechanism for transcript events.
- Make transcript updates idempotent where possible.
- Prevent rapid partial events from creating an unbounded render queue.
- Keep audio capture logic separate from transcript rendering.
- Ensure UI components do not independently invent commitment decisions.
- Represent loading, listening, reconnecting, completed, and error states explicitly.
- Preserve keyboard focus when transcript text updates.
- Test state transitions independently of visual styling.
Backend lifecycle and resource cleanup
- Allocate provider resources only after a valid session request.
- Associate provider sessions with client session IDs and enforce a maximum lifetime.
- Close provider streams when the client stops or disconnects.
- Flush any required final audio before requesting final transcription.
- Wait only a bounded period for final recognition before reporting a timeout.
- Release buffers, tasks, and temporary files when sessions end.
- Make cleanup safe to call more than once.
- Record cleanup failures without leaking provider credentials or raw sensitive content.
- Test cleanup after normal completion, cancellation, timeout, and unexpected exceptions.
Acceptance behavior and edge-case consistency
- Verify that later recognized words can be displayed while an earlier word remains provisional.
- Verify that provisional words can change when new partial hypotheses arrive.
- Verify that commitment is determined by the configured policy and not by a hard-coded animation delay.
- Verify that missing confidence values do not crash or silently become high confidence.
- Verify that a final hypothesis can correct a provisional span before the session is complete.
- Verify that duplicate partial or final events do not duplicate transcript text.
- Verify that stale events from a previous session are ignored.
- Verify that all displayed text is traceable to a recognizer event or clearly labelled mock event.
- Verify that finalization and cleanup happen even when the user stops capture unexpectedly.
Future extension boundaries
- Keep multilingual support as a later extension unless the initial research question specifically requires it.
- Keep speaker adaptation out of the first version unless it is essential to the defined contribution.
- Do not add translation, summarization, sentiment analysis, or chatbot features to the core pipeline.
- Design the adapter boundary so future recognizers can be evaluated without changing the UI.
- Keep alternative-hypothesis tracking optional until a simple policy has been measured.
- Add domain-specific vocabulary only with a clear evaluation plan and fair comparison.
- Do not add a second recognition model solely to make the architecture look more advanced.
- Require each extension to have a measurable reason and a controlled experiment.
- Preserve the project's core focus on selective commitment and streaming transcript behavior.
Final engineering discipline
- Build the smallest end-to-end path first and validate it before adding optional features.
- Prefer a working real-time pipeline over a visually polished mock that does not process real speech.
- Keep all user-visible status messages truthful about whether the system is using real or mock recognition.
- Use typed interfaces and explicit error paths for external service boundaries.
- Run automated tests after changing transcript alignment or commitment behavior.
- Inspect representative transcript traces whenever policy behavior changes.
- Keep implementation decisions aligned with the measured research question.
- Document known limitations instead of hiding them behind optimistic defaults.
- Treat the system as an experiment whose results may show that a proposed policy is not beneficial.
Release readiness
- Verify the application from a clean install using the documented dependency setup.
- Run the full automated test suite before deploying a new version.
- Confirm that microphone permissions, streaming transport, and provider configuration work in the target browser.
- Confirm that no secret keys or private recordings are included in source control.
- Check the deployed application for console errors and backend exceptions during a real session.
- Measure end-to-end latency with representative audio after deployment.
- Verify that the application can stop a session and release resources reliably.
- Review the privacy behavior and ensure audio retention matches the documented configuration.
- Reassess the research claims against the final measured results before presenting the project.