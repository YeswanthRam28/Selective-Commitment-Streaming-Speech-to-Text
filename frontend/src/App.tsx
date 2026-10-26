import React, { useState, useEffect, useRef } from 'react';
import type { PolicyConfig, TranscriptSpan, TranscriptUpdateEvent } from './types';
import { AudioRecorder } from './audio/recorder';
import { TranscriptView } from './components/TranscriptView';
import { Controls } from './components/Controls';
import { MetricsPanel } from './components/MetricsPanel';

const DEFAULT_CONFIG: PolicyConfig = {
  mode: 'selective_commitment',
  required_stability_updates: 2,
  max_provisional_duration_sec: 2.5,
  fixed_delay_updates: 3,
};

export const App: React.FC = () => {
  const [provider, setProvider] = useState<'deep_learning' | 'mock' | 'aws'>('deep_learning');
  const [config, setConfig] = useState<PolicyConfig>(DEFAULT_CONFIG);
  const [isListening, setIsListening] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);
  const [spans, setSpans] = useState<TranscriptSpan[]>([]);
  const [updateCount, setUpdateCount] = useState(0);
  const [replayResult, setReplayResult] = useState<any | null>(null);
  const [serverHealth, setServerHealth] = useState<{
    status: string;
    deep_learning_available: boolean;
    cuda_available: boolean;
    gpu_name?: string;
    aws_streaming_available: boolean;
  } | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);

  useEffect(() => {
    // Check backend health
    fetch('http://localhost:8000/health')
      .then((res) => res.json())
      .then((data) => setServerHealth(data))
      .catch(() => setServerHealth(null));
  }, []);

  const handleUpdateConfig = (newConfig: Partial<PolicyConfig>) => {
    setConfig((prev) => ({ ...prev, ...newConfig }));
  };

  const startStreaming = async () => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws/transcribe');
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            action: 'start',
            provider,
            policy_config: config,
          })
        );

        // Initialize mic audio capture
        const recorder = new AudioRecorder((pcmData) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(pcmData);
          }
        });

        recorder
          .start()
          .then(() => {
            recorderRef.current = recorder;
            setIsListening(true);
          })
          .catch((err) => {
            console.error('Microphone error:', err);
            alert(`Microphone access error: ${err.message || err}`);
            stopStreaming();
          });
      };

      ws.onmessage = (event) => {
        try {
          const data: TranscriptUpdateEvent = JSON.parse(event.data);
          if (data.spans) {
            setSpans(data.spans);
            setUpdateCount((c) => c + 1);
          }
        } catch {
          // ignore non-json frames
        }
      };

      ws.onclose = () => {
        setIsListening(false);
      };

      ws.onerror = (e) => {
        console.error('WebSocket error:', e);
        stopStreaming();
      };
    } catch (err) {
      console.error('Failed to start streaming:', err);
    }
  };

  const stopStreaming = () => {
    if (recorderRef.current) {
      recorderRef.current.stop();
      recorderRef.current = null;
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsListening(false);
  };

  const runReplayEvaluation = async () => {
    setIsReplaying(true);
    try {
      const res = await fetch('http://localhost:8000/api/replay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ policy_config: config }),
      });
      const data = await res.json();
      setReplayResult(data);
    } catch (err) {
      alert('Failed to connect to backend evaluation API. Is backend running on port 8000?');
    } finally {
      setIsReplaying(false);
    }
  };

  const clearTranscript = () => {
    setSpans([]);
    setUpdateCount(0);
    setReplayResult(null);
  };

  return (
    <>
      {/* Background doodles */}
      <div className="bg-doodle-left"></div>
      <div className="bg-doodle-right"></div>

      <div className="app-container">
        {/* Editorial Navbar */}
        <header className="navbar">
          <div className="nav-left">
            <button className="pill-btn" onClick={() => handleUpdateConfig({ mode: 'selective_commitment' })}>
              Selective Engine
            </button>
            <button className="pill-btn" onClick={() => handleUpdateConfig({ mode: 'raw_streaming' })}>
              Raw Baseline
            </button>
          </div>

          <a href="#" className="brand-center">
            TWO <span>26</span>
          </a>

          <div className="nav-right">
            <div className="nav-schedule">
              Streaming Audio<br />
              16kHz Mono PCM
            </div>
            {!isListening ? (
              <button
                className="cta-black-pill"
                onClick={startStreaming}
                disabled={isReplaying}
                title="Start streaming audio from microphone"
              >
                <span>Start Stream</span>
                <span className="cta-orange-circle">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><line x1="7" y1="17" x2="17" y2="7"></line><polyline points="7 7 17 7 17 17"></polyline></svg>
                </span>
              </button>
            ) : (
              <button
                className="cta-black-pill"
                style={{ background: '#ff4800' }}
                onClick={stopStreaming}
                title="Stop current streaming session"
              >
                <span>Stop Stream</span>
                <span className="cta-orange-circle" style={{ background: '#121212' }}>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"></rect></svg>
                </span>
              </button>
            )}
          </div>
        </header>

        {/* Hero Poster */}
        <section className="hero-poster">
          <div className="sticker-tag sticker-left">Selective Commitment</div>
          <div className="sticker-tag sticker-right">Zero Output Flicker</div>
          <h1 className="hero-giant-title">
            STREAMING<br />SUSPICIOUSLY GOOD
          </h1>
          <div className="hero-badges-row">
            <span className="status-badge-pill badge-online">
              {serverHealth?.cuda_available
                ? `CUDA Active (${serverHealth.gpu_name || 'RTX 3050'})`
                : (serverHealth ? 'Backend Online (CPU)' : 'Connecting to Backend...')}
            </span>
            <span className="status-badge-pill">
              {provider === 'deep_learning'
                ? 'Local Deep Learning Model'
                : (provider === 'aws' ? 'AWS Transcribe Stream' : 'Deterministic Mock Mode')}
            </span>
          </div>
        </section>

        {/* Main Content Layout */}
        <main className="main-content">
          {/* Controls Card */}
          <Controls
            isListening={isListening}
            provider={provider}
            config={config}
            onToggleProvider={setProvider}
            onUpdateConfig={handleUpdateConfig}
            onStartStreaming={startStreaming}
            onStopStreaming={stopStreaming}
            onRunReplay={runReplayEvaluation}
            onClear={clearTranscript}
            isReplaying={isReplaying}
          />

          {/* Live Transcript View */}
          <TranscriptView spans={spans} isListening={isListening} />

          {/* Real-time Metrics and Evaluation */}
          <MetricsPanel
            spans={spans}
            updateCount={updateCount}
            replayResult={replayResult}
          />
        </main>

        {/* Studio Footer */}
        <footer className="app-footer">
          <p>
            TWO 26 • Selective Commitment Speech Recognition Policy Engine. Decoupling provisional hypotheses from stable output.
          </p>
        </footer>
      </div>
    </>
  );
};

export default App;
