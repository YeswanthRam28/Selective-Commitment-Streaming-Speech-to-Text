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
  const [provider, setProvider] = useState<'mock' | 'aws'>('mock');
  const [config, setConfig] = useState<PolicyConfig>(DEFAULT_CONFIG);
  const [isListening, setIsListening] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);
  const [spans, setSpans] = useState<TranscriptSpan[]>([]);
  const [updateCount, setUpdateCount] = useState(0);
  const [replayResult, setReplayResult] = useState<any | null>(null);
  const [serverHealth, setServerHealth] = useState<{ status: string; aws_streaming_available: boolean } | null>(null);

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
    <div className="app-container">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="nav-brand">
          <span className="brand-logo">??</span>
          <div>
            <h1 className="brand-title">Selective Commitment ASR</h1>
            <p className="brand-subtitle">Real-time Streaming Hypothesis Policy Engine</p>
          </div>
        </div>

        <div className="nav-badges">
          <span className={`status-pill ${serverHealth ? 'status-online' : 'status-offline'}`}>
            {serverHealth ? 'Backend Online' : 'Backend Disconnected'}
          </span>
          <span className="provider-pill">
            {provider === 'aws'
              ? serverHealth?.aws_streaming_available
                ? 'AWS Transcribe Ready'
                : 'AWS (Fallback to Mock)'
              : 'Deterministic Mock Mode'}
          </span>
        </div>
      </header>

      {/* Main Content Layout */}
      <main className="main-content">
        {/* Controls Bar */}
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

      {/* Footer / Architecture notes */}
      <footer className="app-footer">
        <p>
          Selective Commitment Policy decouples provisional hypothesis tokens from committed text,
          isolating revisions and eliminating output flicker during streaming recognition.
        </p>
      </footer>
    </div>
  );
};

export default App;
