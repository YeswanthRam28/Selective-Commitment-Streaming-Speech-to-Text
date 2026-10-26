import React from 'react';
import type { PolicyConfig, PolicyMode } from '../types';

interface ControlsProps {
  isListening: boolean;
  provider: 'whisper' | 'deep_learning' | 'mock' | 'aws';
  config: PolicyConfig;
  onToggleProvider: (p: 'whisper' | 'deep_learning' | 'mock' | 'aws') => void;
  onUpdateConfig: (cfg: Partial<PolicyConfig>) => void;
  onStartStreaming: () => void;
  onStopStreaming: () => void;
  onRunReplay: () => void;
  onClear: () => void;
  isReplaying: boolean;
}

export const Controls: React.FC<ControlsProps> = ({
  isListening,
  provider,
  config,
  onToggleProvider,
  onUpdateConfig,
  onStartStreaming,
  onStopStreaming,
  onRunReplay,
  onClear,
  isReplaying,
}) => {
  return (
    <div className="studio-card controls-card">
      <div className="controls-grid">
        {/* Policy Selector */}
        <div className="control-group">
          <label className="control-label" htmlFor="policy-select">Commitment Policy</label>
          <select
            id="policy-select"
            className="control-select"
            value={config.mode}
            disabled={isListening}
            onChange={(e) => onUpdateConfig({ mode: e.target.value as PolicyMode })}
          >
            <option value="selective_commitment">Selective Commitment (Proposed)</option>
            <option value="raw_streaming">Raw Streaming (Immediate Partial Baseline)</option>
            <option value="fixed_delay">Fixed Delay (N-lag Baseline)</option>
          </select>
        </div>

        {/* Recognizer Provider */}
        <div className="control-group">
          <label className="control-label" htmlFor="provider-select">Recognizer Adapter</label>
          <select
            id="provider-select"
            className="control-select"
            value={provider}
            disabled={isListening}
            onChange={(e) => onToggleProvider(e.target.value as 'whisper' | 'deep_learning' | 'aws' | 'mock')}
          >
            <option value="whisper">OpenAI Whisper (On-Device GPU)</option>
            <option value="deep_learning">CTC Acoustic Model Wav2Vec2 (GPU)</option>
            <option value="mock">Deterministic Synthetic Mock</option>
            <option value="aws">AWS Transcribe Streaming</option>
          </select>
        </div>

        {/* Stability Threshold (Selective Commitment) */}
        {config.mode === 'selective_commitment' && (
          <div className="control-group">
            <label className="control-label" htmlFor="stability-input">
              Stability Threshold: {config.required_stability_updates} updates
            </label>
            <input
              id="stability-input"
              type="range"
              min="1"
              max="5"
              step="1"
              className="control-slider"
              value={config.required_stability_updates}
              disabled={isListening}
              onChange={(e) => onUpdateConfig({ required_stability_updates: parseInt(e.target.value) })}
            />
          </div>
        )}

        {/* Max Provisional Deadline */}
        {config.mode === 'selective_commitment' && (
          <div className="control-group">
            <label className="control-label" htmlFor="deadline-input">
              Commitment Deadline: {config.max_provisional_duration_sec}s
            </label>
            <input
              id="deadline-input"
              type="range"
              min="0.5"
              max="6.0"
              step="0.5"
              className="control-slider"
              value={config.max_provisional_duration_sec}
              disabled={isListening}
              onChange={(e) => onUpdateConfig({ max_provisional_duration_sec: parseFloat(e.target.value) })}
            />
          </div>
        )}

        {/* Fixed Lag (Fixed Delay Mode) */}
        {config.mode === 'fixed_delay' && (
          <div className="control-group">
            <label className="control-label" htmlFor="lag-input">
              Lag Frontier: {config.fixed_delay_updates} tokens
            </label>
            <input
              id="lag-input"
              type="range"
              min="1"
              max="6"
              step="1"
              className="control-slider"
              value={config.fixed_delay_updates}
              disabled={isListening}
              onChange={(e) => onUpdateConfig({ fixed_delay_updates: parseInt(e.target.value) })}
            />
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="controls-actions">
        {!isListening ? (
          <button
            id="start-capture-btn"
            className="btn btn-primary"
            onClick={onStartStreaming}
            disabled={isReplaying}
          >
            <span className="btn-icon">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
            </span>
            Start Streaming
          </button>
        ) : (
          <button
            id="stop-capture-btn"
            className="btn btn-danger"
            onClick={onStopStreaming}
          >
            <span className="btn-icon">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"></rect></svg>
            </span>
            Stop Streaming
          </button>
        )}

        <button
          id="replay-test-btn"
          className="btn btn-secondary"
          onClick={onRunReplay}
          disabled={isListening || isReplaying}
        >
          <span className="btn-icon">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
          </span>
          Run Replay Evaluation
        </button>

        <button
          id="clear-btn"
          className="btn btn-outline"
          onClick={onClear}
          disabled={isListening || isReplaying}
        >
          Clear
        </button>
      </div>
    </div>
  );
};
