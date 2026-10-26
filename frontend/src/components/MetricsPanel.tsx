import React from 'react';
import type { TranscriptSpan } from '../types';

interface ReplayResult {
  session_id: string;
  policy_mode: string;
  total_revisions: number;
  total_committed: number;
  final_transcript: string;
  wer_metrics: {
    wer: number;
    total_edits: number;
    ref_len: number;
    hyp_len: number;
  };
  updates_count: number;
}

interface MetricsPanelProps {
  spans: TranscriptSpan[];
  updateCount: number;
  replayResult: ReplayResult | null;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  spans,
  updateCount,
  replayResult,
}) => {
  const committedCount = spans.filter(
    (s) => s.state === 'committed' || s.state === 'finalized'
  ).length;
  const provisionalCount = spans.length - committedCount;
  const totalRevisions = spans.reduce((acc, s) => acc + s.revised_count, 0);

  return (
    <div className="studio-card metrics-card">
      <h3 className="card-subtitle">Real-time Policy Metrics</h3>
      <div className="metrics-grid">
        <div className="metric-box">
          <div className="metric-value text-cyan">{spans.length}</div>
          <div className="metric-label">Total Words</div>
        </div>

        <div className="metric-box">
          <div className="metric-value text-emerald">{committedCount}</div>
          <div className="metric-label">Committed</div>
        </div>

        <div className="metric-box">
          <div className="metric-value text-amber">{provisionalCount}</div>
          <div className="metric-label">Provisional</div>
        </div>

        <div className="metric-box">
          <div className="metric-value text-purple">{totalRevisions}</div>
          <div className="metric-label">Revisions (Churn)</div>
        </div>

        <div className="metric-box">
          <div className="metric-value">{updateCount}</div>
          <div className="metric-label">Hypothesis Updates</div>
        </div>
      </div>

      {/* Benchmark Replay Summary (if run) */}
      {replayResult && (
        <div className="benchmark-box">
          <div className="benchmark-title">
            <span>Offline Benchmark Results ({replayResult.policy_mode})</span>
            <span className="wer-badge">WER: {(replayResult.wer_metrics.wer * 100).toFixed(1)}%</span>
          </div>
          <div className="benchmark-details">
            <p><strong>Edits:</strong> {replayResult.wer_metrics.total_edits} / {replayResult.wer_metrics.ref_len} ref tokens</p>
            <p><strong>Total Revisions (Flicker):</strong> {replayResult.total_revisions}</p>
            <p><strong>Final Transcript:</strong> <em>"{replayResult.final_transcript}"</em></p>
          </div>
        </div>
      )}
    </div>
  );
};
