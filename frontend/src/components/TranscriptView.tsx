import React from 'react';
import type { TranscriptSpan } from '../types';

interface TranscriptViewProps {
  spans: TranscriptSpan[];
  isListening: boolean;
}

export const TranscriptView: React.FC<TranscriptViewProps> = ({ spans, isListening }) => {
  return (
    <div className="studio-card transcript-card">
      <div className="transcript-header">
        <div className="transcript-title-group">
          <h2 className="card-title">Live Incremental Transcript</h2>
          {isListening && (
            <span className="live-pill">
              <span className="pulse-dot"></span> Listening
            </span>
          )}
        </div>

        {/* Accessibility Legend */}
        <div className="transcript-legend">
          <span className="legend-item">
            <span className="legend-dot dot-committed"></span> Committed (Stable)
          </span>
          <span className="legend-item">
            <span className="legend-dot dot-provisional"></span> Provisional (Editable)
          </span>
          <span className="legend-item">
            <span className="legend-dot dot-revised"></span> Revised (Corrected)
          </span>
        </div>
      </div>

      <div className="transcript-body" tabIndex={0} aria-label="Transcript content area">
        {spans.length === 0 ? (
          <p className="empty-prompt">
            {isListening
              ? 'Listening for speech input...'
              : 'Click "Start Streaming" to begin capturing audio or run the Replay Test.'}
          </p>
        ) : (
          <div className="transcript-tokens">
            {spans.map((span) => {
              const isCommitted = span.state === 'committed' || span.state === 'finalized';
              const isRevised = span.state === 'revised' || span.revised_count > 0;

              let stateClass = 'token-provisional';
              if (isCommitted) stateClass = 'token-committed';
              if (isRevised && !isCommitted) stateClass = 'token-revised';

              return (
                <span
                  key={span.span_id}
                  className={`transcript-token ${stateClass}`}
                  title={`State: ${span.state} | Updates: ${span.update_count} | Revisions: ${span.revised_count}${
                    span.history.length > 0 ? ` (Prev: ${span.history.join(', ')})` : ''
                  }${span.confidence ? ` | Conf: ${(span.confidence * 100).toFixed(0)}%` : ''}`}
                >
                  <span className="token-text">{span.text}</span>
                  {!isCommitted && (
                    <span className="token-badge">{span.update_count}u</span>
                  )}
                  {isRevised && (
                    <span className="token-rev-badge">?{span.revised_count}</span>
                  )}
                </span>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
