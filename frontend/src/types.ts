export type SpanState = 'provisional' | 'committed' | 'revised' | 'finalized';

export type PolicyMode = 'selective_commitment' | 'raw_streaming' | 'fixed_delay';

export interface TranscriptSpan {
  span_id: string;
  text: string;
  state: SpanState;
  start_time?: number;
  end_time?: number;
  confidence?: number;
  update_count: number;
  first_displayed_at: number;
  committed_at?: number;
  revised_count: number;
  history: string[];
}

export interface TranscriptUpdateEvent {
  session_id: string;
  seq: number;
  timestamp: number;
  spans: TranscriptSpan[];
  newly_committed_ids: string[];
  revised_ids: string[];
  full_text: string;
  is_final: boolean;
}

export interface PolicyConfig {
  mode: PolicyMode;
  required_stability_updates: number;
  confidence_threshold?: number;
  max_provisional_duration_sec: number;
  fixed_delay_updates: number;
}
