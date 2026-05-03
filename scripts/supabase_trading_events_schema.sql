create table if not exists trading_events (
  id bigserial primary key,
  event_id text not null,
  session_id text not null,
  sequence bigint not null,
  ts timestamptz not null,
  kind text not null,
  source text not null,
  payload_json jsonb not null,
  created_at timestamptz not null default now(),
  unique(session_id, sequence),
  unique(session_id, event_id)
);

create index if not exists idx_trading_events_session_seq
on trading_events(session_id, sequence);

create index if not exists idx_trading_events_kind
on trading_events(kind);
