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

create table if not exists orders (
  id bigserial primary key,
  session_id text not null,
  sequence bigint not null,
  order_id text,
  client_order_id text,
  symbol text not null,
  side text not null,
  order_type text not null,
  status text not null,
  reason text,
  amount double precision not null,
  price double precision,
  created_ts timestamptz not null,
  updated_ts timestamptz not null,
  source text not null,
  meta_json jsonb not null default '{}'::jsonb,
  unique(session_id, order_id),
  unique(session_id, client_order_id)
);

create index if not exists idx_orders_session_ts
on orders(session_id, updated_ts);

create index if not exists idx_orders_symbol_status
on orders(symbol, status);

create table if not exists fills (
  id bigserial primary key,
  session_id text not null,
  sequence bigint not null,
  fill_id text not null,
  order_id text not null,
  client_order_id text,
  symbol text not null,
  side text not null,
  amount double precision not null,
  price double precision not null,
  fee double precision not null,
  ts timestamptz not null,
  command_reason text,
  meta_json jsonb not null default '{}'::jsonb,
  unique(session_id, fill_id)
);

create index if not exists idx_fills_session_ts
on fills(session_id, ts);

create index if not exists idx_fills_order
on fills(session_id, order_id);

create table if not exists active_trade_lots (
  id bigserial primary key,
  session_id text not null,
  symbol text not null,
  direction text not null,
  remaining_qty double precision not null,
  entry_price double precision not null,
  entry_ts timestamptz not null,
  entry_order_id text not null,
  entry_fill_id text not null,
  entry_fee_remaining double precision not null,
  entry_notional_remaining double precision not null,
  meta_json jsonb not null default '{}'::jsonb
);

create index if not exists idx_active_trade_lots_session_symbol
on active_trade_lots(session_id, symbol, id);

create table if not exists closed_trades (
  id bigserial primary key,
  trade_id text not null,
  session_id text not null,
  symbol text not null,
  direction text not null,
  entry_order_id text not null,
  exit_order_id text not null,
  entry_fill_id text not null,
  exit_fill_id text not null,
  entry_time timestamptz not null,
  exit_time timestamptz not null,
  entry_price double precision not null,
  exit_price double precision not null,
  qty double precision not null,
  notional double precision not null,
  pnl_abs_gross double precision not null,
  pnl double precision not null,
  pnl_pct double precision not null,
  fee double precision not null,
  exit_reason text,
  duration_minutes double precision not null,
  p_long double precision,
  p_short double precision,
  signal_gap double precision,
  risk_pct double precision,
  meta_json jsonb not null default '{}'::jsonb,
  unique(session_id, trade_id)
);

create index if not exists idx_closed_trades_session_exit
on closed_trades(session_id, exit_time);

create index if not exists idx_closed_trades_symbol
on closed_trades(symbol, exit_time);
