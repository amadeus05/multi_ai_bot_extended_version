create table if not exists order_intents (
  client_order_id text primary key,
  symbol text not null,
  side text not null,
  status text not null,
  reason text,
  order_id text,
  reject_reason text,
  ts timestamptz,
  updated_at timestamptz not null default now()
);

create index if not exists idx_order_intents_status
on order_intents(status, updated_at);
