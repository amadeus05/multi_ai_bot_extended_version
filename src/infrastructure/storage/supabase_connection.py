from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupabaseConnection:
    url: str
    service_key: str
    schema: str = "public"

    def rest_url(self, table: str) -> str:
        return f"{self.url.rstrip('/')}/rest/v1/{table}"

    def headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Profile": self.schema,
            "Content-Profile": self.schema,
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers
