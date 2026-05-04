from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupabaseConnection:
    url: str
    service_key: str
    schema: str = "public"

    def validate_service_key(self) -> None:
        if not self.url or not self.service_key:
            raise ValueError("Supabase storage requires SUPABASE_URL and SUPABASE_SERVICE_KEY.")
        if self.service_key.startswith("sb_publishable_"):
            raise ValueError(
                "SUPABASE_SERVICE_KEY contains a publishable/anon key. "
                "Use a Supabase secret/service-role key for backend storage writes."
            )

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
