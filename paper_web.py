from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)

PROJECT_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "runners", PROJECT_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
sys.path[:0] = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "runners"), str(PROJECT_ROOT)]


def _load_run_paper_main():
    module_path = PROJECT_ROOT / "runners" / "run_paper.py"
    spec = importlib.util.spec_from_file_location("v333_run_paper", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load paper runner from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


run_paper_main = _load_run_paper_main()

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _task_status(task: asyncio.Task[None] | None) -> dict[str, Any]:
    if task is None:
        return {"running": False, "status": "not_started", "error": None}
    if not task.done():
        return {"running": True, "status": "running", "error": None}
    if task.cancelled():
        return {"running": False, "status": "cancelled", "error": None}

    exc = task.exception()
    if exc is None:
        return {"running": False, "status": "stopped", "error": None}
    return {
        "running": False,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _log_paper_task_result(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        logger.info("Paper trading task was cancelled")
        return

    exc = task.exception()
    if exc is None:
        logger.warning("Paper trading task stopped without an error")
        return

    logger.exception("Paper trading task failed", exc_info=exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started_at_monotonic = time.monotonic()
    app.state.started_at_utc = _utc_now()
    logger.info("FastAPI startup | service=light-paper-web")
    logger.info("Paper trading startup requested | runner=%s", PROJECT_ROOT / "runners" / "run_paper.py")
    app.state.paper_task = asyncio.create_task(run_paper_main(), name="paper-trading")
    app.state.paper_task.add_done_callback(_log_paper_task_result)
    logger.info("Paper trading task started | task_name=%s", app.state.paper_task.get_name())

    try:
        yield
    finally:
        task: asyncio.Task[None] = app.state.paper_task
        if not task.done():
            logger.info("Paper trading task stopping")
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        logger.info("FastAPI shutdown | service=light-paper-web")


app = FastAPI(title="Light Paper Trading", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "light-paper-web",
        "uptime": "/uptime",
    }


@app.get("/uptime")
async def uptime(request: Request) -> JSONResponse:
    now = _utc_now()
    started_at_utc: datetime = request.app.state.started_at_utc
    uptime_seconds = int(time.monotonic() - request.app.state.started_at_monotonic)
    paper_trading = _task_status(request.app.state.paper_task)
    status_code = 200 if paper_trading["running"] else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "service": "light-paper-web",
            "started_at": started_at_utc.isoformat(),
            "now": now.isoformat(),
            "uptime_seconds": uptime_seconds,
            "paper_trading": paper_trading,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("paper_web:app", host="0.0.0.0", port=5000, reload=False)
