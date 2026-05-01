#!/usr/bin/env python3
"""
Собирает исходный код и текстовые конфиги проекта в один .txt с путями от корня и разделителями.
Запуск из корня проекта: python scripts/export_project_bundle.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Каталоги, которые не обходим
SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".eggs",
    }
)

# Только сразу под корнем проекта (не трогаем src/.../models/ с кодом)
SKIP_TOP_LEVEL_DIRS = frozenset({"models"})

# Расширения текстовых файлов (код + конфиги)
TEXT_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".pyw",
        ".toml",
        ".ini",
        ".cfg",
        ".yaml",
        ".yml",
        ".json",
        ".txt",
        ".md",
        ".rst",
        ".env",
        ".example",
        ".sh",
        ".bat",
        ".ps1",
        ".sql",
        ".csv",
        ".tsv",
        ".xml",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
    }
)

# Имена файлов без расширения / с особыми именами — всегда пробуем как текст
EXTRA_FILENAMES = frozenset(
    {
        ".env",
        ".env.example",
        ".env.local",
        "Dockerfile",
        "Makefile",
        "requirements.txt",
        "LICENSE",
        "AUTHORS",
    }
)

# Бинарные / тяжёлые суффиксы — пропускаем
BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",
        ".pkl",
        ".pickle",
        ".joblib",
        ".parquet",
        ".feather",
        ".npy",
        ".npz",
        ".h5",
        ".hdf5",
        ".onnx",
        ".pt",
        ".pth",
        ".bin",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
    }
)

SEP_LINE = "=" * 80
HEADER_LINE = "-" * 80


def should_skip_dir(name: str) -> bool:
    if name in SKIP_DIRS:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def is_text_candidate(rel: Path) -> bool:
    name = rel.name
    if name in EXTRA_FILENAMES:
        return True
    if name.startswith(".env"):
        return True
    suf = rel.suffix.lower()
    if suf in BINARY_EXTENSIONS:
        return False
    if suf in TEXT_EXTENSIONS:
        return True
    if not suf and name in {"Dockerfile", "Makefile", "LICENSE"}:
        return True
    return False


def collect_files(root: Path, max_bytes: int) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if parts and parts[0] in SKIP_TOP_LEVEL_DIRS:
            continue
        if any(should_skip_dir(p) for p in parts):
            continue
        if not is_text_candidate(rel):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        out.append(rel)
    out.sort(key=lambda p: p.as_posix().lower())
    return out


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Экспорт кода и конфигов (включая .env) в один текстовый файл."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Корень проекта (по умолчанию — родитель scripts/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Выходной файл (по умолчанию project_export_YYYYMMDD_HHMMSS.txt в корне).",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=5.0,
        help="Не включать файлы больше этого размера (МБ). По умолчанию 5.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = (args.root or (script_dir.parent)).resolve()
    if not root.is_dir():
        print(f"Корень не найден: {root}", file=sys.stderr)
        return 1

    max_bytes = int(args.max_file_mb * 1024 * 1024)
    rel_files = collect_files(root, max_bytes=max_bytes)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output
    if out_path is None:
        out_path = root / f"project_export_{ts}.txt"
    else:
        out_path = out_path.resolve()

    lines: list[str] = []
    lines.append(SEP_LINE)
    lines.append("PROJECT EXPORT")
    lines.append(f"Root: {root.as_posix()}")
    lines.append(f"UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Files: {len(rel_files)}")
    lines.append(SEP_LINE)
    lines.append("")

    for rel in rel_files:
        abs_path = root / rel
        lines.append(SEP_LINE)
        lines.append(f"FILE: {rel.as_posix()}")
        lines.append(HEADER_LINE)
        lines.append(read_text_safe(abs_path))
        if not lines[-1].endswith("\n"):
            lines.append("")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    # ASCII-only message: Windows cp1252 консоль часто ломает кириллицу в print
    print(f"OK: {out_path} ({len(rel_files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
