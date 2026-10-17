"""Log every raw file (download date, SHA-256, size) - hard rule 3.

    python scripts/register_raw.py

Scans data/raw/ and writes data/raw/_provenance.csv (git-ignored with the data)
plus docs/raw_provenance.md (committed). Files are only read, never modified.
Source URLs come from data/raw/sources.csv (path,source_url,downloaded_on) that you
fill in once; anything missing there is flagged.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SOURCES = RAW / "sources.csv"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sources = {}
    if SOURCES.exists():
        with open(SOURCES, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sources[row["path"].replace("\\", "/")] = row
    rows = []
    for p in sorted(RAW.rglob("*")):
        if not p.is_file() or p.name in {".gitkeep", "sources.csv", "_provenance.csv"}:
            continue
        rel = p.relative_to(RAW).as_posix()
        src = sources.get(rel, {})
        rows.append({
            "path": rel,
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "file_modified": datetime.fromtimestamp(p.stat().st_mtime, UTC).date().isoformat(),
            "downloaded_on": src.get("downloaded_on", "MISSING"),
            "source_url": src.get("source_url", "MISSING"),
        })
    with open(RAW / "_provenance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path"])
        w.writeheader()
        w.writerows(rows)
    lines = ["# Raw file provenance", "",
             f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} "
             "by scripts/register_raw.py.", "",
             "| file | downloaded | sha256 (first 16) | bytes | source |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['path']} | {r['downloaded_on']} | `{r['sha256'][:16]}` "
                     f"| {r['bytes']} | {r['source_url']} |")
    (ROOT / "docs" / "raw_provenance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [r["path"] for r in rows if r["source_url"] == "MISSING"]
    print(f"{len(rows)} raw files registered -> docs/raw_provenance.md")
    if missing:
        print("No source URL / date in data/raw/sources.csv for:", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
