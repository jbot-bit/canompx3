"""Weekly gold.db snapshot to C:\\backups\\gold-db.

Run manually or via Windows Task Scheduler. Single-writer pipeline so a
file-level copy is safe; SHA-256 verifies integrity. Keeps the most recent
4 snapshots, prunes the rest.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pipeline.paths import GOLD_DB_PATH

BACKUP_ROOT = Path(r"C:\backups\gold-db")
KEEP = 4


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    src = Path(GOLD_DB_PATH)
    if not src.exists():
        print(f"[FAIL] source missing: {src}", file=sys.stderr)
        return 2
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_size = src.stat().st_size
    print(f"[INFO] source {src} ({raw_size / 1e9:.2f} GB)")

    src_hash = _sha256(src)
    print(f"[INFO] source SHA256: {src_hash}")

    out_gz = BACKUP_ROOT / f"gold-{ts}.db.gz"
    print(f"[INFO] writing {out_gz}")
    with src.open("rb") as fin, gzip.open(out_gz, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)

    out_size = out_gz.stat().st_size
    ratio = out_size / raw_size if raw_size else 0
    print(f"[OK] wrote {out_size / 1e9:.2f} GB (ratio {ratio:.2%})")

    sidecar = out_gz.with_suffix(out_gz.suffix + ".sha256")
    sidecar.write_text(f"{src_hash}  gold.db\n", encoding="utf-8")

    snapshots = sorted(BACKUP_ROOT.glob("gold-*.db.gz"))
    if len(snapshots) > KEEP:
        for old in snapshots[: len(snapshots) - KEEP]:
            print(f"[PRUNE] {old.name}")
            old.unlink(missing_ok=True)
            old.with_suffix(old.suffix + ".sha256").unlink(missing_ok=True)

    print(f"[DONE] {len(snapshots[-KEEP:])} snapshots retained in {BACKUP_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
