#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed-root", type=Path, required=True)
    parser.add_argument("--restore-root", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    args = parser.parse_args()

    compressed_root = args.compressed_root.resolve()
    restore_root = args.restore_root.resolve()
    manifest = json.loads(args.batch_manifest.read_text())
    shards = manifest["shards"]
    if len(shards) != 8:
        raise RuntimeError(f"Expected 8 shards, got {len(shards)}")

    verified = []
    for shard in shards:
        archive = compressed_root / shard["local_name"]
        if not archive.is_file():
            raise FileNotFoundError(archive)
        actual = sha256(archive)
        if actual != shard["sha256"]:
            raise RuntimeError(
                f"Archive hash mismatch for {archive.name}: {actual} != {shard['sha256']}"
            )
        verified.append(
            {"archive": archive.name, "sha256": actual, "size_bytes": archive.stat().st_size}
        )

    restore_root.mkdir(parents=True, exist_ok=True)
    for index, shard in enumerate(shards, start=1):
        archive = compressed_root / shard["local_name"]
        print(json.dumps({"phase": "extract", "shard": index, "total": len(shards), "archive": archive.name}), flush=True)
        subprocess.run(
            [
                "tar",
                "--zstd",
                "-xf",
                str(archive),
                "-C",
                str(restore_root),
                "--wildcards",
                "data/efficient_wam/*",
            ],
            check=True,
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch": manifest.get("batch_name"),
        "archive_count": len(verified),
        "archives": verified,
        "restore_root": str(restore_root),
        "extracted_scope": "data/efficient_wam/*",
    }
    output = restore_root / "RESTORE_COMPLETE.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"phase": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
