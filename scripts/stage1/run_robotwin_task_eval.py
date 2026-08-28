from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or None


def _ensure_policy_link(
    robotwin_root: Path, policy_dir: Path, policy_name: str
) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")
    target = policy_root / policy_name
    source = policy_dir.resolve()
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(f"Policy link conflict: {target} -> {target.resolve()}")
        return target
    if target.exists():
        raise RuntimeError(f"Policy path already exists and is not a symlink: {target}")
    target.symlink_to(source, target_is_directory=True)
    return target


def _override(name: str, value: Any) -> list[str]:
    rendered = repr(str(value)) if isinstance(value, Path) else repr(value)
    return [f"--{name}", rendered]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one auditable OVCR-S RoboTwin task evaluation"
    )
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--safety-src", type=Path, required=True)
    parser.add_argument("--efficient-inference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--policy-name",
        choices=("ovcrs_policy", "aha_ovcrs_policy"),
        default="ovcrs_policy",
    )
    parser.add_argument(
        "--task-config",
        choices=("demo_clean", "demo_randomized"),
        default="demo_clean",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    robotwin_root = args.robotwin_root.expanduser().resolve()
    policy_dir = args.policy_dir.expanduser().resolve()
    runtime_config = args.runtime_config.expanduser().resolve()
    safety_src = args.safety_src.expanduser().resolve()
    efficient_root = args.efficient_inference_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    for required in (policy_dir, runtime_config, safety_src, efficient_root):
        if not required.exists():
            raise FileNotFoundError(required)

    policy_link = _ensure_policy_link(
        robotwin_root, policy_dir, args.policy_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        str(runtime_config),
        "--overrides",
        *_override("policy_name", args.policy_name),
        *_override("task_name", args.task),
        *_override("task_config", args.task_config),
        *_override("eval_num_episodes", args.episodes),
        *_override("seed", args.seed),
        *_override("eval_output_dir", output_dir),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    environment["PYTHONUNBUFFERED"] = "1"
    python_paths = [str(safety_src), str(efficient_root)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "task": args.task,
        "task_config": args.task_config,
        "episodes": args.episodes,
        "seed": args.seed,
        "policy_name": args.policy_name,
        "physical_gpu": args.gpu_id,
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": _sha256(runtime_config),
        "policy_dir": str(policy_dir),
        "policy_link": str(policy_link),
        "safety_commit": _git_commit(policy_dir),
        "efficient_commit": _git_commit(efficient_root),
        "robotwin_commit": _git_commit(robotwin_root),
        "command": command,
    }
    manifest_path = output_dir / "launch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    log_path = output_dir / "eval.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=robotwin_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
