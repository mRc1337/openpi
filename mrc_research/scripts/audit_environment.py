"""Read-only environment audit for CASK/OpenPI development."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import platform
import subprocess
import sys


PINNED_OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def command_output(command: list[str], *, cwd: pathlib.Path | None = None) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def dependency_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def audit(openpi_dir: pathlib.Path) -> dict[str, object]:
    openpi_dir = openpi_dir.resolve()
    git_head = command_output(["git", "rev-parse", "HEAD"], cwd=openpi_dir)
    git_status = command_output(["git", "status", "--short"], cwd=openpi_dir)
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    dependencies = {
        module: dependency_available(module)
        for module in ("jax", "flax", "torch", "openpi", "pytest")
    }
    python_311 = sys.version_info[:2] == (3, 11)
    linux = platform.system() == "Linux"
    pinned_commit = git_head.get("stdout") == PINNED_OPENPI_COMMIT
    runtime_ready = (
        python_311
        and linux
        and bool(gpu.get("available"))
        and dependencies["jax"]
        and dependencies["flax"]
        and dependencies["openpi"]
    )
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "is_3_11": python_311,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "openpi_supported_linux": linux,
        },
        "dependencies": dependencies,
        "gpu": gpu,
        "openpi": {
            "path": str(openpi_dir),
            "exists": openpi_dir.exists(),
            "head": git_head,
            "status": git_status,
            "matches_pinned_commit": pinned_commit,
        },
        "runtime_ready": runtime_ready,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openpi-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2] / "openpi",
    )
    parser.add_argument("--json-out", type=pathlib.Path)
    args = parser.parse_args()
    report = audit(args.openpi_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.json_out is not None:
        args.json_out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
