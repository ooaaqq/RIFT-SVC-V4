from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

AUDIT_CHECKPOINT_PATTERN = re.compile(r"step-(\d+)\.pt$")
RESUME_CHECKPOINT_PATTERN = re.compile(r"resume-step-(\d+)\.pt$")
ERROR_PATTERN = re.compile(
    r"traceback|cuda out of memory|outofmemoryerror|runtimeerror|\bnan\b|\binf\b",
    re.IGNORECASE,
)
TRAINING_KEYS = (
    "total_loss",
    "flow",
    "learning_rate",
    "speaker_learning_rate",
    "grad_norm",
    "steps_per_second",
    "frames_per_second",
    "samples_per_second",
    "max_cuda_memory_gib",
)


@dataclass(frozen=True)
class LogHistory:
    training: list[dict[str, Any]]
    validations: list[dict[str, Any]]
    errors: list[str]


@dataclass(frozen=True)
class CheckpointStatus:
    audit_count: int
    latest_audit_step: int | None
    latest_audit_path: str | None
    latest_audit_age_seconds: float | None
    resume_count: int
    latest_resume_step: int | None
    latest_resume_path: str | None
    latest_resume_age_seconds: float | None
    best_exists: bool
    final_exists: bool


def parse_log(path: Path) -> LogHistory:
    if not path.is_file():
        return LogHistory([], [], [])
    training: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    lines = path.read_text(errors="replace").splitlines()
    for line in lines:
        stripped = line.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict) or "step" not in payload:
            continue
        if "validation_flow" in payload:
            validations.append(payload)
        elif "flow" in payload:
            training.append(payload)
    errors = [line.strip() for line in lines[-500:] if ERROR_PATTERN.search(line)]
    return LogHistory(training, validations, errors[-5:])


def latest_panel_event(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "audio-panel" / "panel_events.jsonl"
    if not path.is_file():
        return None
    latest = None
    for line in path.read_text(errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and "checkpoint_step" in payload:
            latest = payload
    return latest


def checkpoint_status(run_dir: Path, now: float | None = None) -> CheckpointStatus:
    now = time.time() if now is None else now

    def collect(pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
        candidates = []
        if run_dir.is_dir():
            for path in run_dir.glob("*.pt"):
                if match := pattern.fullmatch(path.name):
                    candidates.append((int(match.group(1)), path))
        return sorted(candidates)

    def latest(
        candidates: list[tuple[int, Path]],
    ) -> tuple[int | None, str | None, float | None]:
        if not candidates:
            return None, None, None
        step, path = candidates[-1]
        return step, str(path), max(0.0, now - path.stat().st_mtime)

    audits = collect(AUDIT_CHECKPOINT_PATTERN)
    resumes = collect(RESUME_CHECKPOINT_PATTERN)
    latest_audit_step, latest_audit_path, latest_audit_age = latest(audits)
    latest_resume_step, latest_resume_path, latest_resume_age = latest(resumes)
    return CheckpointStatus(
        audit_count=len(audits),
        latest_audit_step=latest_audit_step,
        latest_audit_path=latest_audit_path,
        latest_audit_age_seconds=latest_audit_age,
        resume_count=len(resumes),
        latest_resume_step=latest_resume_step,
        latest_resume_path=latest_resume_path,
        latest_resume_age_seconds=latest_resume_age,
        best_exists=(run_dir / "best.pt").is_file(),
        final_exists=(run_dir / "final.pt").is_file(),
    )


def _run_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_status() -> list[dict[str, str]]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    fields = (
        "index",
        "name",
        "utilization",
        "memory_used_mib",
        "memory_total_mib",
        "temperature_c",
        "power_w",
    )
    devices: list[dict[str, str]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            devices.append(dict(zip(fields, values, strict=True)))
    return devices


def matching_processes(pattern: str) -> list[dict[str, str | int]]:
    output = _run_command(["ps", "-eo", "pid=,etimes=,args="])
    if not output:
        return []
    current_pid = os.getpid()
    matches: list[dict[str, str | int]] = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid, elapsed, command = parts
        if int(pid) == current_pid or pattern not in command:
            continue
        matches.append(
            {"pid": int(pid), "elapsed_seconds": int(elapsed), "command": command}
        )
    return matches


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _median(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value for row in records if (value := _finite_number(row.get(key))) is not None
    ]
    return statistics.median(values) if values else None


def _load_limits(config_path: Path, max_steps: int | None) -> dict[str, int]:
    payload: dict[str, Any] = {}
    if config_path.is_file():
        payload = json.loads(config_path.read_text())
    training = payload.get("training", {})
    resolved_max = max_steps if max_steps is not None else training.get("max_steps")
    if not isinstance(resolved_max, int) or resolved_max <= 0:
        raise ValueError("max steps must be supplied by --max-steps or the config")
    return {
        "max_steps": resolved_max,
        "audit_checkpoint_every_steps": int(
            training.get("audit_checkpoint_every_steps", 0)
        ),
        "resume_checkpoint_every_steps": int(
            training.get("resume_checkpoint_every_steps", 0)
        ),
        "validation_every_steps": int(training.get("validation_every_steps", 0)),
    }


def build_snapshot(
    *,
    log_path: Path,
    run_dir: Path,
    config_path: Path,
    max_steps: int | None,
    window: int,
    process_pattern: str,
    stale_after: float = 300.0,
) -> dict[str, Any]:
    now = time.time()
    limits = _load_limits(config_path, max_steps)
    history = parse_log(log_path)
    recent = history.training[-window:]
    last = history.training[-1] if history.training else {}
    step = int(last.get("step", 0))
    rates = {key: _median(recent, key) for key in TRAINING_KEYS}
    steps_per_second = rates["steps_per_second"]
    eta_seconds = (
        max(0, limits["max_steps"] - step) / steps_per_second
        if steps_per_second and steps_per_second > 0
        else None
    )
    validations = []
    for item in history.validations:
        if _finite_number(item.get("validation_flow")) is None:
            continue
        validation = {
            "step": int(item["step"]),
            "validation_flow": float(item["validation_flow"]),
        }
        condition = item.get("condition_f0_voicing", {}).get("overall", {})
        for name in ("voiced_ratio", "f0_hz_median"):
            if (value := _finite_number(condition.get(name))) is not None:
                validation[name] = value
        validations.append(validation)
    best_validation = (
        min(validations, key=lambda item: item["validation_flow"])
        if validations
        else None
    )
    latest_validation = validations[-1] if validations else None
    checkpoints = checkpoint_status(run_dir, now)
    processes = matching_processes(process_pattern)
    log_age = max(0.0, now - log_path.stat().st_mtime) if log_path.is_file() else None
    if checkpoints.final_exists and step >= limits["max_steps"]:
        state = "COMPLETE"
    elif processes and (log_age is None or log_age > stale_after):
        state = "STALE"
    elif processes:
        state = "RUNNING"
    elif log_age is not None and log_age <= stale_after:
        state = "RECENT_NO_PROCESS"
    else:
        state = "STOPPED"
    warnings = list(history.errors)
    for key in TRAINING_KEYS:
        if key in last and _finite_number(last[key]) is None:
            warnings.append(f"non-finite {key} at step {step}")
    if state == "STALE":
        warnings.append(
            f"training process exists but log is stale ({format_duration(log_age)})"
        )
    checkpoint_lag_fields = (
        (
            "audit",
            checkpoints.latest_audit_step,
            limits["audit_checkpoint_every_steps"],
        ),
        (
            "resume",
            checkpoints.latest_resume_step,
            limits["resume_checkpoint_every_steps"],
        ),
    )
    for name, checkpoint_step, interval in checkpoint_lag_fields:
        if checkpoint_step is not None and interval:
            lag = step - checkpoint_step
            if lag > interval * 2:
                warnings.append(
                    f"latest {name} checkpoint trails the log by {lag:,} steps"
                )
    try:
        usage = shutil.disk_usage(run_dir if run_dir.exists() else run_dir.parent)
        disk = {
            "free_gib": usage.free / 2**30,
            "total_gib": usage.total / 2**30,
            "used_percent": usage.used / usage.total * 100,
        }
    except OSError:
        disk = None
    latest_metrics = {
        key: _finite_number(last.get(key)) for key in TRAINING_KEYS if key in last
    }
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "state": state,
        "step": step,
        "max_steps": limits["max_steps"],
        "progress_percent": step / limits["max_steps"] * 100,
        "eta_seconds": eta_seconds,
        "log_age_seconds": log_age,
        "metrics_median": rates,
        "latest_metrics": latest_metrics,
        "latest_validation": latest_validation,
        "best_validation": best_validation,
        "recent_validations": validations[-6:],
        "latest_audio_panel": latest_panel_event(run_dir),
        "checkpoints": asdict(checkpoints),
        "processes": processes,
        "gpus": gpu_status(),
        "disk": disk,
        "warnings": warnings[-8:],
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _number(value: Any, digits: int = 4) -> str:
    number = _finite_number(value)
    return f"{number:.{digits}f}" if number is not None else "n/a"


def render_dashboard(snapshot: dict[str, Any], width: int = 40) -> str:
    progress = min(1.0, max(0.0, snapshot["progress_percent"] / 100))
    filled = round(width * progress)
    bar = "#" * filled + "-" * (width - filled)
    metrics = snapshot["metrics_median"]
    checkpoint = snapshot["checkpoints"]
    lines = [
        f"RIFT-SVC V4 TRAINING  {snapshot['timestamp']}",
        f"State: {snapshot['state']}",
        "",
        f"[{bar}] {snapshot['progress_percent']:6.2f}%",
        f"Step: {snapshot['step']:,} / {snapshot['max_steps']:,}    "
        f"ETA: {format_duration(snapshot['eta_seconds'])}",
        f"Rate (median): {_number(metrics['steps_per_second'], 3)} step/s    "
        f"{_number(metrics['frames_per_second'], 0)} frame/s",
        f"Loss: {_number(metrics['total_loss'], 6)}    "
        f"Flow: {_number(metrics['flow'], 6)}    "
        f"Grad: {_number(metrics['grad_norm'], 4)}",
        "LR backbone: "
        f"{_number(snapshot['latest_metrics'].get('learning_rate'), 9)}    "
        "speaker: "
        f"{_number(snapshot['latest_metrics'].get('speaker_learning_rate'), 9)}",
        "",
    ]
    latest_validation = snapshot["latest_validation"]
    best_validation = snapshot["best_validation"]
    if latest_validation and best_validation:
        online = latest_validation.get("online_validation_flow")
        online_text = f"    online {_number(online, 6)}" if online is not None else ""
        lines.append(
            "Validation EMA: "
            f"latest {_number(latest_validation['validation_flow'], 6)} "
            f"@ {latest_validation['step']:,}    "
            f"best {_number(best_validation['validation_flow'], 6)} "
            f"@ {best_validation['step']:,}"
            f"{online_text}"
        )
        history = "  ".join(
            f"{item['step'] // 1000}k:{item['validation_flow']:.5f}"
            for item in snapshot["recent_validations"]
        )
        lines.append(f"Recent validation: {history}")
        if "voiced_ratio" in latest_validation:
            lines.append(
                "Condition F0: voiced "
                f"{latest_validation['voiced_ratio'] * 100:.1f}%    "
                f"median {_number(latest_validation.get('f0_hz_median'), 1)} Hz"
            )
    else:
        lines.append("Validation: no completed validation record")
    panel = snapshot.get("latest_audio_panel")
    if panel:
        pitch = panel.get("aggregate_pitch", {})
        lines.append(
            f"Audio panel @ {int(panel['checkpoint_step']):,}: "
            f"voicing F1 {_number(pitch.get('voicing_f1'), 3)}    "
            f"F0 MAE {_number(pitch.get('f0_cents_mae'), 1)} cents    "
            f"gross {_number(100 * pitch.get('gross_pitch_error_ratio', 0), 1)}%"
        )
    lines.extend(
        [
            f"Audit checkpoints: {checkpoint['audit_count']}    "
            f"latest {checkpoint['latest_audit_step'] or 0:,}",
            f"Resume checkpoints: {checkpoint['resume_count']}    "
            f"latest {checkpoint['latest_resume_step'] or 0:,}    "
            f"best={'yes' if checkpoint['best_exists'] else 'no'}    "
            f"final={'yes' if checkpoint['final_exists'] else 'no'}",
            f"Log age: {format_duration(snapshot['log_age_seconds'])}",
        ]
    )
    if snapshot["processes"]:
        oldest = max(
            snapshot["processes"], key=lambda process: process["elapsed_seconds"]
        )
        lines.append(
            f"Processes: {len(snapshot['processes'])}    main PID {oldest['pid']}    "
            f"uptime {format_duration(oldest['elapsed_seconds'])}"
        )
    for gpu in snapshot["gpus"]:
        lines.append(
            f"GPU {gpu['index']}: {gpu['name']}    util {gpu['utilization']}%    "
            f"VRAM {gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB    "
            f"{gpu['temperature_c']} C    {gpu['power_w']} W"
        )
    disk = snapshot["disk"]
    if disk:
        lines.append(
            f"Disk: {disk['free_gib']:.1f} GiB free / {disk['total_gib']:.1f} GiB    "
            f"used {disk['used_percent']:.1f}%"
        )
    if snapshot["warnings"]:
        lines.append("")
        lines.append("WARNINGS")
        lines.extend(f"- {warning}" for warning in snapshot["warnings"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously monitor RIFT V4 training"
    )
    parser.add_argument("--log", type=Path, required=True, help="training JSON log")
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="checkpoint directory"
    )
    parser.add_argument("--config", type=Path, default=Path("config/v4.json"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument(
        "--window", type=int, default=100, help="training rows used for medians"
    )
    parser.add_argument("--process-pattern", default="rift_v4.train")
    parser.add_argument(
        "--stale-after",
        type=float,
        default=300.0,
        help="seconds without a log update before a running job is stale",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="emit one machine-readable snapshot"
    )
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()
    if args.refresh <= 0:
        parser.error("--refresh must be positive")
    if args.window <= 0:
        parser.error("--window must be positive")
    if args.stale_after <= 0:
        parser.error("--stale-after must be positive")
    try:
        while True:
            snapshot = build_snapshot(
                log_path=args.log,
                run_dir=args.run_dir,
                config_path=args.config,
                max_steps=args.max_steps,
                window=args.window,
                process_pattern=args.process_pattern,
                stale_after=args.stale_after,
            )
            if args.json:
                print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
            else:
                if not args.no_clear and sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                print(render_dashboard(snapshot), flush=True)
            if args.once or args.json:
                break
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
