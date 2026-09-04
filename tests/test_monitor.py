import json
import time
from pathlib import Path

from rift_v4.monitor import (
    checkpoint_status,
    format_duration,
    latest_panel_event,
    parse_log,
    render_dashboard,
)


def test_parse_log_ignores_pretty_config_and_collects_metrics(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "{\n"
        '  "recordings": 10\n'
        "}\n"
        + json.dumps(
            {
                "step": 20,
                "flow": 0.2,
                "learning_rate": 0.001,
                "steps_per_second": 3.5,
            }
        )
        + "\n"
        + json.dumps({"step": 20, "validation_flow": 0.3})
        + "\n"
    )
    history = parse_log(log)
    assert [row["step"] for row in history.training] == [20]
    assert history.validations == [{"step": 20, "validation_flow": 0.3}]
    assert history.errors == []


def test_checkpoint_status_uses_numeric_step_order(tmp_path: Path) -> None:
    (tmp_path / "step-0005000.pt").touch()
    latest = tmp_path / "step-0010000.pt"
    latest.touch()
    resume = tmp_path / "resume-step-0005000.pt"
    resume.touch()
    (tmp_path / "best.pt").touch()
    status = checkpoint_status(tmp_path, now=time.time())
    assert status.audit_count == 2
    assert status.latest_audit_step == 10_000
    assert status.latest_audit_path == str(latest)
    assert status.resume_count == 1
    assert status.latest_resume_step == 5_000
    assert status.latest_resume_path == str(resume)
    assert status.best_exists
    assert not status.final_exists


def test_latest_panel_event_ignores_incomplete_lines(tmp_path: Path) -> None:
    panel = tmp_path / "audio-panel"
    panel.mkdir()
    (panel / "panel_events.jsonl").write_text(
        '{"checkpoint_step": 50000, "aggregate_pitch": {}}\n{"checkpoint_step"'
    )

    assert latest_panel_event(tmp_path)["checkpoint_step"] == 50_000


def test_duration_and_dashboard_are_compact() -> None:
    assert format_duration(90) == "1m 30s"
    assert format_duration(3_661) == "1h 01m 01s"
    snapshot = {
        "timestamp": "2026-08-30T08:00:00+08:00",
        "state": "RUNNING",
        "step": 50_000,
        "max_steps": 300_000,
        "progress_percent": 100 / 6,
        "eta_seconds": 70_000,
        "log_age_seconds": 2,
        "metrics_median": {
            "total_loss": 0.26,
            "flow": 0.15,
            "learning_rate": 0.0002,
            "grad_norm": 0.9,
            "steps_per_second": 3.5,
            "frames_per_second": 26_000,
            "samples_per_second": 90,
            "max_cuda_memory_gib": 7.6,
        },
        "latest_metrics": {
            "learning_rate": 0.0002,
        },
        "latest_validation": {
            "step": 50_000,
            "validation_flow": 0.16,
            "online_validation_flow": 0.15,
        },
        "best_validation": {"step": 50_000, "validation_flow": 0.16},
        "recent_validations": [{"step": 50_000, "validation_flow": 0.16}],
        "checkpoints": {
            "audit_count": 25,
            "latest_audit_step": 50_000,
            "latest_audit_path": "/runs/step-0050000.pt",
            "latest_audit_age_seconds": 10,
            "resume_count": 10,
            "latest_resume_step": 50_000,
            "latest_resume_path": "/runs/resume-step-0050000.pt",
            "latest_resume_age_seconds": 10,
            "best_exists": True,
            "final_exists": False,
        },
        "processes": [{"pid": 1, "elapsed_seconds": 10, "command": "train"}],
        "gpus": [],
        "disk": {"free_gib": 100.0, "total_gib": 200.0, "used_percent": 50.0},
        "warnings": [],
    }
    rendered = render_dashboard(snapshot)
    assert "50,000 / 300,000" in rendered
    assert "Loss: 0.260000" in rendered
    assert "Grad: 0.9000" in rendered
    assert "LR backbone: 0.000200000" in rendered
    assert "speaker: n/a" in rendered
    assert "best 0.160000 @ 50,000" in rendered
    assert "online 0.150000" in rendered
    assert "ETA: 19h 26m 40s" in rendered
    assert "Audit checkpoints: 25" in rendered
    assert "Resume checkpoints: 10" in rendered
