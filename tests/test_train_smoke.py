"""End-to-end smoke test: full train/val loop on dummy data, CPU-only."""

import subprocess
import sys
from pathlib import Path


def test_default_pipeline_runs(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "model_trainer.train",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=2",
        "trainer.limit_val_batches=1",
        "trainer.log_every_n_steps=1",
        "data.length=32",
        "data.val_length=16",
        "data.batch_size=8",
        "data.vocab_size=128",
        "model.hidden_size=16",
        "model.num_layers=1",
        "model.vocab_size=128",
        "experiment.test_after_fit=false",
        "callbacks.early_stopping.patience=99",
        f"hydra.run.dir={tmp_path}/run",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"train failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
