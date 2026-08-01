"""
utils/logging_callbacks.py

Provides WandB and TensorBoard integration.
Both are optional runtime dependencies — the module degrades gracefully
if neither is installed.
"""
import os
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import AppConfig


def resolve_report_to(config: "AppConfig") -> Optional[str]:
    """
    Returns the HuggingFace Trainer `report_to` string based on config.

    Priority order:
    1. `training.report_to` if explicitly set
    2. `logging.backend` if `logging.enabled` is True
    3. "none" (no logging)
    """
    if config.training.report_to is not None:
        return config.training.report_to
    if config.logging.enabled:
        return config.logging.backend  # "wandb" | "tensorboard"
    return "none"


def build_callbacks(config: "AppConfig") -> List:
    """
    Returns a list of HuggingFace Trainer callbacks depending on config.
    """
    callbacks = []

    if not config.logging.enabled:
        return callbacks

    backend = config.logging.backend.lower()

    # ------------------------------------------------------------------
    # WandB Callback
    # ------------------------------------------------------------------
    if backend == "wandb":
        try:
            import wandb
            from transformers import TrainerCallback

            class WandBSetupCallback(TrainerCallback):
                """Initialises a WandB run before training starts."""
                def __init__(self, cfg):
                    self.cfg = cfg

                def on_train_begin(self, args, state, control, **kwargs):
                    if not wandb.run:
                        wandb.init(
                            project=self.cfg.logging.project_name,
                            name=self.cfg.logging.run_name,
                            tags=self.cfg.logging.tags,
                            config=self.cfg.model_dump(),
                        )

                def on_train_end(self, args, state, control, **kwargs):
                    if self.cfg.logging.log_model and wandb.run:
                        artifact = wandb.Artifact(
                            name=f"model-{wandb.run.id}",
                            type="model",
                        )
                        artifact.add_dir(args.output_dir)
                        wandb.log_artifact(artifact)
                    if wandb.run:
                        wandb.finish()

            callbacks.append(WandBSetupCallback(config))
            print(f"=== WandB Logging ENABLED → project='{config.logging.project_name}' ===")
        except ImportError:
            print("WARNING: WandB requested but 'wandb' package not installed. "
                  "Run `pip install wandb` to enable. Continuing without WandB logging.")

    # ------------------------------------------------------------------
    # TensorBoard Callback
    # ------------------------------------------------------------------
    elif backend == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
            from transformers import TrainerCallback

            class TensorBoardSetupCallback(TrainerCallback):
                """Writes loss and LR scalars to TensorBoard every logging step."""
                def __init__(self, log_dir: str):
                    self.log_dir = log_dir
                    self.writer: Optional[SummaryWriter] = None

                def on_train_begin(self, args, state, control, **kwargs):
                    os.makedirs(self.log_dir, exist_ok=True)
                    self.writer = SummaryWriter(log_dir=self.log_dir)
                    print(f"=== TensorBoard Logging ENABLED → dir='{self.log_dir}' ===")
                    print(f"    Run: tensorboard --logdir {self.log_dir}")

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if self.writer and logs:
                        step = state.global_step
                        for k, v in logs.items():
                            if isinstance(v, (int, float)):
                                self.writer.add_scalar(k, v, step)

                def on_train_end(self, args, state, control, **kwargs):
                    if self.writer:
                        self.writer.flush()
                        self.writer.close()

            callbacks.append(TensorBoardSetupCallback(config.logging.tensorboard_dir))
        except ImportError:
            print("WARNING: TensorBoard requested but 'tensorboard' not installed. "
                  "Run `pip install tensorboard` to enable. Continuing without TensorBoard.")

    return callbacks
