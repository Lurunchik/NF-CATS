from typing import Any, Dict, List, Optional

import wandb
from allennlp.training import GradientDescentTrainer, TrackEpochCallback, TrainerCallback

DEFAULT_METRICS = [
    'training_loss',
    'training_reg_loss',
    'training_f1',
    'training_accuracy',
    'training_weighted_f05',
    'training_pos_weighted_f05',
    'validation_loss',
    'validation_f1',
    'validation_accuracy',
    'validation_weighted_f05',
    'validation_pos_weighted_f05',
]


@TrainerCallback.register('wandb')
class WnBCallback(TrackEpochCallback):
    def __init__(self, metrics_to_include: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self._metrics_to_include = metrics_to_include or DEFAULT_METRICS

    def on_epoch(
        self, trainer: 'GradientDescentTrainer', metrics: Dict[str, Any], epoch: int, is_primary: bool = True, **kwargs,
    ) -> None:
        if is_primary:
            wandb.log({key: val for key, val in metrics.items() if key in self._metrics_to_include})
        super().on_epoch(trainer, metrics, epoch, is_primary, **kwargs)
