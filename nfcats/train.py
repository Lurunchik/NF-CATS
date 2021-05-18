import json
import logging
from datetime import datetime
from pathlib import Path

import wandb
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.models import load_archive

from .classifier import NFQCatsClassifier
from .dataset_reader import TextClassificationCsvReader
from .sampler import BalancedBatchSampler
from .wandb_callback import WnBCallback

model_params = 'nft_classifier.jsonnet'

DATA_PATH = 'data/v03'
logging.basicConfig(level='DEBUG')
if __name__ == '__main__':
    wandb.init(project='non-factoid-classification', reinit=True, name='roberta-tuned-on-squad')
    params = Params.from_file(
        model_params,
        ext_vars={
            'TRAIN_DATA_PATH': f'{DATA_PATH}/train.csv',
            'VALID_DATA_PATH': f'{DATA_PATH}/valid.csv',
            'EPOCHS': '10',
            'DEVICE': '0',
        },
    )

    date = datetime.utcnow().strftime('%H%M%S-%d%m')
    serialization_dir = Path(f'./logs/{date}_{model_params}')

    train_model(params=params, serialization_dir=str(serialization_dir))

    wandb.config.update(  # noqa: E1101 pylint: disable=no-member
        {'serialization_dir': str(serialization_dir), **params.as_flat_dict()}
    )

    with (serialization_dir / 'metrics.json').open() as f:
        metrics = json.load(f)
        wandb.run.summary.update(metrics)

    wandb.save(str(serialization_dir / 'model.tar.gz'))
