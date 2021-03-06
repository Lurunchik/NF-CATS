# pylint: disable=F401
import json
import logging
from datetime import datetime
from pathlib import Path

import wandb
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.models import load_archive

from nfcats import DATA_PATH, ROOT_PATH
from nfcats.classifier import NFQCatsClassifier
from nfcats.dataset_reader import TextClassificationCsvReader
from nfcats.sampler import BalancedBatchSampler
from nfcats.wandb_callback import WnBCallback

WB_LOGIN = 'your_login'


def main():
    logging.basicConfig(format='%(asctime)s [%(name)s] %(levelname)s - %(message)s', level=logging.DEBUG)

    model_params = ROOT_PATH / 'nft_classifier.jsonnet'

    wandb.init(entity=WB_LOGIN, project='non-factoid-classification', reinit=True, name='roberta-tuned-on-squad')
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


if __name__ == "__main__":
    main()
