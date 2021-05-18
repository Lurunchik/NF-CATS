import pandas as pd
import tqdm
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

from nfcats import MODEL_PATH, DATA_PATH
from nfcats.classifier import NFQCatsClassifier
from nfcats.dataset_reader import TextClassificationCsvReader
from nfcats.sampler import BalancedBatchSampler
from nfcats.utils import pandas_classification_report
from nfcats.wandb_callback import WnBCallback

if __name__ == '__main__':
    archive = load_archive(MODEL_PATH, cuda_device=-1)
    predictor = Predictor.from_archive(
        archive, predictor_name='text_classifier', dataset_reader_to_load='csv_text_label'
    )

    test_df = pd.read_csv(f'{DATA_PATH}/test.csv', sep=',')

    y_pred = [predictor.predict(q) for q in tqdm.tqdm(test_df.question)]
    print(pandas_classification_report(y_true=test_df.category, y_pred=[p['label'] for p in y_pred]))

    print(predictor.predict('why do we need a taxonomy?'))
