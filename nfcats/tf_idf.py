import bisect
import csv
import json
import logging
import pathlib
import pickle
import shutil
import threading
import time
from typing import List, Tuple

import dill
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from nfcats import DATA_PATH, ROOT_PATH

LOGGER = logging.getLogger('tf_idf')


class TrialSaver:
    def __init__(self, folder: pathlib.Path, num_best: int = 1):
        if num_best < 1:
            raise ValueError('num_best must be greater than 0')

        self._folder = folder
        self._num_best = num_best

        self._lock = threading.Lock()
        self._results: List[float] = []
        self._folders: List[pathlib.Path] = []

    def save(self, trial: optuna.Trial, pipeline: Pipeline, result: float):
        with self._lock:
            result = -result  # negation allows to keep the smallest result at the end of the sorted result list
            if len(self._results) >= self._num_best and self._results[-1] <= result:
                return

            trial_folder = self._folder / f'trial_{trial.number}'
            trial_folder.mkdir(parents=True, exist_ok=True)

            with (trial_folder / f'{trial.number}.model').open('wb') as f:
                dill.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

            with (trial_folder / 'params.json').open('w', encoding='utf-8') as f:
                json.dump(trial.params, f, ensure_ascii=False, indent=4)

            index = bisect.bisect_left(self._results, result)
            self._results.insert(index, result)
            self._folders.insert(index, trial_folder)

            while len(self._results) > self._num_best:
                self._results.pop()
                shutil.rmtree(self._folders.pop())


def get_word_vectorizer(trial: optuna.Trial) -> TfidfVectorizer:
    max_ngram = trial.suggest_int('features.word.max_ngram', 1, 2)
    min_df = trial.suggest_int('features.word.min_df', 2, 50)

    return TfidfVectorizer(
        analyzer='word',
        min_df=min_df,
        ngram_range=(1, max_ngram),
        sublinear_tf=trial.suggest_categorical('features.word.sublinear_tf', [False, True]),
        use_idf=trial.suggest_categorical('features.word.use_idf', [False, True]),
        smooth_idf=trial.suggest_categorical('features.word.smooth_idf', [False, True]),
        binary=trial.suggest_categorical('features.word.binary', [False, True]),
    )


def get_char_vectorizer(trial: optuna.Trial) -> TfidfVectorizer:
    min_ngram = trial.suggest_int('features.char.min_ngram', 1, 5)
    max_ngram = trial.suggest_int('features.char.max_ngram', min_ngram, 5)
    min_df = trial.suggest_int('features.char.min_df', 2, 50)

    if trial.suggest_categorical('features.char.word_boundaries', [False, True]):

        def char_analyzer(tokens):
            return ' '.join(f'<{token}>' for token in tokens)

    else:

        def char_analyzer(tokens):
            return ' '.join(tokens)

    return TfidfVectorizer(
        analyzer=char_analyzer,
        min_df=min_df,
        ngram_range=(min_ngram, max_ngram),
        sublinear_tf=trial.suggest_categorical('features.char.sublinear_tf', [False, True]),
        use_idf=trial.suggest_categorical('features.char.use_idf', [False, True]),
        smooth_idf=trial.suggest_categorical('features.char.smooth_idf', [False, True]),
        binary=trial.suggest_categorical('features.char.binary', [False, True]),
    )


def parametrize(trial: optuna.Trial):
    feature_type = trial.suggest_categorical('features.type', ['word', 'char', 'all'])

    featurizers = []
    if feature_type in ['word', 'all']:
        featurizers.append(('word', get_word_vectorizer(trial)))
    if feature_type in ['char', 'all']:
        featurizers.append(('char', get_char_vectorizer(trial)))

    features = FeatureUnion(featurizers)

    classifier = LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=1000000,
        random_state=42,
        class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
        C=trial.suggest_loguniform('C', low=1e-3, high=1e3),
    )

    return Pipeline([('features', features), ('classifier', classifier)])


def load_question_dataset(path: pathlib.Path) -> Tuple[List[str], List[str]]:
    with path.open(encoding='utf-8') as f:
        data = list(csv.DictReader(f))

    texts = [item['question'] for item in data]
    labels = [item['category'] for item in data]
    return texts, labels


def main():
    logging.basicConfig(format='%(asctime)s [%(name)s] %(levelname)s - %(message)s', level=logging.INFO)

    train_texts, train_labels = load_question_dataset(DATA_PATH / 'train.csv')
    test_texts, test_labels = load_question_dataset(DATA_PATH / 'test.csv')
    val_texts, val_labels = load_question_dataset(DATA_PATH / 'valid.csv')

    trial_saver = TrialSaver(num_best=5, folder=ROOT_PATH / 'tf_idf_models')

    def run_trial(trial: optuna.Trial) -> float:
        pipeline = parametrize(trial)

        start = time.perf_counter()
        pipeline.fit(train_texts, train_labels)
        training_time = time.perf_counter() - start
        LOGGER.info('Training time: %.3fs', training_time)

        test_score = f1_score(y_true=test_labels, y_pred=pipeline.predict(test_texts), average='macro')
        val_score = f1_score(y_true=val_labels, y_pred=pipeline.predict(val_texts), average='macro')

        trial.user_attrs['test'] = test_score
        trial.user_attrs['val'] = val_score

        LOGGER.info('Trial %i test score: %s', trial.number, test_score)
        LOGGER.info('Trial %i val score: %s', trial.number, val_score)

        trial_saver.save(trial=trial, pipeline=pipeline, result=val_score)
        return val_score

    study = optuna.create_study(
        storage=f'sqlite:///{ROOT_PATH}/hypertuning.db', study_name='tf-idf', direction='maximize', load_if_exists=True,
    )
    study.optimize(run_trial, n_trials=1000)


if __name__ == "__main__":
    main()
