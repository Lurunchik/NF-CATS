import csv
import logging
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Tokenizer
from overrides import overrides

LOGGER = logging.getLogger(__name__)


@DatasetReader.register('csv_text_label')
class TextClassificationCsvReader(TextClassificationJsonReader):
    _default_transformer_model = 'bert-base-uncased'

    def __init__(
        self,
        lower: bool = True,
        sep: str = ',',
        label_field: str = 'label',
        text_field: str = 'text',
        filter_rows_by_values: Dict = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer or PretrainedTransformerTokenizer(self._default_transformer_model),
            token_indexers=token_indexers or {'tokens': PretrainedTransformerIndexer(self._default_transformer_model)},
            **kwargs,
        )
        self._filters = filter_rows_by_values or {}
        self._sep = sep
        self._lower = lower
        self._label, self._text = label_field, text_field

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)  # if `file_path` is a URL, redirect to the cache

        LOGGER.info('Reading file at %s', file_path)

        with open(file_path) as dataset_file:
            data = csv.DictReader(dataset_file, delimiter=self._sep)
            for row in data:
                for field_name, field_value in self._filters.items():
                    if row[field_name] != field_value:
                        break
                else:
                    text = row[self._text].lower() if self._lower else row[self._text]
                    instance = self.text_to_instance(text=text, label=row[self._label])
                    if instance is not None:
                        yield instance
