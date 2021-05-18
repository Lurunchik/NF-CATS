# pylint: disable=W0223
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from allennlp.common import Registrable
from allennlp.data import TextFieldTensors
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask


class SentenceEmbedder(Registrable, ABC):
    @abstractmethod
    def get_output_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, tokens: Optional[TextFieldTensors] = None, texts: Optional[List[str]] = None,) -> torch.Tensor:
        pass


@SentenceEmbedder.register('trainable')
class TrainableEmbedder(torch.nn.Module, SentenceEmbedder):
    def __init__(
        self,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
    ) -> None:
        super().__init__()
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder

    def get_output_dim(self) -> int:
        return self._seq2vec_encoder.get_output_dim()

    def forward(self, tokens: Optional[TextFieldTensors] = None, texts: Optional[List[str]] = None,) -> torch.Tensor:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder is not None:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        return embedded_text
