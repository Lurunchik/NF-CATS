from typing import Any, Dict, List, Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.training.metrics import FBetaMeasure
from overrides import overrides

from nfcats.embedder import SentenceEmbedder


@Model.register('nfq_cats_classifier')
class NFQCatsClassifier(Model):
    default_predictor = 'text_classifier'
    label_namespace = 'labels'
    namespace = 'tokens'

    def __init__(
        self,
        vocab: Vocabulary,
        embedder: Optional[SentenceEmbedder] = None,
        embeddings_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        feedforward: Optional[FeedForward] = None,
    ) -> None:
        super().__init__(vocab)
        self._embedder = embedder
        self._feedforward = feedforward

        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        elif self._embedder is not None:
            self._classifier_input_dim = self._embedder.get_output_dim()
        else:
            self._classifier_input_dim = embeddings_dim

        if self._embedder is None and embeddings_dim is None:
            raise ValueError('You must pass `Embedder` or `embeddings_dim`')

        self._dropout = torch.nn.Dropout(dropout) if dropout else None

        self._num_labels = vocab.get_vocab_size(namespace=self.label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset)
        metrics['weighted_f1'] = FBetaMeasure(beta=1.0, average='weighted', labels=None)
        return metrics

    @overrides
    def forward(
        self,
        tokens: Optional[TextFieldTensors] = None,
        texts: Optional[List[str]] = None,
        embeddings: Optional[torch.FloatTensor] = None,
        label: Optional[torch.IntTensor] = None,
    ) -> Dict[str, torch.Tensor]:

        if self._embedder is not None:
            device = self._get_prediction_device()
            embeddings = self._embedder.forward(tokens=tokens, texts=texts)

            if device != -1:
                embeddings = embeddings.to(device)

        if self._dropout:
            embeddings = self._dropout(embeddings)  # pylint: disable=not-callable

        if self._feedforward is not None:
            embeddings = self._feedforward(embeddings)

        logits = self._classification_layer(embeddings)
        output = {'logits': logits}

        if label is not None:
            output['loss'] = torch.nn.functional.cross_entropy(logits, label.long().view(-1))

        return output

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        logits = output_dict['logits']
        idx2label = self.vocab.get_index_to_token_vocabulary(self.label_namespace)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_p, top_indsx = probabilities.topk(1, dim=1)

        output = {'label': [idx2label.get(idx, str(idx)) for idxs in top_indsx.cpu().tolist() for idx in idxs]}

        return output
