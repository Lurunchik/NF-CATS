import math
import random
import warnings
from typing import Iterable, Iterator, List, Sequence, TypeVar

import numpy as np
from allennlp.data import Instance
from allennlp.data.samplers import BatchSampler

Example = TypeVar('Example')


def iterate_random_batches(data: Iterable[Example], batch_size: int) -> Iterator[List[Example]]:
    """
    Uniformly sample random batches of the same size from the data indefinitely (without replacement)

    Args:
        data: Iterable with data examples
        batch_size: Batch size to use for all batches

    Returns:
        Iterator over batches
    """
    population = list(data)

    if len(population) < batch_size:
        raise ValueError(f'Population size {len(population)} must be greater than batch size {batch_size}')

    seen: List[Example] = []
    while True:
        random.shuffle(population)

        num_full, num_trailing = divmod(len(population), batch_size)

        for start in range(0, num_full * batch_size, batch_size):
            batch = population[start : start + batch_size]
            seen.extend(batch)
            yield batch

        if num_trailing > 0:
            trailing = population[-num_trailing:]
            random.shuffle(seen)
            num_missing = batch_size - num_trailing
            seen, population = seen[:num_missing], seen[num_missing:] + trailing
            yield trailing + seen
        else:
            population = seen
            seen = []


@BatchSampler.register('balanced')
class BalancedBatchSampler(BatchSampler):
    def __init__(self, num_classes_per_batch: int = 8, num_examples_per_class: int = 32) -> None:

        self._num_classes_per_batch = num_classes_per_batch
        self._num_examples_per_class = num_examples_per_class

    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        labels = np.array([instance.fields['label'].label for instance in instances])
        unique_labels, counts = np.unique(labels, return_counts=True)
        unique_labels = list(unique_labels)

        min_number_of_examples = counts.min()
        if self._num_examples_per_class > min_number_of_examples:
            warnings.warn(
                f'Setting `num_examples_per_class` to {min_number_of_examples}, '
                f'since there are classes with less examples than {self._num_examples_per_class}'
            )
            self._num_examples_per_class = min_number_of_examples

        num_unique_classes = len(unique_labels)
        if self._num_classes_per_batch > num_unique_classes:
            warnings.warn(
                f'Setting `num_classes_per_batch` to {num_unique_classes}, '
                f'since there are only {num_unique_classes} classes (not {self._num_classes_per_batch})'
            )
            self._num_classes_per_batch = num_unique_classes

        class_examples_generators = {
            label: iterate_random_batches(np.flatnonzero(labels == label), self._num_examples_per_class)
            for label in unique_labels
        }

        batch_classes_generator = iterate_random_batches(unique_labels, self._num_classes_per_batch)

        for _ in range(self.get_num_batches(instances)):
            batch = []
            chosen_labels = next(batch_classes_generator)  # noqa # pylint: disable=stop-iteration-return
            for label in chosen_labels:
                batch.extend(next(class_examples_generators[label]))  # noqa # pylint: disable=stop-iteration-return
            yield batch

    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        return math.ceil(len(instances) / (self._num_classes_per_batch * self._num_examples_per_class))
