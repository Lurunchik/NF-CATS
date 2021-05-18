import enum

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PLACEHOLDER = '    '


class MetricAggregation(str, enum.Enum):
    WEIGHTED = 'weighted'
    MACRO = 'macro'
    MICRO = 'micro'


class Metric(str, enum.Enum):
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'f1-score'
    F0_5 = 'f0.5-score'
    ACCURACY = 'accuracy'


def pandas_classification_report(y_true, y_pred) -> pd.DataFrame:
    """
    Create a report with classification metrics (precision, recall, f-score, and accuracy)
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)

    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average='weighted'
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average='macro'
    )

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, average='micro'
    )

    _, _, f05, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, beta=0.5)
    _, _, avg_f05, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, beta=0.5, average='weighted')
    _, _, macro_f05, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, beta=0.5, average='macro')
    _, _, micro_f05, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, beta=0.5, average='micro')

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'f0.5-score', 'support']
    labels = sorted(set(y_true).union(y_pred))

    class_report_df = pd.DataFrame([precision, recall, f1, f05, support], columns=labels, index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df[PLACEHOLDER] = [PLACEHOLDER] * 5
    class_report_df[MetricAggregation.MACRO.value] = macro_precision, macro_recall, macro_f1, macro_f05, total
    class_report_df[MetricAggregation.WEIGHTED.value] = avg_precision, avg_recall, avg_f1, avg_f05, total
    class_report_df[MetricAggregation.MICRO.value] = micro_precision, micro_recall, micro_f1, micro_f05, total
    class_report_df[Metric.ACCURACY.value] = (
        accuracy_score(y_true, y_pred),
        PLACEHOLDER,
        PLACEHOLDER,
        PLACEHOLDER,
        PLACEHOLDER,
    )

    return class_report_df.T
