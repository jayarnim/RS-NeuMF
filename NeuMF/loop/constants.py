from typing import Literal
from MYUTILS.config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
    SEED,
)

TASK_TYPE = Literal['bce', 'bpr', 'climf']
METRIC_TYPE = Literal['hr', 'precision', 'recall', 'map', 'ndcg']