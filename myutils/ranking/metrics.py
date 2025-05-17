from typing import Literal, List
import pandas as pd
from ..config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)
from ..msr.python_evaluation import (
    hit_ratio_at_k,
    precision_at_k, 
    recall_at_k,
    map_at_k, 
    ndcg_at_k, 
)


def eval_top_k(
    rating_true: pd.DataFrame,
    rating_pred: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_LABEL_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
    k: int=DEFAULT_K,
    metric: List[Literal['hr', 'precision', 'recall', 'map', 'ndcg']]=['hr', 'precision', 'recall', 'map', 'ndcg'],
):
    rating_true = (
        rating_true[rating_true[col_rating]==1]
        .drop_duplicates(subset=[col_user, col_item])
        .sort_values(by=col_user, ascending=True)
        )

    rating_pred = (
        rating_pred
        .drop_duplicates(subset=[col_user, col_item])
        .sort_values(by=[col_user, col_prediction], ascending=[True, False], kind='stable')
        .groupby(col_user)
        .head(k)
        )

    kwargs = dict(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        k=k,
    )

    hr_ = hit_ratio_at_k(**kwargs) if 'hr' in metric else None
    prec_ = precision_at_k(**kwargs) if 'precision' in metric else None
    rec_ = recall_at_k(**kwargs) if 'recall' in metric else None
    map_ = map_at_k(**kwargs) if 'map' in metric else None
    ndcg_ = ndcg_at_k(**kwargs) if 'ndcg' in metric else None

    result = dict(
        top_k=k,
        hit_ratio=hr_, 
        precision=prec_, 
        recall=rec_, 
        map=map_, 
        ndcg=ndcg_,
    )

    return result
