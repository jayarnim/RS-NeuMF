import pandas as pd
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)
from .msr.python_evaluation import (
    hit_ratio_at_k,
    precision_at_k, 
    recall_at_k,
    map_at_k, 
    ndcg_at_k, 
)

def _sep_true_pred(
    result: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_RATING_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
    k: int=DEFAULT_K,
):
    TRUE_COL_LIST = [col_user, col_item, col_rating]
    PRED_COL_LIST = [col_user, col_item, col_prediction]

    rating_true = (
        result[TRUE_COL_LIST]
        [result[col_rating]==1]
        .sort_values(by=col_user, ascending=True)
    )

    rating_pred = (
        result[PRED_COL_LIST]
        .sort_values(by=[col_user, col_prediction], ascending=[True, False], kind='stable')
        .groupby(col_user)
        .head(k)
    )

    return rating_true, rating_pred


def _metric_dict(**kwargs):
    hr_k = hit_ratio_at_k(**kwargs)
    prec_k = precision_at_k(**kwargs)
    rec_k = recall_at_k(**kwargs)
    map_k = map_at_k(**kwargs)
    ndcg_k = ndcg_at_k(**kwargs)

    return dict(
        top_k=kwargs.get('k', DEFAULT_K),
        hit_ratio=hr_k, 
        precision=prec_k, 
        recall=rec_k, 
        map=map_k, 
        ndcg=ndcg_k,
    )


def top_k(
    result: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_RATING_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
    k: int=DEFAULT_K,
):
    kwargs = dict(
        result=result,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        k=k,
    )
    rating_true, rating_pred = _sep_true_pred(**kwargs)

    kwargs = dict(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        k=k,
    )
    return _metric_dict(**kwargs)

def top_k_loop(
    result: pd.DataFrame,
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_rating: str=DEFAULT_RATING_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
):
    top_k_list = [5, 10, 15, 20, 25, 50, 100]
    eval_list = []

    for TOP_K in top_k_list:
        kwargs = dict(
            result=result,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction=col_prediction,
            k=TOP_K,
        )
        eval = top_k(**kwargs)
        eval_list.append(eval)

    return pd.DataFrame(eval_list)