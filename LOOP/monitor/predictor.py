from tqdm import tqdm
import pandas as pd
import torch
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
)


def predict(
    model,
    dataloader: torch.utils.data.dataloader.DataLoader, 
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_label: str=DEFAULT_LABEL_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
):
    # device setting
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(DEVICE)

    # cpu -> gpu
    model = model.to(device)

    # evaluation
    model.eval()

    # to save result
    user_idx_list = []
    item_idx_list = []
    label_list = []
    pred_list = []

    iter_obj = tqdm(
        iterable=dataloader, 
        desc=f"TST"
    )

    # mini-batch predict loop
    for user_idx, item_idx, label in iter_obj:
        # to gpu
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        label = label.to(device)

        # predict
        kwargs = dict(
            user_idx=user_idx,
            item_idx=item_idx,
        )
        with torch.no_grad():
            preds = model.predict(**kwargs)

        # to cpu & save
        user_idx_list.extend(user_idx.cpu().tolist())
        item_idx_list.extend(item_idx.cpu().tolist())
        label_list.extend(label.cpu().tolist())
        pred_list.extend(preds.cpu().tolist())

    # list -> df
    result = pd.DataFrame(
        {
            col_user: user_idx_list,
            col_item: item_idx_list,
            col_label: label_list,
            col_prediction: pred_list,
        }
    )

    return result