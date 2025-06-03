from tqdm import tqdm
import pandas as pd
import torch
from .constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
)


def predict(
    model,
    tst_loader: torch.utils.data.dataloader.DataLoader, 
    col_user: str=DEFAULT_USER_COL,
    col_item: str=DEFAULT_ITEM_COL,
    col_label: str=DEFAULT_LABEL_COL,
    col_prediction: str=DEFAULT_PREDICTION_COL,
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(DEVICE)
    
    model = model.to(device)
    
    model.eval()

    user_idx_list, item_idx_list, target_list, pred_list = [], [], [], []

    iter_obj = tqdm(
        iterable=tst_loader, 
        desc=f"TST"
    )

    for user_idx_batch, item_idx_batch, target_batch in iter_obj:
        # to gpu
        user_idx_batch = user_idx_batch.to(device)
        item_idx_batch = item_idx_batch.to(device)
        target_batch = target_batch.to(device)

        kwargs = dict(
            user_idx=user_idx_batch,
            item_idx=item_idx_batch,
        )

        # predict
        preds_batch = model.predict(**kwargs)

        # to cpu & save
        user_idx_list.extend(user_idx_batch.cpu().tolist())
        item_idx_list.extend(item_idx_batch.cpu().tolist())
        target_list.extend(target_batch.cpu().tolist())
        pred_list.extend(preds_batch.cpu().tolist())


    result = pd.DataFrame(
        {
            col_user: user_idx_list,
            col_item: item_idx_list,
            col_label: target_list,
            col_prediction: pred_list,
        }
    )

    return result