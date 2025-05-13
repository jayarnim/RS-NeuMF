from tqdm import tqdm
from IPython.display import clear_output
from typing import Literal
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from .loss import TaskLossFN
from MYUTILS.config.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
)


class Module(nn.Module):
    def __init__(
        self, 
        model, 
        trn_neg_per_pos_ratio: int,
        task_type: Literal['bce', 'bpr', 'climf'], 
        lr: float=1e-4, 
        lambda_: float=1e-2, 
    ):
        super(Module, self).__init__()
        # device setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # global attr
        self.model = model.to(self.device)
        self.trn_neg_per_pos_ratio = trn_neg_per_pos_ratio
        self.task_type = task_type
        self.lr = lr
        self.lambda_ = lambda_

        # Loss FN
        self.task_criterion = TaskLossFN(
            fn_type=self.task_type,
        )

        # Optimizer
        self.optimizer = optim.Adam(
            params=model.parameters(), 
            lr=self.lr, 
            weight_decay=lambda_,
        )

        # gradient scaler setting
        self.scaler = GradScaler()

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        n_epochs: int, 
    ):
        trn_task_loss_list = []
        val_task_loss_list = []

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print(f"EPOCH {epoch+1} START ---->>>>")

            # Train
            trn_task_loss = self._train_epoch(trn_loader, n_epochs, epoch)
            trn_task_loss_list.append(trn_task_loss)
            print(
                f"TRN TASK LOSS: {trn_task_loss:.4f}"
            )

            # Validation
            val_task_loss = self._valid_epoch(val_loader, n_epochs, epoch)
            val_task_loss_list.append(val_task_loss)
            print(
                f"VAL TASK LOSS: {val_task_loss:.4f}"
            )

        history = dict(
            trn=trn_task_loss_list,
            val=val_task_loss_list,
        )

        return history

    def predict(
        self, 
        tst_loader: torch.utils.data.dataloader.DataLoader, 
        col_user = DEFAULT_USER_COL,
        col_item = DEFAULT_ITEM_COL,
        col_label = DEFAULT_LABEL_COL,
        col_prediction = DEFAULT_PREDICTION_COL,
    ):
        self.model.eval()

        user_idx_list, item_idx_list, target_list, pred_list = [], [], [], []

        iter_obj = tqdm(
            iterable=tst_loader, 
            desc=f"TST"
        )

        for user_idx_batch, item_idx_batch, target_batch in iter_obj:
            # to gpu
            user_idx_batch = user_idx_batch.to(self.device)
            item_idx_batch = item_idx_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            kwargs = dict(
                user_idx=user_idx_batch,
                item_idx=item_idx_batch,
            )

            # predict
            with autocast(self.device.type):
                preds_batch = self.model.predict(**kwargs)

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

    def _train_epoch(self, trn_loader, n_epochs, epoch):
        self.model.train()

        epoch_task_loss = 0.0


        iter_obj = tqdm(
            iterable=trn_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} TRN"
        )

        for user_idx_batch, item_idx_batch, target_batch in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx_batch.to(self.device),
                item_idx=item_idx_batch.to(self.device),
                target=target_batch.to(self.device),
            )

            # forward pass
            with autocast(self.device.type):
                self.optimizer.zero_grad()
                batch_task_loss = self._batch(**kwargs)

            # accumulate loss
            epoch_task_loss += batch_task_loss.item()

            # backward pass
            self._run_fn_opt(batch_task_loss)

        return self._return_fn_loss(
            task_loss=epoch_task_loss, 
            dataloader=trn_loader,
        )

    def _valid_epoch(self, val_loader, n_epochs, epoch):
        self.model.eval()

        epoch_task_loss = 0.0

        iter_obj = tqdm(
            iterable=val_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} VAL"
        )

        with torch.no_grad():
            for user_idx_batch, item_idx_batch, target_batch in iter_obj:
                # to gpu
                kwargs = dict(
                    user_idx=user_idx_batch.to(self.device),
                    item_idx=item_idx_batch.to(self.device),
                    target=target_batch.to(self.device),
                )

                # forward pass
                with autocast(self.device.type):
                    batch_task_loss = self._batch(**kwargs)

                # accumulate loss
                epoch_task_loss += batch_task_loss.item()

        return self._return_fn_loss(
            task_loss=epoch_task_loss, 
            dataloader=val_loader,
        )

    def _batch(self, user_idx, item_idx, target):
        logits = self.model(user_idx, item_idx)
        task_loss = self.task_criterion.compute(logits, target)
        return task_loss

    def _run_fn_opt(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _return_fn_loss(self, task_loss, dataloader):
        if self.task_type=='bce':
            return task_loss/(len(dataloader) * (self.trn_neg_per_pos_ratio + 1))
        else:
            return task_loss/len(dataloader)