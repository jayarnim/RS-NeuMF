from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from .LOSS import TaskLossFN
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
        lr, 
        lambda_, 
        task_type="bpr",
        col_user = DEFAULT_USER_COL,
        col_item = DEFAULT_ITEM_COL,
        col_label = DEFAULT_LABEL_COL,
        col_prediction = DEFAULT_PREDICTION_COL,
    ):
        super(Module, self).__init__()
        # device setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # global attr
        self.model = model.to(self.device)
        self.lr = lr
        self.lambda_ = lambda_
        self.task_type = task_type
        self.col_user = col_user
        self.col_item = col_item
        self.col_label = col_label
        self.col_prediction = col_prediction

        # Optimizer
        self.optimizer = optim.Adam(
            params=model.parameters(), 
            lr=self.lr, 
            weight_decay=lambda_
        )

        # Loss FN
        self.task_criterion = TaskLossFN(type=self.task_type)

        # gradient scaler setting
        self.scaler = GradScaler()

    def fit(
        self, 
        trn_loader, 
        val_loader, 
        n_epochs, 
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

        task_history = dict(
            trn=trn_task_loss_list,
            val=val_task_loss_list,
        )

        return task_history

    def predict(
        self, 
        tst_loader, 
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


        df_true = pd.DataFrame(
            {
                self.col_user: user_idx_list,
                self.col_item: item_idx_list,
                self.col_label: target_list,
            }
        )
        df_pred = pd.DataFrame(
            {
                self.col_user: user_idx_list,
                self.col_item: item_idx_list,
                self.col_prediction: pred_list,
            }
        )
        result = dict(
            true=df_true,
            pred=df_pred
        )

        return result

    def _train_epoch(
        self, 
        trn_loader, 
        n_epochs, 
        epoch, 
    ):
        self.model.train()

        epoch_task_loss = 0.0


        iter_obj = tqdm(
            iterable=trn_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} TRN"
        )

        # model
        for user_idx_batch, item_idx_batch, target_batch in iter_obj:
            # to gpu
            user_idx_batch = user_idx_batch.to(self.device)
            item_idx_batch = item_idx_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            kwargs = dict(
                user_idx_batch=user_idx_batch,
                item_idx_batch=item_idx_batch,
                target_batch=target_batch,
            )

            with autocast(self.device.type):
                # forward pass
                self.optimizer.zero_grad()
                batch_task_loss = self._batch(**kwargs)

            # accumulate loss
            epoch_task_loss += batch_task_loss.item()

            # backward pass of model
            self.scaler.scale(batch_task_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return epoch_task_loss/len(trn_loader)

    def _valid_epoch(
        self, 
        val_loader, 
        n_epochs,
        epoch,
    ):
        self.model.eval()

        epoch_task_loss = 0.0

        iter_obj = tqdm(
            iterable=val_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} VAL"
        )

        # forward pass of model
        with torch.no_grad():
            for user_idx_batch, item_idx_batch, target_batch in iter_obj:
                # to gpu
                user_idx_batch = user_idx_batch.to(self.device)
                item_idx_batch = item_idx_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                kwargs = dict(
                    user_idx_batch=user_idx_batch,
                    item_idx_batch=item_idx_batch,
                    target_batch=target_batch,
                )

                with autocast(self.device.type):
                    batch_task_loss = self._batch(**kwargs)

                # accumulate loss
                epoch_task_loss += batch_task_loss.item()

        return epoch_task_loss/len(val_loader)

    def _batch(
        self,
        user_idx_batch,        # [B_pos + B_neg]
        item_idx_batch,        # [B_pos + B_neg]
        target_batch,          # [B_pos + B_neg]
    ):
        # 예측
        preds = self.model(user_idx_batch, item_idx_batch)       # [B_total]

        # 손실 계산
        mask_pos = (target_batch==1)
        mask_neg = (target_batch==0)
        pred_pos = preds[mask_pos]                                              # [B]
        pred_neg = preds[mask_neg].view(pred_pos.size(0), -1)                   # [B, N]
        batch_task_loss = self.task_criterion.compute(pred_pos, pred_neg)

        return batch_task_loss