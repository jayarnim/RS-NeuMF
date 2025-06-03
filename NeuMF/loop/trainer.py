from tqdm import tqdm
import copy
from IPython.display import clear_output
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from .predictor import predict
from .loss import TaskLossFN
from .constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
    TASK_TYPE,
    METRIC_TYPE,
)
from MYUTILS import ranking


class Module(nn.Module):
    def __init__(
        self, 
        model, 
        trn_neg_per_pos_ratio: int,
        task_type: TASK_TYPE, 
        lr: float=1e-4, 
        lambda_: float=1e-3, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_label: str=DEFAULT_LABEL_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
    ):
        super(Module, self).__init__()
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.model = model.to(self.device)
        self.trn_neg_per_pos_ratio = trn_neg_per_pos_ratio
        self.task_type = task_type
        self.lr = lr
        self.lambda_ = lambda_
        self.col_user = col_user
        self.col_item = col_item
        self.col_label= col_label
        self.col_prediction = col_prediction

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
        loo_loader: torch.utils.data.dataloader.DataLoader, 
        n_epochs: int, 
        metric: METRIC_TYPE='ndcg',
        interval: int=10,
        patience: int=10,
        delta: float=1e-3,
    ):
        trn_task_loss_list = []
        val_task_loss_list = []

        counter = 0
        best_epoch = 0
        best_score = 0
        best_model_state = None

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print(f"EPOCH {epoch+1} START ---->>>>")

            # TRN
            trn_task_loss = self._train_epoch(trn_loader, n_epochs, epoch)
            trn_task_loss_list.append(trn_task_loss)
            print(f"TRN TASK LOSS: {trn_task_loss:.4f}")

            # VAL
            val_task_loss = self._valid_epoch(val_loader, n_epochs, epoch)
            val_task_loss_list.append(val_task_loss)
            print(f"VAL TASK LOSS: {val_task_loss:.4f}")

            # LOO
            if (epoch != 0) and (epoch % interval == 0):
                current_score = self._loo_epoch(loo_loader, metric)
                print(
                    f"LEAVE ONE OUT CURRENT SCORE: {current_score:.4f}",
                    f"BEST SCORE: {best_score:.4f}({best_epoch})",
                    sep='\t',
                )

                if current_score > best_score + delta:
                    best_epoch = epoch + 1
                    best_score = current_score
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    counter = 0
                else:
                    counter += 1
                
                if counter > patience:
                    break
                
            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        clear_output(wait=False)
        
        print(
            f"LEAVE ONE OUT BEST EPOCH: {best_epoch}",
            f"LEAVE ONE OUT BEST SCORE: {best_score:.4f}",
            sep="\n"
        )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        history = dict(
            trn=trn_task_loss_list,
            val=val_task_loss_list,
        )

        return history

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

    def _loo_epoch(self, loo_loader, metric):
        TRUE_COL_LIST = [self.col_user, self.col_item, self.col_label]
        PRED_COL_LIST = [self.col_user, self.col_item, self.col_prediction]

        result = predict(
            model=self.model, 
            tst_loader=loo_loader,
        )
        
        kwargs = dict(
                rating_true=result[TRUE_COL_LIST],
                rating_pred=result[PRED_COL_LIST],
                k=DEFAULT_K,
                metric=metric,
            )

        eval_score = ranking.metrics.eval_top_k(**kwargs)

        return eval_score

    def _batch(self, user_idx, item_idx, target):
        _, logit = self.model(user_idx, item_idx)
        task_loss = self.task_criterion.compute(logit, target)
        return task_loss

    def _run_fn_opt(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _return_fn_loss(self, task_loss, dataloader):
        if self.task_type=='bce':
            n_samples = len(dataloader) * (self.trn_neg_per_pos_ratio + 1)
            return task_loss / n_samples
        else:
            n_pairs = len(dataloader)
            return task_loss / n_pairs