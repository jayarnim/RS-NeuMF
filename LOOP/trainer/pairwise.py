from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast


class PairwiseTrainer:
    def __init__(
        self,
        model,
        task_fn,
        lr: float=1e-2, 
        lambda_: float=1e-4, 
    ):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        self.model = model
        self.task_fn = task_fn
        
        # optimizer
        kwargs = dict(
            params=self.model.parameters(), 
            lr=lr, 
            weight_decay=lambda_,
        )
        self.optimizer = optim.Adam(**kwargs)

        # gradient scaler setting
        self.scaler = GradScaler()

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        epoch: int,
        n_epochs: int,
    ):
        trn_task_loss = self._epoch_trn_loop(trn_loader, epoch, n_epochs)
        val_task_loss = self._epoch_val_loop(val_loader, epoch, n_epochs)
        return trn_task_loss, val_task_loss

    def _epoch_trn_loop(
        self,
        trn_loader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        n_epochs: int,
    ):
        self.model.train()

        epoch_task_loss = 0.0

        iter_obj = tqdm(
            iterable=trn_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} TRN"
        )

        for user_idx, pos_idx, neg_idx in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(self.device),
                pos_idx=pos_idx.to(self.device),
                neg_idx=neg_idx.to(self.device),
            )

            # forward pass
            with autocast(self.device.type):
                batch_task_loss = self._batch_loop(**kwargs)

            # accumulate loss
            epoch_task_loss += batch_task_loss.item()

            # backward pass
            self._run_fn_opt(batch_task_loss)

        return epoch_task_loss / len(trn_loader)

    def _epoch_val_loop(        
        self,
        val_loader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        n_epochs: int,
    ):
        self.model.eval()

        epoch_task_loss = 0.0

        iter_obj = tqdm(
            iterable=val_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} VAL"
        )

        with torch.no_grad():
            for user_idx, pos_idx, neg_idx in iter_obj:
                # to gpu
                kwargs = dict(
                    user_idx=user_idx.to(self.device),
                    pos_idx=pos_idx.to(self.device),
                    neg_idx=neg_idx.to(self.device),
                )

                # forward pass
                with autocast(self.device.type):
                    batch_task_loss = self._batch_loop(**kwargs)

                # accumulate loss
                epoch_task_loss += batch_task_loss.item()

        return epoch_task_loss / len(val_loader)

    def _batch_loop(self, user_idx, pos_idx, neg_idx):
        pos_logit = self.model(user_idx, pos_idx)
        neg_logit = self.model(user_idx, neg_idx)
        batch_task_loss = self.task_fn(pos_logit, neg_logit)
        return batch_task_loss

    def _run_fn_opt(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()