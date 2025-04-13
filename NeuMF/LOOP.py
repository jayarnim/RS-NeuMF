from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from .LOSS import bpr
from config.constants import (
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
        n_epochs,
        col_user = DEFAULT_USER_COL,
        col_item = DEFAULT_ITEM_COL,
        col_label = DEFAULT_LABEL_COL,
        col_prediction = DEFAULT_PREDICTION_COL,
        ):
        super().__init__()
        # device setting
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # gradient scaler setting
        self.scaler = GradScaler()

        # global attr
        self.model = model.to(self.device)
        self.lr = lr
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.col_user = col_user
        self.col_item = col_item
        self.col_label = col_label
        self.col_prediction = col_prediction

        # optimizer
        self.optimizer = optim.Adam(
            params=model.parameters(), 
            lr=lr, 
            weight_decay=lambda_
            )

    def train(self, trn_loader, val_loader):
        trn_loss, val_loss = [], []

        for epoch in range(self.n_epochs):
            if epoch % 10 == 0:
                print(f"EPOCH {epoch+1} START ---->>>>")
            
            # TRN
            trn_epoch_loss = self._train_epoch(trn_loader=trn_loader, epoch=epoch)
            trn_loss.append(trn_epoch_loss)
            print(f"TRN LOSS: {trn_epoch_loss:.4f}")

            # VAL
            val_epoch_loss = self._valid_epoch(val_loader=val_loader, epoch=epoch)
            val_loss.append(val_epoch_loss)
            print(f"VAL LOSS: {val_epoch_loss:.4f}")

        history = dict(
            trn=trn_loss,
            val=val_loss
            )

        return history

    def predict(self, tst_loader):
        self.model.eval()

        users, items, labels, preds = [], [], [], []

        iter_obj = tqdm(
            iterable=tst_loader, 
            desc="TST"
            )

        for user_batch, item_batch, label_batch in iter_obj:
            # to gpu
            user_batch = user_batch.to(self.device)
            item_batch = item_batch.to(self.device)
            label_batch = label_batch.to(self.device)

            # pred
            with autocast(self.device.type):
                pred_batch = self.model.predict(user_batch, item_batch)

            # to cpu & save
            users.extend(user_batch.cpu().tolist())
            items.extend(item_batch.cpu().tolist())
            labels.extend(label_batch.cpu().tolist())
            preds.extend(pred_batch.cpu().tolist())

        df_true = pd.DataFrame(
            {
                self.col_user: users,
                self.col_item: items,
                self.col_label: labels,
                }
            )
        df_pred = pd.DataFrame(
            {
                self.col_user: users,
                self.col_item: items,
                self.col_prediction: preds,
                }
            )
        result = dict(
            true=df_true,
            pred=df_pred
        )

        return result

    def _train_epoch(self, trn_loader, epoch):
        self.model.train()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=trn_loader, 
            desc=f"EPOCH {epoch+1}/{self.n_epochs} TRN"
            )

        for user_batch, pos_batch, neg_batch in iter_obj:
            # to gpu
            user_batch = user_batch.to(self.device)
            pos_batch = pos_batch.to(self.device)
            neg_batch = neg_batch.to(self.device)

            with autocast(self.device.type):
                # Forward Pass
                self.optimizer.zero_grad()
                pos_score = self.model(user_batch, pos_batch)
                neg_score = self.model(user_batch, neg_batch)

                # Calculate Loss
                batch_loss = bpr(pos_score, neg_score)
                epoch_loss += batch_loss.item()

            # Back Propagation
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return epoch_loss/len(trn_loader)

    def _valid_epoch(self, val_loader, epoch):
        self.model.eval()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=val_loader, 
            desc=f"EPOCH {epoch+1}/{self.n_epochs} VAL"
            )

        with torch.no_grad():
            for user_batch, pos_batch, neg_batch in iter_obj:
                # to gpu
                user_batch = user_batch.to(self.device)
                pos_batch = pos_batch.to(self.device)
                neg_batch = neg_batch.to(self.device)

                with autocast(self.device.type):
                    # Forward Pass
                    pos_score = self.model(user_batch, pos_batch)
                    neg_score = self.model(user_batch, neg_batch)

                    # Calculate Loss
                    batch_loss = bpr(pos_score, neg_score)
                    epoch_loss += batch_loss.item()

        return epoch_loss/len(val_loader)