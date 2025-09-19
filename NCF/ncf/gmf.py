import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
    ):
        super(Module, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)
        logit = self.score(user_idx, item_idx)
        return logit

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)

        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)

        return pred

    def score(self, user_idx, item_idx):
        pred_vector = self.gmf(user_idx, item_idx)
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def gmf(self, user_idx, item_idx):
        user_slice = self.user_embed(user_idx)
        item_slice = self.item_embed(item_idx)
        pred_vector = user_slice * item_slice
        return pred_vector

    def _init_layers(self):
        self.user_embed = nn.Embedding(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.item_embed = nn.Embedding(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.logit_layer = nn.Linear(
            in_features=self.n_factors,
            out_features=1,
        )