import torch
import torch.nn as nn
from . import gmf, ncf


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
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
        self.hidden = hidden
        self.dropout = dropout

        # debugging args error
        self._assert_arg_error()

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
        return self._score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            _, logit = self._score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def _score(self, user_idx, item_idx):
        pred_vector_gmf, _ = self.gmf(user_idx, item_idx)
        pred_vector_ncf, _ = self.ncf(user_idx, item_idx)

        pred_vector = torch.cat(
            tensors=(pred_vector_gmf, pred_vector_ncf), 
            dim=-1
        )

        logit = self.logit_layer(pred_vector).squeeze(-1)

        return pred_vector, logit

    def _init_layers(self):
        self.gmf = gmf(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors // 2,
        )
        self.ncf = ncf(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden,
            dropout=self.dropout,
        )
        self.logit_layer = nn.Linear(
            in_features=self.n_factors//2 + self.hidden[-1],
            out_features=1,
        )

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 2)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_factors * 2}"
        assert CONDITION, ERROR_MESSAGE