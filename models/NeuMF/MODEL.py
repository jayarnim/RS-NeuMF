import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        model_name: str="NeuMF",
    ):
        super(Module, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.model_name = model_name

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._param_layers()

    def forward(self, user_idx, item_idx):
        logit = self._score(user_idx, item_idx)
        return logit

    def predict(self, user_idx, item_idx):
        with torch.no_grad():
            logit = self._score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def _score(self, user_idx, item_idx):
        gmf_out = self._gmf(user_idx, item_idx)
        mlp_out = self._mlp(user_idx, item_idx)

        concat = torch.cat(
            tensors=(gmf_out, mlp_out), 
            dim=-1
        )
        logit = self.logit_layers(concat).squeeze(-1)

        return logit

    def _gmf(self, user, item):
        gmf_user = self.embed_user_gmf(user)
        gmf_item = self.embed_item_gmf(item)
        gmf_out = gmf_user * gmf_item
        return gmf_out

    def _mlp(self, user_idx, item_idx):
        user_slice = self.embed_user_mlp(user_idx)
        item_slice = self.embed_item_mlp(item_idx)
        
        concat = torch.cat(
            tensors=(user_slice, item_slice), 
            dim=-1
        )
        mlp_out = self.mlp_layers(concat)

        return mlp_out

    def _param_layers(self):
        self.embed_user_gmf = nn.Embedding(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.embed_item_gmf = nn.Embedding(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )

        self.embed_user_mlp = nn.Embedding(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors*2,
            padding_idx=self.n_users,
        )
        self.embed_item_mlp = nn.Embedding(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors*2,
            padding_idx=self.n_items,
        )

        self.mlp_layers = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )
        
        self.logit_layers = nn.Linear(
            in_features=self.n_factors + self.hidden[-1],
            out_features=1,
        )

    def _generate_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 4)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_factors * 4}"
        assert CONDITION, ERROR_MESSAGE