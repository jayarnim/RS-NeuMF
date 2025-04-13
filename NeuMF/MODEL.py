import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(self, n_users, n_items, n_factors, hidden, dropout, name="NeuMF"):
        super().__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.name = name

        # debugging error
        self._assert_arg_error()

        # generate layers
        self._layer_generator()

    def forward(self, user, item):
        return self._score(user, item)

    def predict(self, user, item):
        with torch.no_grad():
            logit = self._score(user, item)
            pred = torch.sigmoid(logit)
        return pred

    def _score(self, user, item):
        gmf_out = self._gmf(user, item)
        mlp_out = self._mlp(user, item)
        concat = torch.cat((gmf_out, mlp_out), dim=-1)
        logit = self.output_layer(concat).squeeze(-1)
        return logit

    def _gmf(self, user, item):
        gmf_user = self.embed_user_gmf(user)
        gmf_item = self.embed_item_gmf(item)
        gmf_out = gmf_user * gmf_item
        return gmf_out

    def _mlp(self, user, item):
        mlp_user = self.embed_user_mlp(user)
        mlp_item = self.embed_item_mlp(item)
        concat = torch.cat((mlp_user, mlp_item), dim=-1)
        mlp_out = self.mlp(concat)
        return mlp_out

    def _layer_generator(self):
        self.embed_user_gmf = nn.Embedding(self.n_users, self.n_factors)
        self.embed_item_gmf = nn.Embedding(self.n_items, self.n_factors)

        self.embed_user_mlp = nn.Embedding(self.n_users, self.n_factors*2)
        self.embed_item_mlp = nn.Embedding(self.n_items, self.n_factors*2)

        self.mlp = nn.Sequential(*list(self._generate_layers()))

        self.output_layer = nn.Linear(self.n_factors*2, 1)

    def _generate_layers(self):
        idx = 1
        while idx < len(self.hidden):
            yield nn.Linear(self.hidden[idx-1], self.hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 4)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_factors * 4}"
        assert CONDITION, ERROR_MESSAGE

        CONDITION = (self.hidden[-1] == self.n_factors)
        ERROR_MESSAGE = f"Final MLP layer must match input size: {self.n_factors}"
        assert CONDITION, ERROR_MESSAGE