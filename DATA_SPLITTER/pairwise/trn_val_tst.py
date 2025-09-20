from typing import Optional
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from torch.nn.utils.rnn import pad_sequence
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    LOADING_TYPE,
    FILTER_BY,
    SEED,
)
from ..utils.python_splitters import python_stratified_split
from .negative_sampling_dataloader import PairwiseNegativeSamplingDataLoader
from ..pointwise.negative_sampling_dataloader import PointwiseNegativeSamplingDataLoader
from .curriculum_dataloader import PairwiseCurriculumDataLoader
from ..pointwise.curriculum_dataloader import PointwiseCurriculumDataLoader
from .userpair_dataloader import PairwiseUserpairDataLoader
from ..pointwise.userpair_dataloader import PointwiseUserpairDataLoader
from .phase_dataloader import PairwisePhaseDataLoader
from ..pointwise.phase_dataloader import PointwisePhaseDataLoader


class DataSplitter:
    def __init__(
        self, 
        origin: pd.DataFrame,
        n_users: int, 
        n_items: int,
        n_phases: Optional[int]=None,
        col_user: str=DEFAULT_USER_COL, 
        col_item: str=DEFAULT_ITEM_COL,
        loading_type: LOADING_TYPE="general",
    ):
        self.origin = origin
        self.n_users = n_users
        self.n_items = n_items
        self.n_phases = n_phases
        self.col_user = col_user
        self.col_item = col_item

        kwargs = dict(
            origin=self.origin,
            col_user=self.col_user,
            col_item=self.col_item,
        )
        if loading_type=="general":
            self.pairwise = PairwiseNegativeSamplingDataLoader(**kwargs)
            self.pointwise = PointwiseNegativeSamplingDataLoader(**kwargs)
        elif loading_type=="curriculum":
            self.pairwise = PairwiseCurriculumDataLoader(**kwargs)
            self.pointwise = PointwiseCurriculumDataLoader(**kwargs)
        elif loading_type=="userpair":
            self.pairwise = PairwiseUserpairDataLoader(**kwargs)
            self.pointwise = PointwiseUserpairDataLoader(**kwargs)
        elif loading_type=="phase":
            self.pairwise = PairwisePhaseDataLoader(**kwargs, n_phases=self.n_phases)
            self.pointwise = PointwisePhaseDataLoader(**kwargs, n_phases=self.n_phases)
        else:
            raise TypeError(f"Invalid loading_type: {loading_type}")

    def get(
        self, 
        filter_by: FILTER_BY="user",
        trn_val_tst_ratio: list=[0.8, 0.1, 0.1],
        neg_per_pos: list=[1, 1, 99, 99],
        batch_size: list=[256, 256, 256, 1000],
        max_hist: Optional[int]=None,
        shuffle: bool=True,
        seed: int=SEED,
    ):
        # split original data
        kwargs = dict(
            filter_by=filter_by,
            trn_val_tst_ratio=trn_val_tst_ratio,
            seed=seed,
        )
        trn, val, tst, loo = self._data_splitter(**kwargs)
        split_list = [trn, val, tst, loo]

        # generate data loaders
        loaders = []
        zip_obj = zip(split_list, neg_per_pos, batch_size)

        for idx, (split, split_neg_per_pos, split_batch) in enumerate(zip_obj):
            kwargs = dict(
                data=split, 
                neg_per_pos=split_neg_per_pos, 
                batch_size=split_batch, 
                shuffle=shuffle,
            )
            if (idx==0) or (idx==1):
                loader = self.pairwise.get(**kwargs)
            if (idx==2) or (idx==3):
                loader = self.pointwise.get(**kwargs)
            loaders.append(loader)

        # generate user-item interaction matrix
        interactions = self._interactions_generator(trn)

        # generate histories per user: {user: List}
        kwargs = dict(
            interactions=interactions,
            max_hist=max_hist,
        )
        histories = self._histories_generator(**kwargs)

        return loaders, interactions, histories

    def _interactions_generator(self, data):
        kwargs = dict(
            size=(self.n_users + 1, self.n_items + 1),
            dtype=torch.int32,
        )
        interactions = torch.zeros(**kwargs)

        kwargs = dict(
            data=data[self.col_user].values, 
            dtype=torch.long,
        )
        user_indices = torch.tensor(**kwargs)
        kwargs = dict(
            data=data[self.col_item].values, 
            dtype=torch.long,
        )
        item_indices = torch.tensor(**kwargs)

        interactions[user_indices, item_indices] = 1

        return interactions

    def _histories_generator(
        self, 
        interactions: torch.Tensor, 
        max_hist: Optional[int]=None,
    ):
        tfidf_dict = self._tfidf(interactions) if max_hist is not None else None
        pos_per_user_list = []

        for user in range(self.n_users):
            # user row (interaction 벡터)
            user_row = interactions[user]

            # interacted item indices
            items = torch.nonzero(user_row, as_tuple=False).squeeze(-1)

            # interaction X -> padding idx
            if items.numel() == 0:
                item_ids = torch.tensor([self.n_items], dtype=torch.long)
            # interaction O
            else:
                item_ids = items
                # top-k based on tf-idf score
                if max_hist is not None and len(items) > max_hist:
                    # scores
                    kwargs = dict(
                        data=[tfidf_dict.get((int(user), int(item)), 0.0) for item in items],
                        dtype=torch.float32,
                    )
                    scores = torch.tensor(**kwargs)
                    # top-k idx selection
                    top_k_vals, top_k_indices = torch.topk(scores, k=max_hist)
                    item_ids = item_ids[top_k_indices]

            pos_per_user_list.append(item_ids)

        # padding
        kwargs = dict(
            sequences=pos_per_user_list, 
            batch_first=True, 
            padding_value=self.n_items,
        )
        pos_per_user_padded = pad_sequence(**kwargs)

        return pos_per_user_padded

    def _tfidf(
        self, 
        interactions: torch.Tensor,
    ):
        # drop padding idx
        interactions_unpadded = interactions[:-1, :-1]

        # compute tfidf
        tfidf = TfidfTransformer(norm=None)
        tfidf_matrix = tfidf.fit_transform(interactions_unpadded)

        # sparse matrix -> dict: {(u_idx, i_idx): tf-idf score}
        tfidf_dict = {}
        rows, cols = tfidf_matrix.nonzero()
        for row, col in zip(rows, cols):
            tfidf_dict[(row, col)] = tfidf_matrix[row, col]

        return tfidf_dict

    def _data_splitter(
        self,
        filter_by: FILTER_BY,
        trn_val_tst_ratio: list,
        seed: int,
    ):
        # for leave one out data set
        loo = (
            self.origin
            .groupby(self.col_user)
            .sample(n=1, random_state=seed)
            .sort_values(by=self.col_user)
            .reset_index(drop=True)
        )

        # for trn, val, tst data set
        trn_val_tst = (
            self.origin[~self.origin[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)
            .isin(set(loo[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)))]
            .reset_index(drop=True)
        )

        # trn_val_tst -> [trn, val, tst]
        kwargs = dict(
            data=trn_val_tst,
            filter_by=filter_by,
            ratio=trn_val_tst_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=seed,
        )
        trn, val, tst = python_stratified_split(**kwargs)

        return trn, val, tst, loo