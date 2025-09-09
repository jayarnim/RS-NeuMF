from typing import Optional
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    LOADING_TYPE,
    FILTER_BY,
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
        batch_size: list=[128, 128, 1, 1],
        max_hist: Optional[int]=None,
        shuffle: bool=True,
        seed: int=42,
    ):
        # split original data
        kwargs = dict(
            filter_by=filter_by,
            trn_val_tst_ratio=trn_val_tst_ratio,
            seed=seed,
        )
        split_list = self._split(**kwargs)

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
        user_item_binary_matrix_np = self._user_item_binary_matrix(split_list[0])

        # user-item interaction matrix: ndarray -> tensor
        interactions = torch.from_numpy(user_item_binary_matrix_np)

        # generate histories per user: {user: List}
        kwargs = dict(
            user_item_binary_matrix=user_item_binary_matrix_np,
            max_hist=max_hist,
        )
        histories = self._histories(**kwargs)

        return loaders, interactions, histories

    def _user_item_binary_matrix(
        self,
        data: pd.DataFrame,
    ):
        # generate user-item interaction matrix
        kwargs = dict(
            shape=(self.n_users+1, self.n_items+1), 
            dtype=np.int32,
        )
        user_item_matrix = np.zeros(**kwargs)

        # mark interactions in matrix
        user_indices = data[self.col_user].values
        item_indices = data[self.col_item].values
        user_item_matrix[user_indices, item_indices] = 1

        return user_item_matrix

    def _histories(
        self, 
        user_item_binary_matrix: np.ndarray, 
        max_hist: Optional[int]=None,
    ):
        tfidf_dict = self._tfidf(user_item_binary_matrix) if max_hist is not None else None

        pos_per_user_list = []

        for user in range(self.n_users):
            # search interacted item ids
            user_row = user_item_binary_matrix[user]
            items = np.where(user_row > 0)[0]

            # if no interaction -> padding idx
            if len(items) == 0:
                kwargs = dict(
                    data=[self.n_items], 
                    dtype=torch.long,          
                )
                item_ids = torch.tensor(**kwargs)
            else:
                kwargs = dict(
                    data=items, 
                    dtype=torch.long,
                )
                item_ids = torch.tensor(**kwargs)

                # select based on tf-idf score
                if max_hist is not None and len(items) > max_hist:
                    # search tf-idf score
                    kwargs = dict(
                        data=[tfidf_dict.get((user, item), 0.0) for item in items],
                        dtype=torch.float32,
                    )
                    scores = torch.tensor(**kwargs)

                    # slice top-k items
                    kwargs = dict(
                        input=scores, 
                        k=max_hist,   
                    )
                    topk_vals, topk_indices = torch.topk(**kwargs)
                    item_ids = item_ids[topk_indices]

            pos_per_user_list.append(item_ids)

        kwargs = dict(
            object=pos_per_user_list,
            dtype=object,
        )
        pos_per_user_np = np.array(**kwargs)

        return pos_per_user_np

    def _tfidf(
        self, 
        user_item_binary_matrix: np.ndarray,
    ):
        # drop padding idx
        user_item_binary_matrix_unpadded = user_item_binary_matrix[:-1, :-1]

        # compute tfidf
        tfidf = TfidfTransformer(norm=None)
        tfidf_matrix = tfidf.fit_transform(user_item_binary_matrix_unpadded)

        # sparse matrix -> dict: {(u_idx, i_idx): tf-idf score}
        tfidf_dict = {}
        rows, cols = tfidf_matrix.nonzero()
        for row, col in zip(rows, cols):
            tfidf_dict[(row, col)] = tfidf_matrix[row, col]

        return tfidf_dict

    def _split(
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