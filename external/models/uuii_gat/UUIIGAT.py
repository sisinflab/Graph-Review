"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from ast import literal_eval as make_tuple

from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import random

from .pointwise_pos_neg_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .UUIIGATModel import UUIIGATModel

from torch_sparse import SparseTensor


class UUIIGAT(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_a", "a", "a", 0.1, float, None),
            ("_b", "b", "b", 0.1, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_batch_eval", "batch_eval", "batch_eval", 512, int, None),
            ("_weight_size", "weight_size", "weight_size", "(64,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_heads", "heads", "heads", "(1,)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_message_dropout", "message_dropout", "message_dropout", 0.1, float, None),
            ("_n_uu", "n_uu", "n_uu", 2, int, None),
            ("_n_ii", "n_ii", "n_ii", 2, int, None),
            ("_sim_uu", "sim_uu", "sim_uu", 'dot', str, None),
            ("_loader", "loader", "loader", 'SentimentInteractionsTextualAttributesUUII', str, None)
        ]
        self.autoset_params()

        self._sim_ii = self._sim_uu

        np.random.seed(self._seed)
        random.seed(self._seed)

        self._sampler = Sampler(self._batch_size, self._data.transactions)

        self.df_val_rat = pd.DataFrame(columns=['user', 'item', 'rating'])
        self.df_test_rat = pd.DataFrame(columns=['user', 'item', 'rating'])

        # create dataframes for validation and test set with user-item-rating
        idx = 0
        for k, v in data.val_dict.items():
            for kk, vv in v.items():
                self.df_val_rat = pd.concat([self.df_val_rat,
                                             pd.DataFrame({'user': k, 'item': int(kk), 'rating': vv}, index=[idx])])
                idx += 1

        idx = 0
        for k, v in data.test_dict.items():
            for kk, vv in v.items():
                self.df_test_rat = pd.concat([self.df_test_rat,
                                              pd.DataFrame({'user': k, 'item': int(kk), 'rating': vv}, index=[idx])])
                idx += 1

        self.df_val_rat = self.df_val_rat.astype({'user': int, 'item': int, 'rating': float})
        self.df_test_rat = self.df_test_rat.astype({'user': int, 'item': int, 'rating': float})

        self.df_val_rat['user'] = self.df_val_rat['user'].map(data.public_users)
        self.df_val_rat['item'] = self.df_val_rat['item'].map(data.public_items)
        self.df_test_rat['user'] = self.df_test_rat['user'].map(data.public_users)
        self.df_test_rat['item'] = self.df_test_rat['item'].map(data.public_items)

        # remove interactions whose user and/or item is not in the training set
        self.df_val_rat = self.df_val_rat[self.df_val_rat['user'] <= self._num_users - 1]
        self.df_test_rat = self.df_test_rat[self.df_test_rat['user'] <= self._num_users - 1]
        self.df_val_rat = self.df_val_rat[self.df_val_rat['item'] <= self._num_items - 1]
        self.df_test_rat = self.df_test_rat[self.df_test_rat['item'] <= self._num_items - 1]

        self._n_layers = len(self._weight_size)

        self._side_edge_textual = self._data.side_information.SentimentInteractionsTextualAttributesUUII
        sim = dict(self._side_edge_textual.object.get_features())
        if 'uu_' + self._sim_uu in list(sim.keys()):
            uu_sparse = sim['uu_' + self._sim_uu]
            rows, cols = uu_sparse.nonzero()
            sim_uu = SparseTensor(row=torch.tensor(rows, dtype=torch.int64), col=torch.tensor(cols, dtype=torch.int64))
        else:
            raise KeyError(f'uu_{self._sim_uu} similarity matrix not implemented!')
        if 'ii_' + self._sim_ii in list(sim.keys()):
            ii_sparse = sim['ii_' + self._sim_ii]
            rows, cols = ii_sparse.nonzero()
            sim_ii = SparseTensor(row=torch.tensor(rows, dtype=torch.int64), col=torch.tensor(cols, dtype=torch.int64))
        else:
            raise KeyError(f'ii_{self._sim_ii} similarity matrix not implemented!')

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))

        self._model = UUIIGATModel(
            num_users=self._num_users,
            num_items=self._num_items,
            num_uu_layers=self._n_uu,
            num_ii_layers=self._n_ii,
            sim_ii=sim_ii,
            sim_uu=sim_uu,
            alpha=self._a,
            beta=self._b,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            weight_size=self._weight_size,
            n_layers=self._n_layers,
            heads=self._heads,
            message_dropout=self._message_dropout,
            adj=self.adj,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "UUIIGAT" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        row, col = self._data.sp_i_train.nonzero()
        ratings = self._data.sp_i_train_ratings.data
        edge_index = np.array([row, col, ratings]).transpose()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0

            np.random.shuffle(edge_index)
            edge_index = edge_index.astype(int)

            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(edge_index):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_test = []
        predictions_val = []
        gu, gi, gus, gis = self._model.propagate_embeddings()
        val_len = len(self.df_val_rat)
        with tqdm(total=int(val_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, val_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, val_len)
                current_df = self.df_val_rat[offset:offset_stop]
                guf = (1 - self._model.alpha) * gu[current_df['user'].tolist()] + \
                      self._model.alpha * gus[current_df['user'].tolist()]
                gif = (1 - self._model.beta) * gi[current_df['item'].tolist()] + \
                      self._model.beta * gis[current_df['item'].tolist()]
                p = self._model.predict(guf, gif, current_df['user'].tolist(), current_df['item'].tolist())
                predictions_val += p.detach().cpu().numpy().tolist()
                t.update()
        test_len = len(self.df_test_rat)
        with tqdm(total=int(test_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, test_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, test_len)
                current_df = self.df_test_rat[offset:offset_stop]
                guf = (1 - self._model.alpha) * gu[current_df['user'].tolist()] + \
                      self._model.alpha * gus[current_df['user'].tolist()]
                gif = (1 - self._model.beta) * gi[current_df['item'].tolist()] + \
                      self._model.beta * gis[current_df['item'].tolist()]
                p = self._model.predict(guf, gif, current_df['user'].tolist(), current_df['item'].tolist())
                predictions_test += p.detach().cpu().numpy().tolist()
                t.update()
        return predictions_val, predictions_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0.0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            predictions_val, predictions_test = self.get_recommendations()
            true_val, true_test = self.df_val_rat['rating'].to_numpy(), self.df_test_rat['rating'].to_numpy()
            result_dict = self.evaluator.eval_error(np.array(predictions_val), true_val, np.array(predictions_test),
                                                    true_test)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss:.5f}')
            else:
                self.logger.info(f'Finished')

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][0]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def get_loss(self):
        if self._optimize_internal_loss:
            return min(self._losses)
        else:
            return min([r[0]["val_results"][self._validation_metric] for r in self._results])

    def get_best_arg(self):
        if self._optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmin([r[0]["val_results"][self._validation_metric] for r in self._results])
        return val_results

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
