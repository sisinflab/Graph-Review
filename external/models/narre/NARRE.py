from ast import literal_eval as make_tuple

from tqdm import tqdm
from .pointwise_pos_neg_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .NARREModel import NARREModel

import numpy as np


class NARRE(RecMixin, BaseRecommenderModel):
    r"""
    Neural Attentional Rating Regression with Review-level Explanations
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_batch_eval", "batch_eval", "batch_eval", 64, int, None),
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_u_rev_cnn_k", "u_rev_cnn_k", "u_rev_cnn_k", "(3,)",
             lambda x: list(make_tuple(x)), None),
            ("_u_rev_cnn_f", "u_rev_cnn_f", "u_rev_cnn_f", 100, int, None),
            ("_i_rev_cnn_k", "i_rev_cnn_k", "i_rev_cnn_k", "(3,)",
             lambda x: list(make_tuple(x)), None),
            ("_i_rev_cnn_f", "i_rev_cnn_f", "i_rev_cnn_f", 100, int, None),
            ("_att_u", "att_u", "att_u", 100, int, None),
            ("_att_i", "att_i", "att_i", 100, int, None),
            ("_lat_s", "lat_s", "lat_s", 100, int, None),
            ("_n_lat", "n_lat", "n_lat", 100, int, None),
            ("_pretr", "pretr", "pretr", True, bool, None),
            ("_dropout", "dropout", "dropout", 0.5, float, None),
            ("_loader", "loader", "loader", 'WordsTextualAttributes', str, None)
        ]
        self.autoset_params()

        np.random.seed(self._seed)

        self._interactions_textual = self._data.side_information.WordsTextualAttributes
        initWU, initWI, user_token, item_token = self._interactions_textual.object.get_features()

        self._sampler = Sampler(
            self._data.public_users,
            self._data.public_items,
            self._interactions_textual.object.train_pkl_path,
            self._interactions_textual.object.val_pkl_path,
            self._interactions_textual.object.test_pkl_path,
            user_token,
            item_token,
            self._seed
        )

        self._model = NARREModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            l_w=self._l_w,
            users_vocabulary_features=initWU,
            items_vocabulary_features=initWI,
            textual_words_feature_shape=initWU.shape[1],
            user_review_cnn_kernel=self._u_rev_cnn_k,
            user_review_cnn_features=self._u_rev_cnn_f,
            item_review_cnn_kernel=self._i_rev_cnn_k,
            item_review_cnn_features=self._i_rev_cnn_f,
            attention_size_user=self._att_u,
            attention_size_item=self._att_i,
            latent_size=self._lat_s,
            n_latent=self._n_lat,
            dropout_rate=self._dropout,
            pretrained=self._pretr,
            random_seed=self._seed,
        )

    @property
    def name(self):
        return "NARRE" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_test = []
        predictions_val = []
        with tqdm(total=int(len(self._sampler.val_pkl) // self._batch_eval), disable=not self._verbose) as t:
            for batch in self._sampler.step(len(self._sampler.val_pkl), self._batch_eval):
                p = self._model.predict(batch)
                predictions_val += p.numpy().tolist()
                t.update()
        with tqdm(total=int(len(self._sampler.test_pkl) // self._batch_eval), disable=not self._verbose) as t:
            for batch in self._sampler.step(len(self._sampler.test_pkl), self._batch_eval):
                p = self._model.predict(batch)
                predictions_test += p.numpy().tolist()
                t.update()
        return predictions_val, predictions_test

    def evaluate(self, it=None, loss=0.0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            predictions_val, predictions_test = self.get_recommendations()
            true_val = np.array([v[-1][0] for v in self._sampler.val_pkl])
            true_test = np.array([v[-1][0] for v in self._sampler.test_pkl])
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
