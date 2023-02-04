import numpy as np
import pickle
import random
from operator import itemgetter


class Sampler:
    def __init__(self,
                 public_users,
                 public_items,
                 train_pkl_path,
                 val_pkl_path,
                 test_pkl_path,
                 user_token,
                 item_token,
                 seed):
        self.public_users = public_users
        self.public_items = public_items

        train_pkl_file = open(train_pkl_path, 'rb')
        self.train_pkl = pickle.load(train_pkl_file)
        train_pkl_file.close()

        val_pkl_file = open(val_pkl_path, 'rb')
        self.val_pkl = pickle.load(val_pkl_file)
        val_pkl_file.close()

        test_pkl_file = open(test_pkl_path, 'rb')
        self.test_pkl = pickle.load(test_pkl_file)
        test_pkl_file.close()

        self.user_token = user_token
        self.item_token = item_token
        np.random.seed(seed)
        random.seed(seed)
        train_pkl_file.close()

    def step(self, events: int, batch_size: int):
        _user_token = self.user_token
        _item_token = self.item_token
        _train_pkl = self.train_pkl
        _public_users = self.public_users
        _public_items = self.public_items

        random.shuffle(self.train_pkl)

        def sample(idx):
            uid = _train_pkl[idx][0][0]
            iid = _train_pkl[idx][1][0]
            rid = _train_pkl[idx][4][0]
            u_token = _user_token[uid]
            i_token = _item_token[iid]
            u_pos = _train_pkl[idx][2]
            i_pos = _train_pkl[idx][3]
            u_pos = list(itemgetter(*u_pos.tolist())(_public_items))
            i_pos = list(itemgetter(*i_pos.tolist())(_public_users))
            return _public_users[uid], _public_items[iid], rid, u_token, i_token, u_pos, i_pos

        for batch_start in range(0, events, batch_size):
            user, item, bit, u_t, i_t, u_p, i_p = map(np.array, zip(*[sample(i) for i in
                                                                      range(batch_start,
                                                                            min(batch_start + batch_size, events))]))
            yield user, item, bit.astype('float32'), u_t, i_t, u_p, i_p

    def step_val(self, events: int, batch_size: int):
        _user_token = self.user_token
        _item_token = self.item_token
        _val_pkl = self.val_pkl
        _public_users = self.public_users
        _public_items = self.public_items

        def sample(idx):
            uid = _val_pkl[idx][0][0]
            iid = _val_pkl[idx][1][0]
            rid = _val_pkl[idx][4][0]
            u_token = _user_token[uid]
            i_token = _item_token[iid]
            u_pos = _val_pkl[idx][2]
            i_pos = _val_pkl[idx][3]
            u_pos = list(itemgetter(*u_pos.tolist())(_public_items))
            i_pos = list(itemgetter(*i_pos.tolist())(_public_users))
            return _public_users[uid], _public_items[iid], rid, u_token, i_token, u_pos, i_pos

        for batch_start in range(0, events, batch_size):
            user, item, bit, u_t, i_t, u_p, i_p = map(np.array, zip(*[sample(i) for i in
                                                                      range(batch_start,
                                                                            min(batch_start + batch_size, events))]))
            yield user, item, bit.astype('float32'), u_t, i_t, u_p, i_p

    def step_test(self, events: int, batch_size: int):
        _user_token = self.user_token
        _item_token = self.item_token
        _test_pkl = self.test_pkl
        _public_users = self.public_users
        _public_items = self.public_items

        def sample(idx):
            uid = _test_pkl[idx][0][0]
            iid = _test_pkl[idx][1][0]
            rid = _test_pkl[idx][4][0]
            u_token = _user_token[uid]
            i_token = _item_token[iid]
            u_pos = _test_pkl[idx][2]
            i_pos = _test_pkl[idx][3]
            u_pos = list(itemgetter(*u_pos.tolist())(_public_items))
            i_pos = list(itemgetter(*i_pos.tolist())(_public_users))
            return _public_users[uid], _public_items[iid], rid, u_token, i_token, u_pos, i_pos

        for batch_start in range(0, events, batch_size):
            user, item, bit, u_t, i_t, u_p, i_p = map(np.array, zip(*[sample(i) for i in
                                                                      range(batch_start,
                                                                            min(batch_start + batch_size, events))]))
            yield user, item, bit.astype('float32'), u_t, i_t, u_p, i_p
