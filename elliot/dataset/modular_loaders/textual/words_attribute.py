import typing as t
import numpy as np
import pickle
from types import SimpleNamespace
import gensim.downloader

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class WordsTextualAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger

        self.para = getattr(ns, "para", None)
        self.embedding_dim = getattr(ns, "embedding_dim", None)
        self.word2vec = getattr(ns, "word2vec", None)
        self.train_pkl_path = getattr(ns, "train_pkl", None)
        self.val_pkl_path = getattr(ns, "val_pkl", None)
        self.test_pkl_path = getattr(ns, "test_pkl", None)

        self.model = gensim.downloader.load(self.word2vec)

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "WordsTextualAttributes"
        ns.object = self

        return ns

    def get_features(self):
        para_pkl_file = open(self.para, 'rb')
        para = pickle.load(para_pkl_file)
        para_pkl_file.close()

        vocabulary_user = para['user_vocab']
        vocabulary_item = para['item_vocab']

        initWU = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), self.embedding_dim))
        not_found_user = 0
        for k, v in vocabulary_user.items():
            try:
                initWU[v] = self.model.get_vector(k, norm=True)
            except KeyError:
                not_found_user += 1
                pass

        initWI = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), self.embedding_dim))
        not_found_item = 0
        for k, v in vocabulary_item.items():
            try:
                initWI[v] = self.model.get_vector(k, norm=True)
            except KeyError:
                not_found_item += 1
                pass

        self.logger.info(f"Number of words not found in user vocabulary: {not_found_user}")
        self.logger.info(f"Number of words not found in item vocabulary: {not_found_item}")

        return initWU, initWI, para['u_text'], para['i_text']

