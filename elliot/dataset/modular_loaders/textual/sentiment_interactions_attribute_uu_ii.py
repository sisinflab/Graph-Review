import typing as t
from scipy import sparse
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class SentimentInteractionsTextualAttributesUUII(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.ii_dot_path = getattr(ns, "ii_dot", None)
        self.ii_max_path = getattr(ns, "ii_max", None)
        self.ii_min_path = getattr(ns, "ii_min", None)
        self.ii_global_path = getattr(ns, "ii_global", None)
        self.uu_dot_path = getattr(ns, "uu_dot", None)
        self.uu_max_path = getattr(ns, "uu_max", None)
        self.uu_min_path = getattr(ns, "uu_min", None)
        self.uu_global_path = getattr(ns, "uu_global", None)

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "SentimentInteractionsTextualAttributesUUII"
        ns.object = self

        return ns

    def get_features(self):
        if self.ii_dot_path:
            yield 'ii_dot', sparse.load_npz(self.ii_dot_path)
        if self.ii_max_path:
            yield 'ii_max', sparse.load_npz(self.ii_max_path)
        if self.ii_min_path:
            yield 'ii_min', sparse.load_npz(self.ii_min_path)
        if self.ii_global_path:
            yield 'ii_global', sparse.load_npz(self.ii_global_path)

        if self.uu_dot_path:
            yield 'uu_dot', sparse.load_npz(self.uu_dot_path)
        if self.uu_max_path:
            yield 'uu_max', sparse.load_npz(self.uu_max_path)
        if self.uu_min_path:
            yield 'uu_min', sparse.load_npz(self.uu_min_path)
        if self.uu_global_path:
            yield 'uu_global', sparse.load_npz(self.uu_global_path)
