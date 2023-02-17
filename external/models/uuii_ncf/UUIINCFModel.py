from abc import ABC

import torch
import torch_geometric
from torch_geometric.nn import LGConv
import numpy as np
import random
from collections import OrderedDict


class UUIINCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_uu_layers,
                 num_ii_layers,
                 learning_rate,
                 embed_k,
                 sim_uu,
                 sim_ii,
                 alpha,
                 beta,
                 dropout,
                 dense_size,
                 random_seed,
                 name="UUIINCF",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.n_uu_layers = num_uu_layers
        self.n_ii_layers = num_ii_layers

        self.dense_layer_size = [self.embed_k * 2] + dense_size

        # collaborative embeddings
        self.Gu = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gu.weight)
        self.Gu.to(self.device)
        self.Gi = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gi.weight)
        self.Gi.to(self.device)

        # similarity embeddings
        self.Gus = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gus.weight)
        self.Gus.to(self.device)
        self.Gis = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gis.weight)
        self.Gis.to(self.device)

        self.alpha = alpha
        self.beta = beta

        self.dropout = dropout

        # user-user graph
        self.sim_uu = sim_uu

        # item-item graph
        self.sim_ii = sim_ii

        # graph convolutional network for user-user graph
        propagation_network_uu_list = []
        for layer in range(self.n_uu_layers):
            propagation_network_uu_list.append((LGConv(normalize=True), 'x, edge_index -> x'))
        self.propagation_network_uu = torch_geometric.nn.Sequential('x, edge_index', propagation_network_uu_list)
        self.propagation_network_uu.to(self.device)

        # graph convolutional network for item-item graph
        propagation_network_ii_list = []
        for layer in range(self.n_ii_layers):
            propagation_network_ii_list.append((LGConv(normalize=True), 'x, edge_index -> x'))
        self.propagation_network_ii = torch_geometric.nn.Sequential('x, edge_index', propagation_network_ii_list)
        self.propagation_network_ii.to(self.device)

        # Dense part
        dense_network_list = []
        for idx, _ in enumerate(self.dense_layer_size[:-1]):
            dense_network_list.append(('dense_' + str(idx), torch.nn.Linear(in_features=self.dense_layer_size[idx],
                                                                            out_features=self.dense_layer_size[
                                                                                idx + 1],
                                                                            bias=True)))
            dense_network_list.append(('relu_' + str(idx), torch.nn.ReLU()))
            if self.dropout > 0.0:
                dense_network_list.append(('dropout_' + str(idx), torch.nn.Dropout(p=self.dropout)))
        dense_network_list.append(('out', torch.nn.Linear(in_features=self.dense_layer_size[-1],
                                                          out_features=1,
                                                          bias=True)))
        self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))
        self.dense_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.MSELoss()

    def propagate_embeddings(self):
        gus = self.Gus.weight.to(self.device)
        for layer in range(self.n_uu_layers):
            gus = list(self.propagation_network_uu.children())[layer](
                gus.to(self.device),
                self.sim_uu.to(self.device))

        gis = self.Gis.weight.to(self.device)
        for layer in range(self.n_ii_layers):
            gis = list(self.propagation_network_ii.children())[layer](
                gis.to(self.device),
                self.sim_ii.to(self.device))

        return gus, gis

    def forward(self, inputs, **kwargs):
        gus, gis = inputs
        gamma_u = torch.squeeze(gus).to(self.device)
        gamma_i = torch.squeeze(gis).to(self.device)

        rui = self.dense_network(torch.concat([gamma_u, gamma_i], dim=1))

        return rui

    def predict(self, gu, gi, **kwargs):
        rui = self.dense_network(torch.concat([gu, gi], dim=1))
        return rui

    def train_step(self, batch):
        gus, gis = self.propagate_embeddings()
        user, item, r = batch
        gu = (1 - self.alpha) * self.Gu.weight[user] + self.alpha * gus[user]
        gi = (1 - self.beta) * self.Gi.weight[item] + self.beta * gis[item]
        rui = self.forward(inputs=(gu, gi))

        loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()
