"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC
from torch_geometric.nn import GCNConv, LGConv

import torch
import torch_geometric
import numpy as np
import random


class UUIIGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 num_uu_layers,
                 num_ii_layers,
                 learning_rate,
                 embed_k,
                 n_layers,
                 sim_uu,
                 sim_ii,
                 alpha,
                 beta,
                 weight_size,
                 adj,
                 random_seed,
                 name="UUIIGCN",
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
        self.weight_size_list = weight_size
        self.adj = adj
        self.n_uu_layers = num_uu_layers
        self.n_ii_layers = num_ii_layers

        # collaborative filtering
        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        # similarity embeddings
        self.Gus = torch.nn.Embedding(self.num_users, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gus.weight)
        self.Gus.to(self.device)
        self.Gis = torch.nn.Embedding(self.num_items, self.embed_k)
        torch.nn.init.xavier_uniform_(self.Gis.weight)
        self.Gis.to(self.device)

        self.Bu = torch.nn.Embedding(self.num_users, 1)
        torch.nn.init.xavier_normal_(self.Bu.weight)
        self.Bu.to(self.device)
        self.Bi = torch.nn.Embedding(self.num_items, 1)
        torch.nn.init.xavier_normal_(self.Bi.weight)
        self.Bi.to(self.device)

        self.Mu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((1, 1))))
        self.Mu.to(self.device)

        self.alpha = alpha
        self.beta = beta

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

        self.n_layers = n_layers
        self.weight_size = weight_size
        self.weight_size_list = [self.embed_k] + ([self.weight_size] * self.n_layers)

        propagation_network_list = []

        for layer in range(self.n_layers):
            propagation_network_list.append((GCNConv(in_channels=self.weight_size_list[layer],
                                                     out_channels=self.weight_size_list[layer + 1],
                                                     normalize=True,
                                                     add_self_loops=False,
                                                     bias=False), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = torch.nn.MSELoss()

    def propagate_embeddings(self, evaluate=False):
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

        current_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)

        for layer in range(self.n_layers):
            if evaluate:
                self.propagation_network.eval()
                with torch.no_grad():
                    current_embeddings = torch.relu(list(
                        self.propagation_network.children()
                    )[layer](current_embeddings.to(self.device), self.adj.to(self.device)))
            else:
                current_embeddings = torch.relu(list(
                    self.propagation_network.children()
                )[layer](current_embeddings.to(self.device), self.adj.to(self.device)))

        if evaluate:
            self.propagation_network.train()

        gu, gi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
        return gu, gi, gus, gis

    def forward(self, inputs, **kwargs):
        gu, gi, bu, bi, = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        beta_u = torch.squeeze(bu).to(self.device)
        beta_i = torch.squeeze(bi).to(self.device)

        mu = torch.squeeze(self.Mu).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1) + beta_u + beta_i + mu

        return xui

    def predict(self, gu, gi, users, items, **kwargs):
        return self.forward(inputs=(gu, gi, self.Bu.weight[users], self.Bi.weight[items]))

    def train_step(self, batch):
        gu, gi, gus, gis = self.propagate_embeddings()
        user, item, r = batch

        guf = (1 - self.alpha) * gu[user] + self.alpha * gus[user]
        gif = (1 - self.beta) * gi[item] + self.beta * gis[item]

        rui = self.forward(inputs=(guf, gif, self.Bu.weight[user], self.Bi.weight[item]))

        loss = self.loss(torch.squeeze(rui), torch.tensor(r, device=self.device, dtype=torch.float))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()
