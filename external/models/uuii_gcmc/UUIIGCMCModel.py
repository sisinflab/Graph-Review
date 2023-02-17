from abc import ABC
from torch_geometric.nn import GCNConv
from collections import OrderedDict

from torch_geometric.nn import LGConv

import torch
import torch_geometric
import numpy as np
import random


class UUIIGCMCModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 convolutional_layer_size,
                 dense_layer_size,
                 n_convolutional_layers,
                 n_dense_layers,
                 num_relations,
                 relations,
                 num_basis,
                 dropout,
                 accumulation,
                 alpha,
                 beta,
                 n_uu_layers,
                 n_ii_layers,
                 sim_uu,
                 sim_ii,
                 random_seed,
                 name="UUIIGCMC",
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
        self.n_convolutional_layers = n_convolutional_layers
        self.n_dense_layers = n_dense_layers
        self.convolutional_layer_size = [self.embed_k] + ([convolutional_layer_size] * self.n_convolutional_layers)
        self.num_relations = num_relations
        self.relations = torch.tensor(relations, dtype=torch.float64, device=self.device)
        self.num_basis = num_basis
        self.dropout = dropout

        self.alpha = alpha
        self.beta = beta
        self.n_uu_layers = n_uu_layers
        self.n_ii_layers = n_ii_layers

        # user-user graph
        self.sim_uu = sim_uu

        # item-item graph
        self.sim_ii = sim_ii

        if accumulation not in ['stack', 'sum']:
            raise NotImplementedError('This accumulation method has not been implemented yet!')

        self.accumulation = accumulation
        self.dense_layer_size = [self.convolutional_layer_size[-1] if self.accumulation == 'sum' else
                                 self.convolutional_layer_size[-1] * self.num_relations] + (
                                            [dense_layer_size] * self.n_dense_layers)

        # collaborative embeddings
        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        # similarity embeddings
        self.Gus = torch.nn.Embedding(self.num_users, self.dense_layer_size[-1])
        torch.nn.init.xavier_normal_(self.Gus.weight)
        self.Gus.to(self.device)
        self.Gis = torch.nn.Embedding(self.num_items, self.dense_layer_size[-1])
        torch.nn.init.xavier_normal_(self.Gis.weight)
        self.Gis.to(self.device)

        self.P = torch.nn.ParameterList()
        for _ in range(self.num_basis):
            p_s = torch.nn.Parameter(
                torch.nn.init.orthogonal_(torch.empty((self.dense_layer_size[-1], self.dense_layer_size[-1]))))
            p_s.to(self.device)
            self.P.append(p_s)
        self.A = torch.nn.ParameterList()
        for _ in range(self.num_relations):
            a_r = torch.nn.Parameter(
                torch.nn.init.uniform_(torch.empty((self.num_basis, 1))))
            a_r.to(self.device)
            self.A.append(a_r)

        # Convolutional part
        self.convolutions = torch.nn.ModuleList()
        for _ in range(self.num_relations):
            convolutional_network_list = []
            for layer in range(self.n_convolutional_layers):
                convolutional_network_list.append((GCNConv(in_channels=self.convolutional_layer_size[layer],
                                                           out_channels=self.convolutional_layer_size[layer + 1],
                                                           add_self_loops=False,
                                                           bias=False),
                                                   'x, edge_index -> x'))
            convolutional_network = torch_geometric.nn.Sequential('x, edge_index', convolutional_network_list)
            convolutional_network.to(self.device)
            self.convolutions.append(convolutional_network)

        # Dense part
        dense_network_list = []
        for layer in range(self.n_dense_layers):
            dense_network_list.append(('dense_' + str(layer), torch.nn.Linear(in_features=self.dense_layer_size[layer],
                                                                              out_features=self.dense_layer_size[
                                                                                  layer + 1],
                                                                              bias=False)))
        self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))
        self.dense_network.to(self.device)

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def propagate_embeddings(self, evaluate, adjacency_matrix):
        # propagate embeddings with similarity matrix
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

        # propagate embeddings gcmc
        current_embeddings = []
        for _ in range(self.num_relations):
            current_embeddings += [torch.cat((self.Gu, self.Gi), 0)]

        for r in range(self.num_relations):
            for layer in range(self.n_convolutional_layers):
                if evaluate:
                    self.convolutions.eval()
                    with torch.no_grad():
                        current_embeddings[r] = torch.relu(list(
                            self.convolutions[r].children()
                        )[layer](current_embeddings[r].to(self.device), adjacency_matrix[r].to(self.device)))
                else:
                    current_embeddings[r] = torch.relu(list(
                        self.convolutions[r].children()
                    )[layer](current_embeddings[r].to(self.device), adjacency_matrix[r].to(self.device)))

        if self.accumulation == 'stack':
            current_embeddings = torch.cat(current_embeddings, 1)
        else:
            current_embeddings = [torch.unsqueeze(ce, 0) for ce in current_embeddings]
            current_embeddings = torch.cat(current_embeddings, 0)
            current_embeddings = torch.sum(current_embeddings, 0)

        if evaluate:
            self.dense_network.eval()
            with torch.no_grad():
                current_embeddings = self.dense_network(current_embeddings.to(self.device))
        else:
            current_embeddings = torch.nn.functional.dropout(self.dense_network(current_embeddings.to(self.device)),
                                                             p=self.dropout)

        if evaluate:
            self.convolutions.train()
            self.dense_network.train()

        zu, zi = torch.split(current_embeddings, [self.num_users, self.num_items], 0)
        return zu, zi, gus, gis

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        zeta_u = torch.squeeze(gu).to(self.device)
        zeta_i = torch.squeeze(gi).to(self.device)

        xui_r = []

        for r in range(self.num_relations):
            q_r = []
            for s in range(self.num_basis):
                q_r.append(torch.unsqueeze(self.A[r][s] * self.P[s], 0))
            q_r = torch.sum(torch.cat(q_r, 0), 0)
            xui_r.append(torch.unsqueeze(
                torch.sum(torch.mm(zeta_u.to(self.device), q_r.to(self.device)) * zeta_i.to(self.device), 1), 1))
        pui = torch.cat(xui_r, 1)
        xui = torch.sum(self.relations.to(self.device) * torch.softmax(pui.to(self.device), 1), 1)
        return xui, pui

    def predict(self, gu, gi, **kwargs):
        xui, _ = self.forward((gu, gi))
        return xui

    def train_step(self, batch, adj_matrix):
        zu, zi, gus, gis = self.propagate_embeddings(False, adj_matrix)
        user, item, r = batch

        gu = (1 - self.alpha) * zu[user] + self.alpha * gus[user]
        gi = (1 - self.beta) * zi[item] + self.beta * gis[item]

        xui, pui = self.forward(inputs=(gu, gi))

        loss = - torch.sum(torch.nn.functional.log_softmax(pui, 1) * torch.nn.functional.one_hot(
            torch.tensor(r, device=self.device))
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()
