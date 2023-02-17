import numpy as np
import pandas as pd
import scipy
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity


def build_users_neighbour(df):
    ui_dict = {u: df[df[0] == u][1].tolist() for u in df[0].unique().tolist()}
    return ui_dict


def dataframe_to_dict(data):
    ratings = data.set_index(0)[[1, 2]].apply(lambda x: (x[1], float(x[2])), 1) \
        .groupby(level=0).agg(lambda x: dict(x.values)).to_dict()
    return ratings


parser = argparse.ArgumentParser(description="Run similarity calculation on user-user.")
parser.add_argument('--dataset', nargs='?', default='digital_music', help='dataset path')
parser.add_argument('--top_k', nargs='+', type=int, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    help='top k list')
args = parser.parse_args()

mapping = {1.0: np.array([[1.0, 0.0, 0.0]]),
           2.0: np.array([[0.0, 1.0, 0.0]]),
           3.0: np.array([[1.0, 1.0, 0.0]]),
           4.0: np.array([[0.0, 0.0, 1.0]]),
           5.0: np.array([[1.0, 0.0, 1.0]])}

dataset = args.dataset
top_k = args.top_k

train = pd.read_csv(f'./data/{dataset}/Train.tsv', sep='\t', header=None)
user_item_dict = build_users_neighbour(train)

train_dict = dataframe_to_dict(train)
users = list(train_dict.keys())
items = list({k for a in train_dict.values() for k in a.keys()})

initial_num_users = train[0].nunique()
initial_num_items = train[1].nunique()

private_to_public_users = {p: u for p, u in enumerate(users)}
public_to_private_users = {v: k for k, v in private_to_public_users.items()}
private_to_public_items = {p: i for p, i in enumerate(items)}
public_to_private_items = {v: k for k, v in private_to_public_items.items()}

rows = [public_to_private_users[u] for u in train[0].tolist()]
cols = [public_to_private_items[i] for i in train[1].tolist()]

R = scipy.sparse.coo_matrix(([1] * len(rows), (rows, cols)), shape=(initial_num_users, initial_num_items))
R_U = R @ R.T
user_rows, user_cols = R_U.nonzero()
user_values = R_U.data
avg_all_users = user_values.mean()

del R, R_U

user_rows_sim = []
user_cols_sim = []
dot_values = []
min_values = []
max_values = []
avg_values = []
rat_dot_values = []
rat_min_values = []
rat_max_values = []
rat_avg_values = []
no_coeff_values = []
rat_no_coeff_values = []

for idx, (u1, u2, value) in enumerate(zip(user_rows, user_cols, user_values)):
    if (idx + 1) % 1000 == 0:
        print(str(idx + 1) + '/' + str(user_rows.shape[0]))
    if u1 != u2:
        u1_items = user_item_dict[private_to_public_users[u1]]
        u2_items = user_item_dict[private_to_public_users[u2]]
        co_occurred_items = set(u1_items).intersection(set(u2_items))
        num_item_intersection = len(co_occurred_items)
        num_item_union = len(set(u1_items).union(set(u2_items)))
        min_item = min(len(u1_items), len(u2_items))
        max_item = max(len(u1_items), len(u2_items))
        clust_coeff_dot = num_item_intersection / num_item_union
        clust_coeff_min = num_item_intersection / min_item
        clust_coeff_max = num_item_intersection / max_item
        avg_coeff = num_item_intersection / avg_all_users
        sum_ = 0
        sum_rating = 0
        for item in co_occurred_items:
            rating_left = train[(train[0] == private_to_public_users[u1]) & (train[1] == item)][2].values[0]
            rating_right = train[(train[0] == private_to_public_users[u2]) & (train[1] == item)][2].values[0]
            rating_left_np = mapping[rating_left]
            rating_right_np = mapping[rating_right]
            left = np.load(os.path.join(f'./data/{dataset}/reviews/',
                                        str(item) + '_' + str(int(private_to_public_users[u1]))) + '.npy')
            right = np.load(os.path.join(f'./data/{dataset}/reviews/',
                                         str(item) + '_' + str(int(private_to_public_users[u2]))) + '.npy')
            dist = cosine_similarity(left, right)[0, 0]
            if dist > 0.0:
                sum_ += dist
            dist = cosine_similarity(rating_left_np, rating_right_np)[0, 0]
            if dist > 0.0:
                sum_rating += dist
        if num_item_intersection > 0:
            user_rows_sim.append(u1)
            user_cols_sim.append(u2)
            dot_values.append(sum_ * clust_coeff_dot)
            min_values.append(sum_ * clust_coeff_min)
            max_values.append(sum_ * clust_coeff_max)
            avg_values.append(sum_ * avg_coeff)
            rat_dot_values.append(sum_rating * clust_coeff_dot)
            rat_min_values.append(sum_rating * clust_coeff_min)
            rat_max_values.append(sum_rating * clust_coeff_max)
            rat_avg_values.append(sum_rating * avg_coeff)
            no_coeff_values.append(sum_)
            rat_no_coeff_values.append(sum_rating)

for t in top_k:
    R_U = scipy.sparse.coo_matrix((dot_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_dot_{top_k}.npz', R_U)

    R_U = scipy.sparse.coo_matrix((min_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_min_{t}.npz', R_U)

    R_U = scipy.sparse.coo_matrix((max_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_max_{t}.npz', R_U)

    R_U = scipy.sparse.coo_matrix((avg_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_avg_{t}.npz', R_U)

    # rating based

    R_U = scipy.sparse.coo_matrix((rat_dot_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_rat_dot_{top_k}.npz', R_U)

    R_U = scipy.sparse.coo_matrix((rat_min_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_rat_min_{t}.npz', R_U)

    R_U = scipy.sparse.coo_matrix((rat_max_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_rat_max_{t}.npz', R_U)

    R_U = scipy.sparse.coo_matrix((rat_avg_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_rat_avg_{t}.npz', R_U)

    # without coefficient

    R_U = scipy.sparse.coo_matrix((no_coeff_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_no_coeff_{top_k}.npz', R_U)

    R_U = scipy.sparse.coo_matrix((rat_no_coeff_values, (user_rows_sim, user_cols_sim)),
                                  shape=(initial_num_users, initial_num_users)).todense()
    indices_one = np.argsort(-R_U)[:, :t]
    indices_zero = np.argsort(-R_U)[:, t:]
    R_U[np.arange(R_U.shape[0])[:, None], indices_one] = 1.0
    R_U[np.arange(R_U.shape[0])[:, None], indices_zero] = 0.0
    R_U = scipy.sparse.coo_matrix(R_U)
    scipy.sparse.save_npz(f'./data/{dataset}/uu_rat_no_coeff_{t}.npz', R_U)
