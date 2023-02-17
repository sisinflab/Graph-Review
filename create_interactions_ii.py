import numpy as np
import pandas as pd
import scipy
import os
import argparse
from sklearn.metrics.pairwise import cosine_similarity


def build_items_neighbour(df):
    iu_dict = {i: df[df[1] == i][0].tolist() for i in df[1].unique().tolist()}
    return iu_dict


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
item_user_dict = build_items_neighbour(train)

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
R_I = R.T @ R
item_rows, item_cols = R_I.nonzero()
item_values = R_I.data
avg_all_items = item_values.mean()

del R, R_I

item_rows_sim = []
item_cols_sim = []
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

for idx, (i1, i2, value) in enumerate(zip(item_rows, item_cols, item_values)):
    if (idx + 1) % 1000 == 0:
        print(str(idx + 1) + '/' + str(item_rows.shape[0]))
    if i1 != i2:
        i1_users = item_user_dict[private_to_public_items[i1]]
        i2_users = item_user_dict[private_to_public_items[i2]]
        co_occurred_users = set(i1_users).intersection(set(i2_users))
        num_user_intersection = len(co_occurred_users)
        num_user_union = len(set(i1_users).union(set(i2_users)))
        min_user = min(len(i1_users), len(i2_users))
        max_user = max(len(i1_users), len(i2_users))
        clust_coeff_dot = num_user_intersection / num_user_union
        clust_coeff_min = num_user_intersection / min_user
        clust_coeff_max = num_user_intersection / max_user
        avg_coeff = num_user_intersection / avg_all_items
        sum_ = 0
        sum_rating = 0
        for user in co_occurred_users:
            rating_left = train[(train[1] == private_to_public_items[i1]) & (train[0] == user)][2].values[0]
            rating_right = train[(train[1] == private_to_public_items[i2]) & (train[0] == user)][2].values[0]
            rating_left_np = mapping[rating_left]
            rating_right_np = mapping[rating_right]
            left = np.load(os.path.join(f'./data/{dataset}/reviews/',
                                        str(int(private_to_public_items[i1])) + '_' + str(user)) + '.npy')
            right = np.load(os.path.join(f'./data/{dataset}/reviews/',
                                         str(int(private_to_public_items[i2])) + '_' + str(user)) + '.npy')
            dist = cosine_similarity(left, right)[0, 0]
            if dist > 0.0:
                sum_ += dist
            dist = cosine_similarity(rating_left_np, rating_right_np)[0, 0]
            if dist > 0.0:
                sum_rating += dist
        if num_user_intersection > 0:
            item_rows_sim.append(i1)
            item_cols_sim.append(i2)
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
    R_I = scipy.sparse.coo_matrix((dot_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_dot_{t}.npz', R_I)

    R_I = scipy.sparse.coo_matrix((min_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_min_{t}.npz', R_I)

    R_I = scipy.sparse.coo_matrix((max_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_max_{t}.npz', R_I)

    R_I = scipy.sparse.coo_matrix((avg_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_avg_{t}.npz', R_I)

    # rating based

    R_I = scipy.sparse.coo_matrix((rat_dot_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_rat_dot_{t}.npz', R_I)

    R_I = scipy.sparse.coo_matrix((rat_min_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_rat_min_{t}.npz', R_I)

    R_I = scipy.sparse.coo_matrix((rat_max_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_rat_max_{t}.npz', R_I)

    R_I = scipy.sparse.coo_matrix((rat_avg_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_rat_avg_{t}.npz', R_I)

    # without coefficient

    R_I = scipy.sparse.coo_matrix((no_coeff_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_no_coeff_{t}.npz', R_I)

    R_I = scipy.sparse.coo_matrix((rat_no_coeff_values, (item_rows_sim, item_cols_sim)),
                                  shape=(initial_num_items, initial_num_items)).todense()
    indices_one = np.argsort(-R_I)[:, :t]
    indices_zero = np.argsort(-R_I)[:, t:]
    R_I[np.arange(R_I.shape[0])[:, None], indices_one] = 1.0
    R_I[np.arange(R_I.shape[0])[:, None], indices_zero] = 0.0
    R_I = scipy.sparse.coo_matrix(R_I)
    scipy.sparse.save_npz(f'./data/{dataset}/ii_rat_no_coeff_{t}.npz', R_I)
