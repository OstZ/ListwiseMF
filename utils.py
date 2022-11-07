import random

import numpy as np
import pandas as pd
import itertools

import scipy.sparse as sp


def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range
    Parameters
    ----------
    data : np.int32 arrays
    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data
    """
    uniq = list(set(data))
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    n = len(uniq)

    return data, id_dict, n

def load_data(fname):
    """
    load data
    -----
    : fname: datapath
    -----
    :return:

    """
    print('Loading dataset', fname)
    data_dir = 'dataset/' + fname
    files = ['/u.data', '/u.item', '/u.user']
    sep = '\t'
    filename = data_dir + files[0]
    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}
    data = pd.read_csv(
        filename, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'])
    #transform the datatype to list
    sorted_data = data.sort_values(by=['u_nodes','v_nodes'])
    data_array = sorted_data.values.tolist()
    data_array = np.array(data_array)




    #get list of user nodes and so on
    u_nodes = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])

    #map all index to [0,num)
    u_nodes, u_dict, num_users = map_data(u_nodes)
    v_nodes, v_dict, num_items = map_data(v_nodes)

    #set datatype
    u_nodes, v_nodes = u_nodes.astype(np.int32), v_nodes.astype(np.int32)
    ratings = ratings.astype(np.float64)
    return num_users, num_items, u_nodes, v_nodes, ratings

def create_traintest_split(dataset,train_length,filter_length):
    num_u,num_i,u_nodes,v_nodes,ratings = load_data(dataset)
    #default value
    neutral_rating = -1
    #map rating value to [0,n)
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    #create a matrix ro store ratings
    labels = np.full((num_u, num_i), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])


    pairs_nonzero = [[u, v] for u, v in zip(u_nodes, v_nodes)]


    #get pairs for split
    train_pair_list = []
    test_pair_list = []
    item_lists = [list(group) for key,group in itertools.groupby(pairs_nonzero,lambda v:v[0])]
    for item_list in item_lists:
        if(len(item_list) >= filter_length):
            random.shuffle(item_list)
            train_pair_list = train_pair_list + item_list[0:train_length]
            test_pair_list = test_pair_list + item_list[train_length:]
    #ids for train
    idx_train = np.array([u * num_i + v for u, v in train_pair_list])
    #ids for test
    idx_test = np.array([u * num_i + v for u, v in test_pair_list])

    rating_mx_train = np.zeros(num_u * num_i, dtype=np.float32)
    rating_mx_train[idx_train] = labels[idx_train].astype(np.float32) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_u, num_i))

    rating_mx_test = np.zeros(num_u * num_i, dtype=np.float32)
    rating_mx_test[idx_test] = labels[idx_test].astype(np.float32) + 1.
    rating_mx_test = sp.csr_matrix(rating_mx_test.reshape(num_u, num_i))

    return num_u,num_i,rating_mx_train,rating_mx_test


