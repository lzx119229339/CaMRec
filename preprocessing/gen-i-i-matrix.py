#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import pandas as pd
import os
import yaml
import argparse


def gen_item_matrix(all_edge, no_items):
    """
    根据交互数据生成 item-item 矩阵
    参数:
        all_edge: numpy 数组，形如 (user, item) 的交互数据
        no_items: 物品总数
    返回:
        item_graph_matrix: torch.Tensor，形状为 (no_items, no_items)
                           矩阵中每个元素表示两个物品共同被多少用户交互过
    """
    edge_dict = defaultdict(set)
    # 注意这里与 user‑user 不同，将物品作为 key，存储与之交互过的所有用户
    for edge in all_edge:
        import ipdb; ipdb.set_trace()
        user, item = edge
        edge_dict[item].add(user)

    # 这里假设物品 id 从 0 开始（若不连续则可能需要做 id 映射）
    min_item = 0             
    num_item = no_items      
    item_graph_matrix = torch.zeros(num_item, num_item)
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head + 1, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            users_head = edge_dict[head_key]
            users_rear = edge_dict[rear_key]
            inter_len = len(users_head.intersection(users_rear))
            if inter_len > 0:
                item_graph_matrix[head_key - min_item][rear_key - min_item] = inter_len
                item_graph_matrix[rear_key - min_item][head_key - min_item] = inter_len
    bar.close()

    return item_graph_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='games', help='name of dataset')
    args = parser.parse_args()
    dataset_name = args.dataset
    print(f'Generating i-i matrix for {dataset_name} ...\n')

    config = {}
    os.chdir('../src')
    cur_dir = os.getcwd()
    con_dir = os.path.join(cur_dir, 'configs')  # get config dir
    overall_config_file = os.path.join(con_dir, "overall.yaml")
    dataset_config_file = os.path.join(con_dir, "dataset", "{}.yaml".format(dataset_name))
    conf_files = [overall_config_file, dataset_config_file]
    # load configs
    for file in conf_files:
        if os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                tmp_d = yaml.safe_load(f)
                config.update(tmp_d)

    dataset_path = os.path.abspath(config['data_path'] + dataset_name)
    print('data path:\t', dataset_path)
    import ipdb; ipdb.set_trace()
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    train_df = pd.read_csv(os.path.join(dataset_path, config['inter_file_name']), sep='\t')
    # 获取物品总数（注意这里与用户的处理类似）
    num_user = len(pd.unique(train_df[uid_field]))
    num_item = len(pd.unique(train_df[iid_field]))
    train_df = train_df[train_df['x_label'] == 0].copy()
    train_data = train_df[[uid_field, iid_field]].to_numpy()
    # 生成 item-item 矩阵
    item_graph_matrix = gen_item_matrix(train_data, num_item)
    item_graph = item_graph_matrix
    # 统计每个物品的邻居数
    item_num = torch.zeros(num_item)
    item_graph_dict = {}
    edge_list_i = []
    edge_list_j = []

    for i in range(num_item):
        item_num[i] = len(torch.nonzero(item_graph[i]))
        print("this is ", i, "num", item_num[i])

    # 保留所有邻居（不做 top-k 截断），构造邻居字典
    for i in range(num_item):
        neighbor_indices = torch.nonzero(item_graph[i]).squeeze().tolist()
        # 若只有一个邻居，nonzero 返回 int，此处转为列表
        if isinstance(neighbor_indices, int):
            neighbor_indices = [neighbor_indices]
        neighbor_weights = item_graph[i, torch.nonzero(item_graph[i]).squeeze()].tolist()
        item_graph_dict[i] = [neighbor_indices, neighbor_weights]

    np.save(os.path.join(dataset_path, config['item_graph_dict_file']), item_graph_dict, allow_pickle=True)