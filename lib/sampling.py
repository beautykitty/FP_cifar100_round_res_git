#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random
import torch


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users

def mnist_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # begin = i*10 + label_begin[each_class.item()]
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

def mnist_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = test_dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 40 # 每个类选多少张做测试
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i*40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def femnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from FEMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def femnist_noniid(args, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from FEMNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    dict_users = {}
    classes_list = []
    classes_list_gt = []

    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        print("classes_gt:", classes)
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k_len * num_users + i * k_len
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def femnist_noniid_lt(args, num_users, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    dict_users = {}

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k * num_users + i * k
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)), axis=0)
        dict_users[i] = user_data

    return dict_users


def femnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

def cifar10_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 5000
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    classes_list_gt = []
    k_len = args.train_shots_max
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def cifar10_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def cifar100_noniid(args, dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 100, 500
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    print("idxs: ", idxs)
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    print("idxs_labels: ",idxs_labels)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    print("idxs_labels: ",idxs_labels)
    idxs = idxs_labels[0, :]
    print("idxs: ",idxs)
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1
    print("label_begin: ", label_begin)

    ## classes_list
    
    classes_list = classes_list # [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14], [15,16,17,18,19]]
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        # classes = random.sample(range(0,args.num_classes), n)
        # classes = np.sort(classes)
        classes = classes_list[i]
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:",classes)
        #print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # no random 
            begin = label_begin[each_class] # i * args.shots +
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        
        #classes_list.append(classes)

    return dict_users, classes_list


def cifar100_noniid_lt(args, test_dataset, num_users, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 100, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = args.test_shots # 每个类选多少张做测试
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # no random
            begin =  label_begin[each_class] # i*args.test_shots +
            #begin = random.randint(0,90) + label_begin[each_class]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users




def cub_noniid(args, dataset, num_users, n_list, k_list):
    
    # 데이터셋 내 라벨을 새로운 라벨로 재라벨링합니다.  
    dict_users = {}
    
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.data.target)
    print("labels:", labels)
    

    idxs_labels = np.vstack((idxs, labels))
    print("idxs_labels: ", idxs_labels)
    print("idxs_labels shape: ", idxs_labels.shape)
    
    # class당 이미지 수    
    _, img_num_class = np.unique(dataset.data.target, return_counts=True)
           
    # sort labels
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    print("idxs_labels: ", idxs_labels)

    idxs = idxs_labels[0, :]
    print("idxs: ", idxs)
    label_begin = {}
    cnt=0
    
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1
    
    print("label_begin: ",label_begin)
    
    # #### class relabeling
    # class_list = np.sort(class_list)
    # new_labels = {idx: class_num for idx, class_num in enumerate(class_list)}
    # print("new_labels: ", new_labels)

    classes_list = [] 
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        
        # 0~10 중에서 n개 선택
        classes = np.sort(random.sample(range(0,args.num_classes), n))
        # re_classes_list.append(re_classes)
        # print("re_classes: ", re_classes)
        classes_list.append(classes)

        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("new_labels:", dataset.new_labels)
        print("classes:", classes)
        user_data = np.array([])
        
        for each_class in classes:
            begin = label_begin[each_class]
            #print(begin)
            #print(label_begin)
            # random number를 택하는 걸로 import random, rand_nums = random.sample(range(10, 21), k=5)
            
            num_of_class = img_num_class[each_class]
            if k <= num_of_class:           
                user_data = np.concatenate((user_data, idxs[begin: begin+k]), axis=0)
            else: 
                user_data = np.concatenate((user_data, idxs[begin: begin+num_of_class]), axis=0)

        dict_users[i] = user_data
        #classes_list.append(classes)
        
    ## return value 추가
    return dict_users, classes_list


def cub_noniid_lt(args, test_dataset, num_users, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    #print("\n\n lt_classes_list: ", classes_list)
    dict_users = {}
    idxs = np.arange(len(test_dataset))
    #print("idxs:", idxs)
    ##
    labels = np.array(test_dataset.data.target)
    #print("labels:",labels)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    #print("idx_labels:", idxs_labels)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    #print("idx_labels:", idxs_labels)
    idxs = idxs_labels[0, :]
    #print("idx:", idxs)
    
    
    _, img_num_class = np.unique(test_dataset.data.target, return_counts=True)
    #print("img_num_class: ", img_num_class)
    


    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1
        
    #print("label_begin:", label_begin)
    
    for i in range(num_users):
        k = args.test_shots # # 每个类选多少张做测试
        classes = classes_list[i]
        #print("local test classes:", classes)
        user_data = np.array([])
        
        #print("classes: ", classes)
        for each_class in classes:
            #print("each_class: ", each_class)
            begin = label_begin[each_class]
            #print("begin: ", begin)

            num_of_class = args.num_classes # img_num_class[each_class]
            #print("num_of_class: ",num_of_class)
            # if k <= num_of_class:           
            #     user_data = np.concatenate((user_data, idxs[begin: begin+k]), axis=0)
            # else: 
            user_data = np.concatenate((user_data, idxs[begin: begin+num_of_class]), axis=0)

            # begin = i*5 + label_begin[each_class.item()]
            #begin = label_begin[each_class.item()]
            #user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data

    return dict_users





if __name__ == '__main__':
    ###
    from options import args_parser
    from cub2011  import Cub2011
    import torch
    import numpy as np
    
    
    args = args_parser()

    #n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    n_list = np.repeat(args.ways, args.num_users)
    k_list = np.repeat(args.shots, args.num_users)

    # print("k_list: ", k_list)
    # for i in range(len(k_list)) :
    #     if k_list[i] >30:
    #         k_list[i] = 30
    # print(k_list)

    seed_value = 42
    np.random.seed(seed_value)
    
    
    
    trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                   std=[0.267, 0.256, 0.276])])
    trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                  std=[0.267, 0.256, 0.276])])
    
    all_class_list = np.arange(0,20)
    #all_class_list = np.sort(np.random.choice(np.arange(1, 100), size=args.num_classes, replace=False))
    print(all_class_list)
    
    train_dataset = datasets.CIFAR100(args.data_dir + args.dataset, train=True, download=False, transform=trans_cifar100_train)
    test_dataset = datasets.CIFAR100(args.data_dir + args.dataset, train=False, download=True, transform=trans_cifar100_val)

    #train_dataset = Cub2011('../data', train=True, class_list=all_class_list, download=False) 
    #test_dataset = Cub2011('../data', train=False, class_list=all_class_list, download=False)
    
    #num_users = 2

    d, cl = cifar100_noniid(args, train_dataset, args.num_users, n_list, k_list)
    d2 = user_groups_lt = cifar100_noniid_lt(args, test_dataset, args.num_users, classes_list=cl)

    print("dict_users:", d, "\n")
    print("classes_list", cl, "\n")
    # print("dict_users_test:", d2, "\n")

    # d, cl = cub_noniid(args, train_dataset, num_users, n_list, k_list)
    # d2 = cub_noniid_lt(args, test_dataset, num_users, cl)
    
    # print("dict_users:", d, "\n")
    # print("classes_list", cl, "\n")
    # print("dict_users_test:", d2, "\n")


    
