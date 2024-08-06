import os
import time
import numpy as np
from myseg.dataloader import *
from myseg.dataloader_camvid import CamVid_Dataset


def get_dataset_ade20k(args):
    if args.dataset == 'ade20k' or args.dataset =='voc':
        if args.data == 'train':
            train_dataset = CamVid_Dataset(args,args.root_dir, 'train')  # 2975
        elif args.data == 'val':
            train_dataset = CamVid_Dataset(args,args.root_dir, 'val')

        test_dataset = CamVid_Dataset(args,args.root_dir, 'val')
        #test_dataset = torch.utils.data.Subset(test_dataset, range(10))

        # sample training data among users
        user_groups = cityscapes_noniid_extend(args.root_dir, CamVid_Dataset.train_folder, args.num_users)
    else:
        exit('Unrecognized dataset')

    return train_dataset, test_dataset, user_groups

def cityscapes_noniid_extend(root_dir, train_folder, num_users):
    timer = time.time()
    # city_lens = [174, 96, 316, 154, 85, 221, 109, 248, 196, 119, 99, 94, 365, 196, 144, 95, 142, 122]
    # num_users_per_city = int(num_users / 18)  # 144 / 18 = 8

    city_lens = get_city_num(root_dir, train_folder)
    num_classes = len(city_lens)
    num_users_per_city = int(num_users / num_classes)
    print("num_users_per_city: {} / {} = {}".format(num_users, num_classes, num_users_per_city))
    assert num_users % num_classes == 0, "num_users % num_classes != 0"

    dict_users = {}
    for city_idx in range(num_classes):
        num_items = int(city_lens[city_idx] / num_users_per_city)

        city_lens_prefix_sum = sum(city_lens[:city_idx]) # prefix sum of the length of the previous cities
        all_idxs = [(i + city_lens_prefix_sum) for i in range(city_lens[city_idx])]

        for i in range(num_users_per_city):
            dict_users[i + city_idx * num_users_per_city] = set(np.random.choice(all_idxs, num_items, replace=False)) 
            all_idxs = list(set(all_idxs) - dict_users[i + city_idx * num_users_per_city])

        dict_users[(city_idx+1)*num_users_per_city -1] |=  set(all_idxs) # not drop last

    print('Time consumed to get non-iid user indices: {:.2f}s\n'.format((time.time() - timer)))

    return dict_users

def get_city_num(root_dir, train_folder):
    city_names = sorted(os.listdir(os.path.join(root_dir, train_folder))) 
    print("city_names: ", city_names)
    num_classes = len(city_names)
    print("num_classes: ", num_classes)

    city_lens = []
    for i in range(num_classes):
        city_lens.append(len(os.listdir(os.path.join(root_dir, train_folder, city_names[i]))))

    for i in range(num_classes):
        print(city_names[i], city_lens[i])

    print("city_lens: ", city_lens)
    return city_lens


if __name__ == '__main__':
    root_dir = '/home/fll/leo_test/data/cityscapes'
    split_root_dir = "/disk1/fll_data/cityscapes_split_erase20"
    user_groups = cityscapes_noniid_extend(split_root_dir, Cityscapes_Dataset.train_folder, num_users=152)

    def print_user_groups(user_groups): # non-iid split test
        data_sum = 0
        for i in range(len(user_groups)):  # len(user_groups) = 144 or 152
            print(i, user_groups[i])
            print("len(user_groups[{}]): ".format(i), len(user_groups[i]))
            data_sum += len(user_groups[i])
        print("data_sum: ", data_sum)

    print_user_groups(user_groups)





