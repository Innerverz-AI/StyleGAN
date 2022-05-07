from io import BytesIO

import os
import glob
import lmdb
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from lib import utils


class SingleFaceDataset(Dataset):
    def __init__(self, dataset_root_list, transform, resolution=8, isMaster=False):
        self.image_path_list, self.image_num_list = utils.get_all_images(dataset_root_list)

        self.resolution = resolution    # Change this when scale-jump occurs 
        self.transform = transform

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):
        idx = 0
        while item >= self.image_num_list[idx]:
            item -= self.image_num_list[idx]
            idx += 1
        image_path = self.image_path_list[idx][item]
        
        Xs = Image.open(image_path).convert("RGB")

        return self.transform(Xs)

    def __len__(self):
        return sum(self.image_num_list)



class PairedFaceDatasetTrain(Dataset):
    def __init__(self, dataset_root_list, isMaster, same_prob=0.2):
        self.same_prob = same_prob
        self.image_path_list, self.image_num_list = utils.get_all_images(dataset_root_list)

        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):
        idx = 0
        while item >= self.image_num_list[idx]:
            item -= self.image_num_list[idx]
            idx += 1
        image_path = self.image_path_list[idx][item]
        
        Xs = Image.open(image_path).convert("RGB")

        if random.random() > self.same_prob:
            image_path = random.choice(self.image_path_list[random.randint(0, len(self.image_path_list)-1)])
            Xt = Image.open(image_path).convert("RGB")
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.image_num_list)


class PairedFaceDatasetValid(Dataset): # for pggan implementation
    def __init__(self, valid_data_dir, isMaster):
        
        self.source_path_list = sorted(glob.glob(f"{valid_data_dir}/source/*.*g"))
        self.target_path_list = sorted(glob.glob(f"{valid_data_dir}/target/*.*g"))

        # take the smaller number if two dirs have different numbers of images
        self.num = min(len(self.source_path_list), len(self.target_path_list))
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the validation.")

    def __getitem__(self, idx):
        
        Xs = Image.open(self.source_path_list[idx]).convert("RGB")
        Xt = Image.open(self.target_path_list[idx]).convert("RGB")

        return self.transforms(Xs), self.transforms(Xt)

    def __len__(self):
        return self.num


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8, isMaster=False):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution    # Change this when scale-jump occurs 
        self.transform = transform

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for training.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
