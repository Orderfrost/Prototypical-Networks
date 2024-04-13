# coding=utf-8
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class GeneralDataset(data.Dataset):
    """
    Provide a General Dataset
    """

    def __init__(self, mode='train', transform=None, target_transform=None, data_folder='..' + os.sep + 'dataset'):
        """
        初始化一个通用的数据类。
        Args:
        - data_folder: 包含图像和标签信息的文件夹路径
        - transform: 应用与输入图像的转换
        - target_transform: 应用于标签的转换

        """
        super(GeneralDataset, self).__init__()
        self.data_folder = data_folder
        self.mode = mode

        # Convert image to tensor channel first, then normalize it
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1.0 - x),  # Invert the image
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
            ])
        self.target_transform = target_transform

        print("now mode: ", mode)

        if not isinstance(mode, str) or mode not in ['train', 'val', 'test']:
            raise ValueError(f'mode must be one of [train, val, test]')

        # 根据 mode 创建文件夹
        self.data_folder = os.path.join(data_folder, mode)

        self.all_items = self._find_items(self.data_folder)
        self.classes = self._index_classes(self.all_items)

        print("==  now loading img! ==\n")
        self.img_path = self._get_paths()
        self.img = map(self._load_img, self.img_path)
        self.img = list(self.img)


    def __getitem__(self, idx):
        target = self.classes[idx]
        img_path, _ = self.all_items[idx]
        img = self.img[idx]
        # img = Image.open(img_path).convert('RGB')
        # img = img.resize((28, 28))
        # img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _find_items(self, data_folder):
        '''
        从文件夹中找到所有的图像和标签
        '''
        items = []
        # flag = True
        for root, dirs, files in os.walk(data_folder):
            # if flag:
            #     print("== root: ", root)
            #     print("== dirs: ", dirs)
            #     print("== files: ", files)
            #     flag = False

            for dir in dirs:
                label = dir
                dir_path = os.path.join(root, dir)
                for file in os.listdir(os.path.join(root, dir)):
                    if file.endswith(('.jpg', 'jpeg', 'png')):
                        items.append((os.path.join(dir_path, file), label))
        print("== Dataset: Found %d items in %s" % (len(items), data_folder))
        return items

    def _index_classes(self, items):
        '''
        为所有的类别建立索引
        '''
        classes = sorted(set([item[1] for item in items]))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        print("== Dataset: Found %d classes " % len(classes))
        return [class_to_idx[item[1]] for item in items]

    def _load_img(self, img_path):
        """
        加载图像
        """
        img = Image.open(img_path).convert('RGB')
        img = img.resize((28, 28))
        img = self.transform(img)

        return img

    def _get_paths(self):
        """
        返回所有图像的路径并且得到
        """
        paths = []
        for path, _ in self.all_items:
            paths.append(path)
        return paths

    def _num_classes(self):
        """
        返回类别的数量
        """
        idx = {}
        flag = True
