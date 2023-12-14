import os

%matplotlib inline
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from IPython.display import display

import numpy as np
from glob import glob

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transform
from torchvision.datasets import ImageFolder

# 데이터셋 불러오기
train_dir = os.path.join("/content/dogs-vs-cats", "train")
test_dir = os.path.join("/content/dogs-vs-cats", "test")
classes = ['cat', 'dog']
class_info = {idx : os.path.basename(cls) for idx, cls in enumerate(classes)}

img_files = glob(f"{train_dir}/*.jpg")
dataset = []

for img_file in img_files:
  for cls in classes:
    if cls in os.path.basename(img_file):
      label = cls
  dataset.append([img_file, label])

dataset = np.array(dataset)
X = dataset[:, 0]
Y = dataset[:, 1]

x_train, x_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.7, random_state=724)  # 훈련용 70%, 검증용 30%

# 데이터셋 경로 관리
def create_symlink(x_target, original, target):
  for x in x_target:
    src = os.path.abspath(x)
    dst = src.replace(original, target)

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if not os.path.exists(dst):
      os.symlink(src, dst)

create_symlink(x_train, "train", "train_dataset")
create_symlink(x_valid, "train", "valid_dataset")

# GPU 사용
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

batch_size = 64

# 데이터 전처리
train_transform = transform.Compose([
    transform.Resize((256, 256)),
    transform.RandomCrop(224),
    transform.RandomHorizontalFlip(),
    transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transform.Compose([
    transform.Resize((224, 224)),
    transform.ToTensor(),
    transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CatDogDataset(Dataset):
  def __init__(self, directory, transform=None):
    self.directory = os.path.abspath(directory)
    self.filelist = glob(self.directory + '/*.jpg')
    self.transform = transform
  
  def __len__(self):
    return len(self.filelist)
  
  def __getitem__(self, idx):
    filename = self.filelist[idx]
    img = self.get_image(filename)
    label = self.get_label(filename)

    return img, label
