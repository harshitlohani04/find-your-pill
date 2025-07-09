import modal
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import cv2
import pandas as pd
from model import DenseNet121
from torch.utils.data import Dataset
import os

app = modal.Stub("find-your-pill-train")

image = (modal.Image.debian_slim()
        .pip_install_from_requirements("requirements.txt")
        .apt_install(["wget", "ffmpeg", "unzip"])
        .add_local_python_source("model")
        )

volume = modal.Volume.from_name("pill-classifier-data", create_if_missing=True)
model_volume = modal.Volume.from_name("pill-classifier-model", create_if_missing=True)

class PillDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None):
        super(PillDataset, self).__init__()
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx, 1]
        return image, label

@app.function(image=image, volumes={"/data": volume, "/model": model_volume}, gpu="A10G", timeout=3600*5)
def train(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    train.remote()
