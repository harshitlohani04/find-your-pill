import modal
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import cv2
from datasets import load_from_disk
from model import DenseNet121
from torch.utils.data import Dataset, Subset
import os
import json

volume = modal.Volume.from_name("pill-classifier-data", create_if_missing=True)
model_volume = modal.Volume.from_name("pill-classifier-model", create_if_missing=True)

app = modal.App("find-your-pill-train",
                 secrets=[modal.Secret.from_name("huggingface-token")],
                 volumes={"/data": volume, "/model": model_volume})

image = (modal.Image.debian_slim()
        .pip_install_from_requirements("requirements.txt")
        .apt_install(["wget", "ffmpeg", "unzip"])
        .add_local_python_source("model")
        )

# Dataloader for the data
class SkinImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        super(SkinImageDataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform

        # Loading the data
        self.dataset = load_from_disk(self.image_dir)

    @property
    def get_class_name_mapping(self):
        train_data = self.dataset["train"]
        label_names = train_data.features["labels"].names

        label_name_mapping = {idx: name for idx, name in enumerate(label_names)}
        return label_name_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, lbl = self.dataset["train"][idx]["image"], self.dataset['train'][idx]["label"]
        if self.transform:
            img = self.transform(img)
        return img, lbl
    

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-token")], volumes={"/data": volume}, timeout=3600)
def download_data():
    print("Downloading data...")
    try:
        import os
        from huggingface_hub import login
        from datasets import load_dataset

        # condition for existence check
        if os.path.exists("/data/sd-198-metadata"):
            print("skipping downloading as the data already exists")
            return

        login(token=os.environ["HUGGINGFACE_SECRET_KEY"])
        # Creating dir for image data
        os.makedirs("/data/sd-198-metadata", exist_ok=True)
        # loading the dataset
        dataset = load_dataset("resyhgerwshshgdfghsdfgh/SD-198")
        dataset.save_to_disk("/data/sd-198-metadata")
    except Exception as e:
        print(f"Error in downloading data: {e}")

# @app.function(image=image, volumes={"/data":volume})
# def load_dataset():
#     from datasets import load_from_disk
#     print("Loading the data..")
#     try:
#         dataset = load_from_disk("/data/sd-198-metadata")
#         train_data = dataset["train"]

#         # for i in range(len(train_data)):
#         #     print(train_data[i])
#         print(train_data[0]["image"], train_data[0]["label"])
#         train_data[0]["image"].save("/data/sample.jpg")
#     except Exception as e:
#         print(f"Error occured in loading the dataset {e}")


@app.function(image=image, volumes={"/data": volume, "/model": model_volume}, gpu="A10G", timeout=3600*5)
def train():
    from sklearn.metrics import top_k_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    print("Training the model...")
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10), # Rotate between -10 and 10 degrees
        transforms.ToTensor()
    ])
    dataloader = SkinImageDataset(image_dir="/data/sdn-198-metadata", transform=None)
    

@app.local_entrypoint()
def main():
    # download_data.remote()
    # load_dataset.remote()
    # train.remote()
    pass
