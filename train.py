import modal
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import cv2
from datasets import load_from_disk
from model import DenseNet121
from torch.utils.data import Dataset, Subset, DataLoader
import os
import json
import tqdm

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
        return len(self.label_names)

    def __getitem__(self, idx):
        idx = int(idx)
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

@app.function(image=image, volumes={"/data":volume})
def fetch_dataset_information():
    from datasets import load_from_disk
    try:
        dataset = load_from_disk("/data/sd-198-metadata")
        train_data = dataset["train"]
        labels = []
        for img in train_data:
            labels.append(img["label"])
        classes = set(labels)

        return labels, classes, len(classes), train_data
    except Exception as e:
        print(f"Error occured in loading the dataset {e}")


@app.function(image=image, volumes={"/data": volume, "/model": model_volume}, gpu="A10G", timeout=3600*5)
def train():
    # from sklearn.metrics import top_k_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    import torch.optim as optim

    print("Training the model...")
    image_transform = transforms.Compose([
        transforms.Resize((1500, 1500)),
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])
    labels, classes, num_classes, train_data = fetch_dataset_information.remote()
    print(f"the number of classes are : {num_classes} and the classes are : {classes}")
    # training params initialization
    epochs = 100
    criterion = nn.CrossEntropyLoss()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # main training part
    for _, (train_idx, val_idx) in enumerate(skf.split(train_data, labels)):
        train_subset = Subset(SkinImageDataset("/data/sd-198-metadata", transform=image_transform), train_idx)
        val_subset = Subset(SkinImageDataset("/data/sd-198-metadata", transform=image_transform), val_idx)
        # initializing the dataloaders
        trainLoader = DataLoader(
            train_subset,
            shuffle=True,
            batch_size=16
        )
        valLoader = DataLoader(
            val_subset,
            batch_size=16,
            shuffle=False
        )
        #instantiating the model
        model = DenseNet121(num_classes=num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.01)
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.001, epochs=epochs, steps_per_epoch=len(trainLoader))

        best_accuracy = 0
        for epoch in range(epochs):
            # main training
            model.train()
            epoch_loss = 0
            for data, label in trainLoader:
                print(label)
                data, label = data.to(device), label.to(device)
                outputs = model(data)
                print(f"these are the output values : {outputs} and the size is : {outputs.size()}")
                loss = criterion(outputs, label)
                epoch_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()

            avg_epoch_loss = epoch_loss/len(trainLoader)
            # validation loop
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, label in valLoader:
                    data, label = data.to(device), label.to(device)
                    outputs = model(data)

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == label).sum().item()
                    total += label.size(0)

                    val_loss += criterion(outputs, label)
                avg_val_loss = val_loss/len(valLoader)
                accuracy = 100*correct/total
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save({
                        "model_state_dict":model.state_dict(),
                        "accuracy": best_accuracy,
                        "epoch": epoch
                    }, "/model/best_model.pth")
            print(f"Epoch{epoch} average epoch loss : {avg_epoch_loss} -- average val loss : {avg_val_loss} -- best accuracy : {best_accuracy}")

@app.local_entrypoint()
def main():
    # download_data.remote()
    # load_dataset.remote()
    train.remote()
