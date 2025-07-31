import modal
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import cv2
from datasets import load_from_disk
from model import DenseNet121
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np

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

dermatology_mapping = {
        # CATEGORY 1: ACNE DISORDERS
        "Acne_Keloidalis_Nuchae": "ACNE_DISORDERS",
        "Acne_Vulgaris": "ACNE_DISORDERS", 
        "Pomade_Acne": "ACNE_DISORDERS",
        "Steroid_Acne": "ACNE_DISORDERS",
        # CATEGORY 2: FOLLICULAR DISORDERS
        "Pseudofolliculitis_Barbae": "FOLLICULAR_DISORDERS",
        "Pityrosporum_Folliculitis": "FOLLICULAR_DISORDERS",
        "Follicular_Mucinosis": "FOLLICULAR_DISORDERS",
        "Follicular_Retention_Cyst": "FOLLICULAR_DISORDERS",
        "Hidradenitis_Suppurativa": "FOLLICULAR_DISORDERS",
        "Kerion": "FOLLICULAR_DISORDERS",
        "Perioral_Dermatitis": "FOLLICULAR_DISORDERS",
        "Rosacea": "FOLLICULAR_DISORDERS",
        # CATEGORY 3: ECZEMA CONDITIONS
        "Acute_Eczema": "ECZEMA_CONDITIONS",
        "Dry_Skin_Eczema": "ECZEMA_CONDITIONS",
        "Dyshidrosiform_Eczema": "ECZEMA_CONDITIONS",
        "Eczema": "ECZEMA_CONDITIONS",
        "Infantile_Atopic_Dermatitis": "ECZEMA_CONDITIONS",
        "Nummular_Eczema": "ECZEMA_CONDITIONS",
        "Seborrheic_Dermatitis": "ECZEMA_CONDITIONS",
        "Stasis_Dermatitis": "ECZEMA_CONDITIONS",
        "Neurodermatitis": "ECZEMA_CONDITIONS",
        # CATEGORY 4: CONTACT DERMATITIS
        "Allergic_Contact_Dermatitis": "CONTACT_DERMATITIS",
        "Factitial_Dermatitis": "CONTACT_DERMATITIS",
        "Frictional_Lichenoid_Dermatitis": "CONTACT_DERMATITIS",
        "Drug_Eruption": "CONTACT_DERMATITIS",
        "Fixed_Drug_Eruption": "CONTACT_DERMATITIS",
        "Steroid_Use_abusemisuse_Dermatitis": "CONTACT_DERMATITIS",
        # CATEGORY 5: PSORIASIS SPECTRUM
        "Psoriasis": "PSORIASIS_SPECTRUM",
        "Guttate_Psoriasis": "PSORIASIS_SPECTRUM",
        "Inverse_Psoriasis": "PSORIASIS_SPECTRUM",
        "Pustular_Psoriasis": "PSORIASIS_SPECTRUM",
        "Scalp_Psoriasis": "PSORIASIS_SPECTRUM",
        "Mucous_Membrane_Psoriasis": "PSORIASIS_SPECTRUM",
        # CATEGORY 6: NAIL DISORDERS
        "Beau's_Lines": "NAIL_DISORDERS",
        "Clubbing_of_Fingers": "NAIL_DISORDERS",
        "Green_Nail": "NAIL_DISORDERS",
        "Half_and_Half_Nail": "NAIL_DISORDERS",
        "Koilonychia": "NAIL_DISORDERS",
        "Leukonychia": "NAIL_DISORDERS",
        "Median_Nail_Dystrophy": "NAIL_DISORDERS",
        "Nail_Dystrophy": "NAIL_DISORDERS",
        "Nail_Nevus": "NAIL_DISORDERS",
        "Nail_Psoriasis": "NAIL_DISORDERS",
        "Nail_Ridging": "NAIL_DISORDERS",
        "Onychogryphosis": "NAIL_DISORDERS",
        "Onycholysis": "NAIL_DISORDERS",
        "Onychomycosis": "NAIL_DISORDERS",
        "Onychoschizia": "NAIL_DISORDERS",
        "Paronychia": "NAIL_DISORDERS",
        "Pincer_Nail_Syndrome": "NAIL_DISORDERS",
        "Racquet_Nail": "NAIL_DISORDERS",
        "Subungual_Hematoma": "NAIL_DISORDERS",
        "Terry's_Nails": "NAIL_DISORDERS",
        # CATEGORY 7: BENIGN TUMORS
        "Angioma": "BENIGN_TUMORS",
        "Dermatofibroma": "BENIGN_TUMORS",
        "Digital_Fibroma": "BENIGN_TUMORS",
        "Fibroma": "BENIGN_TUMORS",
        "Fibroma_Molle": "BENIGN_TUMORS",
        "Keloid": "BENIGN_TUMORS",
        "Leiomyoma": "BENIGN_TUMORS",
        "Lipoma": "BENIGN_TUMORS",
        "Neurofibroma": "BENIGN_TUMORS",
        "Pyogenic_Granuloma": "BENIGN_TUMORS",
        "Skin_Tag": "BENIGN_TUMORS",
        "Strawberry_Hemangioma": "BENIGN_TUMORS",
        "Syringoma": "BENIGN_TUMORS",
        "Sebaceous_Gland_Hyperplasia": "BENIGN_TUMORS",
        "Seborrheic_Keratosis": "BENIGN_TUMORS",
        "Benign_Keratosis": "BENIGN_TUMORS",
        "Keratoacanthoma": "BENIGN_TUMORS",
        "Cutaneous_Horn": "BENIGN_TUMORS",
        "Dilated_Pore_of_Winer": "BENIGN_TUMORS",
        "Eccrine_Poroma": "BENIGN_TUMORS",
        "Ganglion": "BENIGN_TUMORS",
        "Granulation_Tissue": "BENIGN_TUMORS",
        "Lymphangioma_Circumscriptum": "BENIGN_TUMORS",
        "Myxoid_Cyst": "BENIGN_TUMORS",
        "Trichofolliculoma": "BENIGN_TUMORS",
        # CATEGORY 8: MELANOMA
        "Lentigo_Maligna_Melanoma": "MELANOMA",
        "Malignant_Melanoma": "MELANOMA",
        # CATEGORY 9: OTHER MALIGNANCIES
        "Basal_Cell_Carcinoma": "OTHER_MALIGNANCIES",
        "Bowen's_Disease": "OTHER_MALIGNANCIES",
        "Cutaneous_T-Cell_Lymphoma": "OTHER_MALIGNANCIES",
        "Metastatic_Carcinoma": "OTHER_MALIGNANCIES",
        # CATEGORY 10: NEVUS CONDITIONS
        "Becker's_Nevus": "NEVUS_CONDITIONS",
        "Blue_Nevus": "NEVUS_CONDITIONS",
        "Cafe_Au_Lait_Macule": "NEVUS_CONDITIONS",
        "Compound_Nevus": "NEVUS_CONDITIONS",
        "Congenital_Nevus": "NEVUS_CONDITIONS",
        "Dysplastic_Nevus": "NEVUS_CONDITIONS",
        "Epidermal_Nevus": "NEVUS_CONDITIONS",
        "Halo_Nevus": "NEVUS_CONDITIONS",
        "Junction_Nevus": "NEVUS_CONDITIONS",
        "Linear_Epidermal_Nevus": "NEVUS_CONDITIONS",
        "Nevus_Comedonicus": "NEVUS_CONDITIONS",
        "Nevus_Incipiens": "NEVUS_CONDITIONS",
        "Nevus_Sebaceous_of_Jadassohn": "NEVUS_CONDITIONS",
        "Nevus_Spilus": "NEVUS_CONDITIONS",
        # CATEGORY 11: BACTERIAL INFECTIONS
        "Cellulitis": "BACTERIAL_INFECTIONS",
        "Impetigo": "BACTERIAL_INFECTIONS",
        "Pitted_Keratolysis": "BACTERIAL_INFECTIONS",
        # CATEGORY 12: VIRAL INFECTIONS
        "Herpes_Simplex_Virus": "VIRAL_INFECTIONS",
        "Herpes_Zoster": "VIRAL_INFECTIONS",
        "Molluscum_Contagiosum": "VIRAL_INFECTIONS",
        "Varicella": "VIRAL_INFECTIONS",
        # CATEGORY 13: FUNGAL INFECTIONS
        "Candidiasis": "FUNGAL_INFECTIONS",
        "Tinea_Corporis": "FUNGAL_INFECTIONS",
        "Tinea_Cruris": "FUNGAL_INFECTIONS",
        "Tinea_Faciale": "FUNGAL_INFECTIONS",
        "Tinea_Manus": "FUNGAL_INFECTIONS",
        "Tinea_Pedis": "FUNGAL_INFECTIONS",
        "Tinea_Versicolor": "FUNGAL_INFECTIONS",
        # CATEGORY 14: PARASITIC INFECTIONS
        "Cutanea_Larva_Migrans": "PARASITIC_INFECTIONS",
        "Verruca_Vulgaris": "PARASITIC_INFECTIONS",
        # CATEGORY 15: AUTOIMMUNE CONDITIONS
        "Behcet's_Syndrome": "AUTOIMMUNE_CONDITIONS",
        "Discoid_Lupus_Erythematosus": "AUTOIMMUNE_CONDITIONS",
        "Morphea": "AUTOIMMUNE_CONDITIONS",
        "Pyoderma_Gangrenosum": "AUTOIMMUNE_CONDITIONS",
        "Leukocytoclastic_Vasculitis": "AUTOIMMUNE_CONDITIONS",
        "Vitiligo": "AUTOIMMUNE_CONDITIONS",
        # CATEGORY 16: INFLAMMATORY CONDITIONS
        "Erythema_Multiforme": "INFLAMMATORY_CONDITIONS",
        "Granuloma_Annulare": "INFLAMMATORY_CONDITIONS",
        "Lichen_Planus": "INFLAMMATORY_CONDITIONS",
        "Lichen_Sclerosis_Et_Atrophicus": "INFLAMMATORY_CONDITIONS",
        "Lichen_Simplex_Chronicus": "INFLAMMATORY_CONDITIONS",
        "Urticaria": "INFLAMMATORY_CONDITIONS",
        # CATEGORY 17: GENETIC CONDITIONS
        "Acrokeratosis_Verruciformis": "GENETIC_CONDITIONS",
        "Darier-White_Disease": "GENETIC_CONDITIONS",
        "Epithelioma_Adenoides_Cysticum": "GENETIC_CONDITIONS",
        "Hailey_Hailey_Disease": "GENETIC_CONDITIONS",
        "Hyperkeratosis_Palmaris_Et_Plantaris": "GENETIC_CONDITIONS",
        "Ichthyosis": "GENETIC_CONDITIONS",
        "Keratosis_Pilaris": "GENETIC_CONDITIONS",
        "Kyrle's_Disease": "GENETIC_CONDITIONS",
        # CATEGORY 18: SUN DAMAGE CONDITIONS
        "Actinic_solar_Damage(Actinic_Cheilitis)": "SUN_DAMAGE_CONDITIONS",
        "Actinic_solar_Damage(Actinic_Keratosis)": "SUN_DAMAGE_CONDITIONS",
        "Actinic_solar_Damage(Cutis_Rhomboidalis_Nuchae)": "SUN_DAMAGE_CONDITIONS",
        "Actinic_solar_Damage(Pigmentation)": "SUN_DAMAGE_CONDITIONS",
        "Actinic_solar_Damage(Solar_Elastosis)": "SUN_DAMAGE_CONDITIONS",
        "Actinic_solar_Damage(Solar_Purpura)": "SUN_DAMAGE_CONDITIONS",
        "Actinic_solar_Damage(Telangiectasia)": "SUN_DAMAGE_CONDITIONS",
        "Melasma": "SUN_DAMAGE_CONDITIONS",
        "Solar_Lentigo": "SUN_DAMAGE_CONDITIONS",
        "Favre_Racouchot": "SUN_DAMAGE_CONDITIONS",
        # CATEGORY 19: CYSTS
        "Apocrine_Hydrocystoma": "CYSTS",
        "Epidermoid_Cyst": "CYSTS",
        "Milia": "CYSTS",
        "Trichilemmal_Cyst": "CYSTS",
        "Chalazion": "CYSTS",
        # CATEGORY 20: HAIR DISORDERS
        "Alopecia_Areata": "HAIR_DISORDERS",
        "Androgenetic_Alopecia": "HAIR_DISORDERS",
        "Hypertrichosis": "HAIR_DISORDERS",
        "Scarring_Alopecia": "HAIR_DISORDERS",
        "Trichostasis_Spinulosa": "HAIR_DISORDERS",
        # CATEGORY 21: ORAL CONDITIONS
        "Angular_Cheilitis": "ORAL_CONDITIONS",
        "Aphthous_Ulcer": "ORAL_CONDITIONS",
        "Balanitis_Xerotica_Obliterans": "ORAL_CONDITIONS",
        "Geographic_Tongue": "ORAL_CONDITIONS",
        "Pearl_Penile_Papules": "ORAL_CONDITIONS",
        "Stomatitis": "ORAL_CONDITIONS",
        # CATEGORY 22: VASCULAR CONDITIONS
        "Cutis_Marmorata": "VASCULAR_CONDITIONS",
        "Erythema_Ab_Igne": "VASCULAR_CONDITIONS",
        "Livedo_Reticularis": "VASCULAR_CONDITIONS",
        "Radiodermatitis": "VASCULAR_CONDITIONS",
        "Stasis_Edema": "VASCULAR_CONDITIONS",
        "Stasis_Ulcer": "VASCULAR_CONDITIONS",
        "Schamberg's_Disease": "VASCULAR_CONDITIONS",
        # CATEGORY 23: KERATOTIC CONDITIONS
        "Arsenical_Keratosis": "KERATOTIC_CONDITIONS",
        "Callus": "KERATOTIC_CONDITIONS",
        "Disseminated_Actinic_Porokeratosis": "KERATOTIC_CONDITIONS",
        "Keratolysis_Exfoliativa_of_Wende": "KERATOTIC_CONDITIONS",
        "Mal_Perforans": "KERATOTIC_CONDITIONS",
        "Rhinophyma": "KERATOTIC_CONDITIONS",
        "Pseudorhinophyma": "KERATOTIC_CONDITIONS",
        # CATEGORY 24: PAPULOSQUAMOUS CONDITIONS
        "Pityriasis_Alba": "PAPULOSQUAMOUS_CONDITIONS",
        "Pityriasis_Rosea": "PAPULOSQUAMOUS_CONDITIONS",
        "Lichen_Spinulosis": "PAPULOSQUAMOUS_CONDITIONS",
        "Dermatosis_Papulosa_Nigra": "PAPULOSQUAMOUS_CONDITIONS",
        # CATEGORY 25: REACTIVE CONDITIONS
        "Erythema_Annulare_Centrifigum": "REACTIVE_CONDITIONS",
        "Erythema_Craquele": "REACTIVE_CONDITIONS",
        "Exfoliative_Erythroderma": "REACTIVE_CONDITIONS",
        "Lymphocytic_Infiltrate_of_Jessner": "REACTIVE_CONDITIONS",
        "Lymphomatoid_Papulosis": "REACTIVE_CONDITIONS",
        "Poikiloderma_Atrophicans_Vasculare": "REACTIVE_CONDITIONS",
        # CATEGORY 26: ATROPHIC CONDITIONS
        "Scar": "ATROPHIC_CONDITIONS",
        "Steroid_Striae": "ATROPHIC_CONDITIONS",
        "Striae": "ATROPHIC_CONDITIONS",
        "Xerosis": "ATROPHIC_CONDITIONS",
        # CATEGORY 27: BEHAVIORAL CONDITIONS
        "Neurotic_Excoriations": "BEHAVIORAL_CONDITIONS",
        "Desquamation": "BEHAVIORAL_CONDITIONS",
        # CATEGORY 28: SYSTEMIC CONDITIONS
        "Crowe's_Sign": "SYSTEMIC_CONDITIONS",
        "Histiocytosis_X": "SYSTEMIC_CONDITIONS",
        "Mucha_Habermann_Disease": "SYSTEMIC_CONDITIONS",
        "Bowenoid_Papulosis": "SYSTEMIC_CONDITIONS",
        # CATEGORY 29: ANATOMICAL VARIANTS
        "Fordyce_Spots": "ANATOMICAL_VARIANTS",
        "Toe_Deformity": "ANATOMICAL_VARIANTS",
        "Ulcer": "ANATOMICAL_VARIANTS",
        "Wound_Infection": "ANATOMICAL_VARIANTS"
    }
category_to_number = {
        "ACNE_DISORDERS": 0,
        "FOLLICULAR_DISORDERS": 1,
        "ECZEMA_CONDITIONS": 2,
        "CONTACT_DERMATITIS": 3,
        "PSORIASIS_SPECTRUM": 4,
        "NAIL_DISORDERS": 5,
        "BENIGN_TUMORS": 6,
        "MELANOMA": 7,
        "OTHER_MALIGNANCIES": 8,
        "NEVUS_CONDITIONS": 9,
        "BACTERIAL_INFECTIONS": 10,
        "VIRAL_INFECTIONS": 11,
        "FUNGAL_INFECTIONS": 12,
        "PARASITIC_INFECTIONS": 13,
        "AUTOIMMUNE_CONDITIONS": 14,
        "INFLAMMATORY_CONDITIONS": 15,
        "GENETIC_CONDITIONS": 16,
        "SUN_DAMAGE_CONDITIONS": 17,
        "CYSTS": 18,
        "HAIR_DISORDERS": 19,
        "ORAL_CONDITIONS": 20,
        "VASCULAR_CONDITIONS": 21,
        "KERATOTIC_CONDITIONS": 22,
        "PAPULOSQUAMOUS_CONDITIONS": 23,
        "REACTIVE_CONDITIONS": 24,
        "ATROPHIC_CONDITIONS": 25,
        "BEHAVIORAL_CONDITIONS": 26,
        "SYSTEMIC_CONDITIONS": 27,
        "ANATOMICAL_VARIANTS": 28
    }
# Dataloader for the data
class SkinImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        super(SkinImageDataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        # Loading the data
        self.dataset = load_from_disk(self.image_dir)
        self.new_label_mapping = self.get_class_name_mapping()

    def get_class_name_mapping(self):
        train_data = self.dataset["train"]
        label_names = train_data.features["label"].names

        label_name_mapping = {idx: name for idx, name in enumerate(label_names)}
        # mapping the old labels to the new ones
        newLabelMapping = {}
        for index, origLabel in label_name_mapping.items():
            newName = dermatology_mapping[origLabel]
            newIndex = category_to_number[newName]
            newLabelMapping[index] = newIndex
        return newLabelMapping

    def __len__(self):
        return self.dataset["train"].num_rows

    def __getitem__(self, idx):
        idx = int(idx)
        img, lbl = self.dataset["train"][idx]["image"], self.dataset['train'][idx]["label"]
        # debugging
        import numpy as np
        img = img.resize((1500, 1500))
        img_array = np.array(img).astype(np.float32) / 255.0
        if img_array.max() == 0:
            print(f"Zero PIL image at index {idx}")
        img_array = img_array.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_array)

        newLabel = self.new_label_mapping[lbl]
        return img_tensor, newLabel


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
    try:
        dataset = load_from_disk("/data/sd-198-metadata")
        train_data = dataset["train"]
        label_names = train_data.features["label"].names

        label_name_mapping = {idx: name for idx, name in enumerate(label_names)}
        # mapping the old labels to the new ones
        newLabelMapping = {}
        for index, origLabel in label_name_mapping.items():
            newName = dermatology_mapping[origLabel]
            newIndex = category_to_number[newName]
            newLabelMapping[index] = newIndex
        # print(newLabelMapping)
        newLabels = []
        for data in train_data:
            origLabel = data["label"]
            new_label = newLabelMapping[origLabel]
            newLabels.append(new_label)
        return train_data, newLabels
    except Exception as e:
        print(f"Error occured in loading the dataset {e}")


@app.function(image=image, volumes={"/data": volume, "/model": model_volume}, gpu="A10G", timeout=3600*5)
def train():
    # from sklearn.metrics import top_k_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    import torch.optim as optim

    print("Training the model...")
    
    labels = [i for _, i in category_to_number.items()]
    classes = [c for c, _ in category_to_number.items()]
    num_classes = len(labels)
    train_data, all_labels = fetch_dataset_information.remote()
    # training params initialization
    # for image in train_data[0:1]["image"]:
    #     image = image.resize((1500, 1500))
    #     print(np.array(image)/255, np.size(np.array(image)/255.0, 1))
    # exit()
    epochs = 15
    criterion = nn.CrossEntropyLoss()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # print(f"training data : {train_data[:]["image"]}")
    # main training part
    for _, (train_idx, val_idx) in enumerate(skf.split(train_data[:]["image"], all_labels)):
        train_subset = Subset(SkinImageDataset("/data/sd-198-metadata"), train_idx)
        val_subset = Subset(SkinImageDataset("/data/sd-198-metadata"), val_idx)
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
                data, label = data.to(device), label.to(device)
                outputs = model(data)
                # print(f"these are the output values : {outputs} and the size is : {outputs.size()}")
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
