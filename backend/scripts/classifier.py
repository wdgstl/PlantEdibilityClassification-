import sys
from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm


# Add the parent directory of 'scripts' to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from scripts.utils import load_model  
except ModuleNotFoundError:
    from backend.scripts.utils import load_model 

from torchvision.models import resnet18
from torchvision import models, transforms
import torch
from PIL import Image
import json
import subprocess
import sys

class classifier:
    def __init__(self, model_path=None, class_mapping_path=None):
        self.default_model_url = 'https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/files/?p=%2Fresnet18_weights_best_acc.tar&dl=1'
        self.default_class_mapping_url = 'https://storage.googleapis.com/kagglesdsdata/datasets/1981237/3270629/plantnet_300K/plantnet300K_species_names.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250408%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250408T152507Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a51809678bed4db86b892ba53ddf4daf9cad76d32d9d5fc59027027e5f2781e614387cf6c4d97dc3b31189d270758d2a8660226852a689d8982fae56ba056796086fa36cdb16edfe182c88f824b2b32cb08ee279d967b8f81ad81d16daaf84c625ec4baf43929484de385eb299de2d239f739017640173620e30dbe409fb44cc43b62d663f637bb03f33595833f8aff407a36340bcf74a3a74bf71ff7918633e6082dba1b6e0aa452096650cece72b65a31d657a410962041e0b18df92d0048798be889e3c6364359857b004762d4981b52d655d1ffb40812ba3fff9730a18bfaa4207e6b1239e59d77a4124cb46ff480a868d9cd1c80d55c9baca2688c80353'
        self.model =  resnet18(num_classes = 1081)
        #supplied custom model
        if model_path:
            self.model_path = model_path
        #otherwise, pull defaultc
        else:
            print(f"Downloading model from {self.default_model_url}\n")
            output_file = "../data/models/resnet18_weights_best_acc.tar"
            subprocess.run(["wget", "-O", output_file, self.default_model_url])
            self.model_path = "../data/models/resnet18_weights_best_acc.tar"
            print("Download Complete")
        
        if class_mapping_path:
            self.class_mapping_path = class_mapping_path
        else:
            print(f"Downloading class mapping json from {self.default_class_mapping_url}\n")
            output_file = "../data/class_mapping/plantnet300K_species_names.json"         
            subprocess.run(["wget", "-O", output_file, self.default_class_mapping_url])
            self.class_mapping_path = '../data/class_mapping/plantnet300K_species_names.json'
            print("Download Complete")
        load_model(self.model, filename=self.model_path, use_gpu=False)
            
    
    def predict(self, image_path):
        self.model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]    
            )
        ])
        img = Image.open(image_path).convert("RGB")
        img_t = preprocess(img)
        batch_t = img_t.unsqueeze(0) 
        with torch.no_grad():  
            out = self.model(batch_t) 

        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        with open(self.class_mapping_path) as f:
            species_dict = json.load(f)
        id_to_species = [species_dict[k] for k in sorted(species_dict.keys(), key=int)]
        output = [] 
        for i in range(top5_prob.size(0)):
            output.append((id_to_species[top5_catid[i]], top5_prob[i].item()))
        return output
    
    def get_metrics(self, path_to_test_parquets):
        dfs = []
        for filename in os.listdir(path_to_test_parquets):
            if filename.endswith(".parquet"):
                file_path = os.path.join(path_to_test_parquets, filename)
                print(f"Processing: {file_path}")
                df = pd.read_parquet(file_path)
                dfs.append(df)
        full_df_test = pd.concat(dfs, ignore_index=True)
        with open(self.class_mapping_path) as f:
            species_dict = json.load(f)
        i = 0
        map_from_df_to_map = {}
        for key in species_dict:
            map_from_df_to_map[i] = key
            i+=1
        full_df_test['species_true'] = full_df_test['label'].map(lambda x: species_dict.get(map_from_df_to_map.get(x)))

        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        print("Running model on test data")
        for idx, row in tqdm(full_df_test.iterrows(), total=len(full_df_test)):
            img_bytes = row['image']['bytes']
            
            temp_path = temp_dir / f"temp_{idx}.jpg"

            with open(temp_path, "wb") as f:
                f.write(img_bytes)

            predictions = self.predict(image_path=temp_path)
            top_5 = []
            for pred in predictions:
                top_5.append(pred[0])

            top_species, _ = max(predictions, key=lambda x: x[1])
            full_df_test.at[idx, 'top_prediction'] = top_species
            full_df_test.at[idx, 'top_5'] = top_5

            try:
                temp_path.unlink()  
            except Exception as e:
                print(f"Failed to delete {temp_path}: {e}")

        metrics = {}
        species_accuracy = (full_df_test['species_true'] == full_df_test['top_prediction']).mean()
        genus_accuracy = (full_df_test['species_true'].str.split("_").str[0] == full_df_test['top_prediction'].str.split("_").str[0] ).mean()
        top_5_species_accuracy = full_df_test.apply(lambda row: row['species_true'] in row['top_5'], axis=1).mean()
        top_5_genus_accuracy = full_df_test.apply(lambda row: row['species_true'].split("_")[0] in [pred.split("_")[0] for pred in row['top_5']],axis=1).mean()
        report = classification_report(
            full_df_test['species_true'],
            full_df_test['top_prediction'],
            zero_division=0,
            output_dict=True
        )

        weighted_precision = report['weighted avg']['precision']
        weighted_recall = report['weighted avg']['recall']
        weighted_f1 = report['weighted avg']['f1-score']
        metrics.update({
            'species_accuracy': species_accuracy,
            'genus_accuracy': genus_accuracy,
            'top_5_species_accuracy': top_5_species_accuracy,
            'top_5_genus_accuracy': top_5_genus_accuracy,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        })
        return metrics


def main():
    #Option 1: Specify Path
    plant_classifier = classifier(model_path='../data/models/resnet18_weights_best_acc.tar', class_mapping_path= '../data/class_mapping/plantnet300K_species_names.json')
    #Option 3: Load Model + Class Mapping from Cloud
    # plant_classifier = classifier()
    argv = sys.argv
    image_path = argv[1]

    predictions = plant_classifier.predict(image_path=image_path)
    for p in predictions:
        print(p)

if __name__ == "__main__":
    main()        





