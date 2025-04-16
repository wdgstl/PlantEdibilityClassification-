import sys
from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch.nn as nn


# Add the parent directory of 'scripts' to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from scripts.utils import load_model  
except ModuleNotFoundError:
    from backend.scripts.utils import load_model 

from torchvision.models import resnet18, resnet50, densenet121, mobilenet_v3_large, shufflenet_v2_x1_0, squeezenet1_0, efficientnet_b0, densenet201
from torchvision import models, transforms
import torch
from PIL import Image
import json
import subprocess
import sys

class classifier:
    def __init__(self, model_type = resnet18, model_path=None, class_mapping_path=None):
        self.model_name = model_type.__name__
        if self.model_name == 'shufflenet_v2_x1_0':
            self.model_name = 'shufflenet'
        elif self.model_name == 'squeezenet1_0':
            self.model_name = 'squeezenet'
        self.default_model_url = f'https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/files/?p=%2F{self.model_name}_weights_best_acc.tar&dl=1'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if self.model_name == 'inception_v3':
            self.model = model_type(num_classes = 1081, aux_logits = False)
        elif self.model_name == 'squeezenet1_0':
            self.model = model_type()
            self.model.classifier[1] = nn.Conv2d(
                in_channels=512,
                out_channels=1081,  
                kernel_size=(1, 1),
                stride=(1, 1)
            )
            self.model.num_classes = 1081 
        else:
            self.model = model_type(num_classes = 1081)
        #supplied custom model
        if model_path:
            self.model_path = model_path
        #otherwise, pull default
        else:
            print(f"Downloading model from {self.default_model_url}\n")
            output_file = os.path.join(base_dir, "..", "data", "models", f"{self.model_name}_weights_best_acc.tar")
            subprocess.run(["wget", "-O", output_file, self.default_model_url])
            self.model_path = output_file
            print("Download Complete")
        
        output_file = os.path.join(base_dir, "..", "data", "class_mapping", "plantnet300K_species_names.json")
        self.class_mapping_path = output_file
        load_model(self.model, filename=self.model_path, use_gpu=False)
            
    
    def predict(self, image_path):
        self.model.eval()
        if self.model_name == "inception_v3":
            input_size = 299
            resize_size = 342
        elif self.model_name == "efficientnet_b0":
            input_size = 224  
            resize_size = int(input_size * 1.14) 
        else:
            input_size = 224
            resize_size = 256
        preprocess = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
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

        full_df_test['top_5'] = None
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





