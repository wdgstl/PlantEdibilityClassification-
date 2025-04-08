import subprocess
from pathlib import Path
import tarfile

#NEED TO FIX
def is_tar_valid(tar_path):
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.getmembers()  
        return True
    except tarfile.ReadError:
        return False

def pull_data():
    model_url = 'https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/files/?p=%2Fresnet18_weights_best_acc.tar&dl=1'
    class_mapping_url = 'https://storage.googleapis.com/kagglesdsdata/datasets/1981237/3270629/plantnet_300K/plantnet300K_species_names.json?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250408%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250408T152507Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a51809678bed4db86b892ba53ddf4daf9cad76d32d9d5fc59027027e5f2781e614387cf6c4d97dc3b31189d270758d2a8660226852a689d8982fae56ba056796086fa36cdb16edfe182c88f824b2b32cb08ee279d967b8f81ad81d16daaf84c625ec4baf43929484de385eb299de2d239f739017640173620e30dbe409fb44cc43b62d663f637bb03f33595833f8aff407a36340bcf74a3a74bf71ff7918633e6082dba1b6e0aa452096650cece72b65a31d657a410962041e0b18df92d0048798be889e3c6364359857b004762d4981b52d655d1ffb40812ba3fff9730a18bfaa4207e6b1239e59d77a4124cb46ff480a868d9cd1c80d55c9baca2688c80353'

    base_dir = Path(__file__).resolve().parent.parent

    models_dir = base_dir / "data/models"
    mapping_dir = base_dir / "data/class_mapping"

    models_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir.mkdir(parents=True, exist_ok=True)

    model_output = models_dir / "resnet18_weights_best_acc.tar"
    class_mapping_output = mapping_dir / "plantnet300K_species_names.json"

   
    if not Path(model_output).exists() or not is_tar_valid(model_output):
        subprocess.run(["wget", "-O", str(model_output), model_url])
    if not Path(class_mapping_output).exists():
        subprocess.run(["wget", "-O", str(class_mapping_output), class_mapping_url])

