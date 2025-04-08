import sys
from pathlib import Path

# Add the parent directory of 'scripts' to the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.classifier import * 
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import os
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

model = classifier(
    model_path=str(BASE_DIR.parent / "data/models/resnet18_weights_best_acc.tar"),
    class_mapping_path=str(BASE_DIR.parent / "data/class_mapping/plantnet300K_species_names.json")
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        results = model.predict(str(temp_file_path))
        os.remove(temp_file_path)
        return {"prediction": results}
    except Exception as e:
        os.remove(temp_file_path)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)