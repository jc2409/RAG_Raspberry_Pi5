import os
import kagglehub

# Sign-in to Kaggle
kagglehub.login()

BASE_PATH = "."

DATA_DIR = os.path.join(BASE_PATH, "data")

os.makedirs(DATA_DIR, exist_ok=True)

os.environ['KAGGLEHUB_CACHE'] = DATA_DIR

# Download latest version
path = kagglehub.dataset_download("rtatman/questionanswer-dataset")

print("Path to dataset files:", path)