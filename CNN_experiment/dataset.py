import os
import zipfile
import urllib.request
import torch
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np


class Coil100FeatureDataset(Dataset):
    def __init__(self, root_dir="./data", object_id=1, download=True):
        self.root_dir = root_dir
        self.coil_dir = os.path.join(root_dir, "coil-100")
        self.object_id = object_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if download:
            self._download_and_extract()
            
        self.resnet = models.resnet18(pretrained=True).to(self.device)
        self.resnet.eval()

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.features, self.noisy_features, self.angles = self._extract_features()

    def _extract_features(self):
        features_list = []
        angles_list = []
        
        for angle in tqdm(range(0, 360, 5)):
            filename = f"obj{self.object_id}__{angle}.png"
            filepath = os.path.join(self.coil_dir, filename)
            
            if not os.path.exists(filepath):
                continue 
            
            img = Image.open(filepath).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.resnet(img_tensor).squeeze()
            
            features_list.append(logits.cpu())
            
            rads = np.deg2rad(angle)
            angles_list.append(torch.tensor(rads, dtype=torch.float32))
            
        X = torch.stack(features_list)
        noise = 0.1 * torch.randn_like(X)
        X_noisy = X + noise
        y = torch.stack(angles_list)
        
        return X, X_noisy, y

    def _download_and_extract(self):
        if not os.path.exists(self.coil_dir):
            url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
            zip_path = os.path.join(self.root_dir, "coil-100.zip")
            
            if not os.path.exists(self.root_dir):
                os.makedirs(self.root_dir)
            
            if not os.path.exists(zip_path):
                print("Downloading COIL-100 dataset (~130MB)...")
                urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Extraction Complete.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.noisy_features[idx], self.angles[idx]
