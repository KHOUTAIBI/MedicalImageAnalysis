import os
import zipfile
import urllib.request
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np

class Coil100FeatureDataset(Dataset):
    def __init__(self, root_dir="./coil-100", object_id=1, download=True):
        """
        object_id: which object to study (1 to 100).
        """
        self.root_dir = root_dir
        self.object_id = object_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim = 512  # ResNet18 avgpool dim

        if download:
            self._download_and_extract()
            
        resnet = models.resnet18(pretrained=True).to(self.device)
        resnet.eval()
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # We "Record" the Neurons
        # We pre-compute all features now so training is instant later.
        print(f"Extracting ResNet features for COIL Object #{object_id}...")
        self.features, self.angles = self._extract_features()
        print(f"dataset ready: {self.features.shape} tensor.")

    def _extract_features(self):
        features_list = []
        angles_list = []
        
        # COIL-100 has 72 images per object (0, 5, 10 ... 355 degrees)
        for angle in tqdm(range(0, 360, 5)):
            # Filename format: obj1__0.png, obj1__5.png ...
            filename = f"obj{self.object_id}__{angle}.png"
            filepath = os.path.join(self.root_dir, filename)
            
            if not os.path.exists(filepath):
                # Fallback for weird naming conventions in some unzipped versions
                # Sometimes it is obj1__0.png, sometimes obj1_0.png
                continue 
            
            # Load & Process Image
            img = Image.open(filepath).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            # Forward Pass (No Grad)
            with torch.no_grad():
                # Output shape: (1, 512, 1, 1) -> Flatten to (512)
                feature = self.feature_extractor(img_tensor).squeeze()
            
            features_list.append(feature.cpu())
            
            # Convert angle to radians for easier math later
            rads = np.deg2rad(angle)
            angles_list.append(torch.tensor(rads, dtype=torch.float32))
            
        # Stack into a single tensor (Dataset Size, 512)
        X = torch.stack(features_list)
        
        # OPTIONAL: Normalize activity to lie on the unit sphere
        # This helps the VAE significantly but is technically a preprocessing choice.
        # The paper implies using normalized inputs often helps with vMF distributions.
        X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        
        y = torch.stack(angles_list)
        return X, y

    def _download_and_extract(self):
        url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
        zip_path = "coil-100.zip"
        
        if not os.path.exists(self.root_dir):
            if not os.path.exists(zip_path):
                print("Downloading COIL-100 dataset (130MB)...")
                urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Extraction Done.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.angles[idx]