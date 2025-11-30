import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import warnings
import math
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

from invariant_cnn import InvariantCNN

warnings.filterwarnings("ignore", category=UserWarning)

class InvariantManifoldGenerator:
    def __init__(self, pca_components=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading InvariantCNN (No Projection)...")
        # output_dim argument is now ignored by the class, but we pass it for compatibility
        self.model = InvariantCNN(output_dim=None).to(self.device)
        self.model.eval()
        
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components)

    def _load_image(self, local_path="tiger.jpg"):
        if os.path.exists(local_path):
            return Image.open(local_path).convert('RGB')
        
        else:
            # use noise image if download fails
            print(f"Warning: Could not find image at {local_path}. Using noise image instead.")            
            noise_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            return Image.fromarray(noise_img).convert('RGB')

    def _prepare_natural_canvas(self, img, input_size=224):
        safe_dim = int(math.ceil(input_size * math.sqrt(2))) 
        safe_dim = max(safe_dim, 320)
        
        w, h = img.size
        scale = safe_dim / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
        
        left = (new_w - safe_dim) // 2
        top = (new_h - safe_dim) // 2
        img_safe = img_resized.crop((left, top, left + safe_dim, top + safe_dim))
        return img_safe, safe_dim

    def generate_manifold_data(self, image_path="tiger.jpg", num_angles=360):
        img = self._load_image(image_path)
        img_safe, _ = self._prepare_natural_canvas(img)
        print(f"Canvas prepared (Natural): {img_safe.size}")
        
        normalize_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        activations = []
        angles = np.linspace(0, 360, num_angles, endpoint=False)
        
        print(f"Extracting features from {num_angles} rotations...")
        for angle in angles:
            rotated = img_safe.rotate(angle, resample=Image.BICUBIC)
            
            target_size = 224
            w, h = rotated.size
            left = (w - target_size) // 2
            top = (h - target_size) // 2
            cropped = rotated.crop((left, top, left + target_size, top + target_size))
            
            input_tensor = normalize_transform(cropped).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model(input_tensor).cpu().numpy().flatten()
            
            activations.append(feat)
            
        activations = np.array(activations)
        print(f"Raw Features: {activations.shape}")
        
        # PCA
        print(f"Reducing dimensions to {self.pca_components}...")
        reduced_activations = self.pca.fit_transform(activations)
        
        epsilon = 1e-8
        means = np.mean(reduced_activations, axis=0)
        stds = np.std(reduced_activations, axis=0) + epsilon
        reduced_activations = (reduced_activations - means) / stds
        
        return torch.FloatTensor(reduced_activations), torch.FloatTensor(angles)