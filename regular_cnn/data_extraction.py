import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image, ImageDraw
import numpy as np
from sklearn.decomposition import PCA
import warnings
import math
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

warnings.filterwarnings("ignore", category=UserWarning)

class NeuralManifoldGenerator:
    def __init__(self, model_name='resnet18', layer_name='avgpool', pca_components=20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name}...")
        weights = models.ResNet18_Weights.DEFAULT
        self.model = getattr(models, model_name)(weights=weights).to(self.device)
        self.model.eval()
        
        self.layer_name = layer_name
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components)
        
        self.features = {}
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        getattr(self.model, layer_name).register_forward_hook(get_activation(layer_name))

    def _load_image(self, local_path="tiger.jpg"):
        if os.path.exists(local_path):
            try:
                print(f"Loading local image: {local_path}")
                return Image.open(local_path).convert('RGB')
            except Exception as e:
                print(f"Error loading local file: {e}")
        
        else:
            # use noise
            print("Local image not found. Generating random noise image.")
            width, height = 512, 512
            noise_img = Image.fromarray(np.uint8(np.random.rand(height, width, 3) * 255))
            return noise_img

    def _prepare_natural_canvas(self, img, input_size=224):
        """
        1. Resize the whole image so the shortest side is ~320px.
        2. This serves as our rotation buffer.
        """
        # 1. Calculate safe dimension (~320px)
        safe_dim = int(math.ceil(input_size * math.sqrt(2))) 
        safe_dim = max(safe_dim, 320)
        
        # 2. Resize entire image down
        w, h = img.size
        scale = safe_dim / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # 3. Center Crop the Safe Canvas
        left = (new_w - safe_dim) // 2
        top = (new_h - safe_dim) // 2
        img_safe = img_resized.crop((left, top, left + safe_dim, top + safe_dim))
        
        return img_safe, safe_dim

    def visualize_pipeline(self, image_path="tiger.jpg", save_path="preprocessing_check_tiger.png"):
        print("Generating preprocessing visualization...")
        raw_img = self._load_image(image_path)
        
        safe_img, safe_dim = self._prepare_natural_canvas(raw_img)
        test_angles = [0, 45, 90]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        # A. Original
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Original Image\n{raw_img.size}")
        axes[0].axis('off')
        
        # B-D. Rotations
        target_size = 224
        for idx, angle in enumerate(test_angles, start=1):
            # Rotate the 320px canvas
            rotated = safe_img.rotate(angle, resample=Image.BICUBIC)
            
            # Crop the final 224px input
            w, h = rotated.size
            left = (w - target_size) // 2
            top = (h - target_size) // 2
            final_input = rotated.crop((left, top, left + target_size, top + target_size))
            
            axes[idx].imshow(final_input)
            axes[idx].set_title(f"Rotation: {angle}°\n(Input: 224x224)")
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.show()

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
            # Rotate
            rotated = img_safe.rotate(angle, resample=Image.BICUBIC)
            
            # Crop 224
            target_size = 224
            w, h = rotated.size
            left = (w - target_size) // 2
            top = (h - target_size) // 2
            cropped = rotated.crop((left, top, left + target_size, top + target_size))
            
            # Normalize
            input_tensor = normalize_transform(cropped).unsqueeze(0).to(self.device)
            
            # Extract
            with torch.no_grad():
                self.model(input_tensor)
            feat = self.features[self.layer_name].cpu().numpy().flatten()
            activations.append(feat)
            
        activations = np.array(activations)
        
        # PCA
        print(f"Reducing dimensions to {self.pca_components}...")
        reduced_activations = self.pca.fit_transform(activations)
        
        # Standardize
        epsilon = 1e-8
        means = np.mean(reduced_activations, axis=0)
        stds = np.std(reduced_activations, axis=0) + epsilon
        reduced_activations = (reduced_activations - means) / stds
        
        return torch.FloatTensor(reduced_activations), torch.FloatTensor(angles)

if __name__ == "__main__":
    gen = NeuralManifoldGenerator()
    gen.visualize_pipeline()