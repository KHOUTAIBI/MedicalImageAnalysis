import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


class RotatingObjectDataset(Dataset):
    def __init__(self, embedding_dim=512, n_samples=1000):
        """
        Generates synthetic "CNN activations" for a rotating object.
        """
        self.n_samples = n_samples
        self.embedding_dim = embedding_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = models.resnet18(pretrained=True)
        self.model.eval()




        self.test_image = 