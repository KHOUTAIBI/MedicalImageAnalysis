import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as gnn

class InvariantCNN(nn.Module):
    def __init__(self, output_dim=None):
        super(InvariantCNN, self).__init__()
        
        # Geometric Space: Rotations of 22.5 degrees (C16)
        self.r2_act = gspaces.Rot2dOnR2(N=16)
        
        # Input & Hidden
        feat_type_in = gnn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        feat_type_hid = gnn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr]) 
        feat_type_out = gnn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        
        # Equivariant Layers
        self.input_layer = gnn.R2Conv(feat_type_in, feat_type_hid, kernel_size=7, padding=3)
        self.layer1 = gnn.SequentialModule(
            gnn.InnerBatchNorm(feat_type_hid),
            gnn.ELU(feat_type_hid, inplace=True),
            gnn.R2Conv(feat_type_hid, feat_type_out, kernel_size=5, padding=2),
            gnn.InnerBatchNorm(feat_type_out),
            gnn.ELU(feat_type_out, inplace=True)
        )
        
    def forward(self, x):
        x = gnn.GeometricTensor(x, self.input_layer.in_type)
        x = self.input_layer(x)
        x = self.layer1(x)
        
        # Extract raw tensor: (Batch, Channels*Orientations, H, W)
        raw_features = x.tensor
        
        # Global Spatial Pooling
        # Collapses spatial grid artifacts
        global_features = torch.mean(raw_features, dim=(2, 3)) 
        
        # Return Raw High-Dim Features
        # Shape: (Batch, 768).
        return global_features