import torch
import torch.nn as nn
from utils.transforms3D import normalize_to_identity_cube, RandomFlipBatch
from torch import distributed as dist
from utils.ddp_handling import ddp_is_running
from typing import Tuple



class TransformerModel(nn.Module):
    def __init__(self, dim_class_hidden: int, dim_out: int, num_layers: int, embedding_layer,  **transformer_args):
        super(TransformerModel, self).__init__()
        self.d_model = transformer_args['d_model']
        self.dim_class_hidden = dim_class_hidden
        self.dim_out = dim_out
        self.embedding_layer = embedding_layer
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(**transformer_args, batch_first=True)
        self.embeddingTransformer = nn.TransformerEncoder(encoder_layer = self.transformerEncoderLayer, 
                                                          num_layers = num_layers,
                                                          enable_nested_tensor = False)
        self.classifier = nn.Sequential(
            nn.Dropout(transformer_args["dropout"]),
            nn.Linear(self.d_model, self.dim_class_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_class_hidden, self.dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input-shape: [bs, context_size, 15, 3]
        x = normalize_to_identity_cube(x)
        x = self.embedding_layer(x) # [bs, context_size, d_model] ≈ [bs, 2000, 128]
        x = self.embeddingTransformer(x) # Shape [bs, context_size, d_model]
        x = self.classifier(x) # Shape [bs, context_size, dim_out]
        return x



def get_embedding_layer(num_support_points_per_streamline: int, d_model: int = 128) -> nn.Module:
    assert d_model >= 45, "Embedding type flip_aug only works with d_model>=45"
    return EmbeddingAugFlipBatchGPU(flipProbability=0.5, 
                                    num_support_points_per_streamline=num_support_points_per_streamline, 
                                    d_model=d_model)



class EmbeddingAugFlipBatchGPU(nn.Module):
    """
    Embed the input tensor with a flip augmentation based embedding. It fills the rest of the embedding with a linear layer.
    This class is designed to be used with the GPU with batch processing.
    """
    def __init__(self, flipProbability: float = 0.5, num_support_points_per_streamline: int = 15, d_model: int = 66):
        super(EmbeddingAugFlipBatchGPU, self).__init__()
        self.flipAugEmbedding = RandomFlipBatch(flipProbability=flipProbability)
        if d_model > 3 * num_support_points_per_streamline:
            self.adaptive = True
            self.linear = nn.Linear(in_features = num_support_points_per_streamline * 3, 
                                    out_features = d_model - num_support_points_per_streamline * 3)
            self.activation = nn.LeakyReLU()
        elif d_model == 3 * num_support_points_per_streamline:
            # Module reduces to flip augmentation and reshaping
            self.adaptive = False
        else:
            raise ValueError(f"d_model = {d_model} must be greater or equal" \
                             +"to 3 * numStreamlines = {3*num_support_points_per_streamline}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): Shape [bs, numStreamlines, numPoints, 3], on device

        Returns:
            torch.Tensor: Embedded Streamlines with shape [bs, numStreamlines, d_model]
        """
        bs, numStreamlines, _, _ = x.size()
        if self.training: # Only apply flip augmentation during training (model.train())
            x = self.flipAugEmbedding(x) # Shape [bs, numStreamlines, numPoints, 3]
        x = x.reshape(bs, numStreamlines, -1) # Shape [bs, numStreamlines, numPoints * 3]
        if self.adaptive:
            rest_embedding = self.activation(self.linear(x)) # Shape [bs, numStreamlines, d_model - numPoints * 3]
            x = torch.cat([x, rest_embedding], dim=2)
        return x # Shape [bs, numStreamlines, d_model]
    
    def __repr__(self):
        if self.adaptive:
            return f"{self.__class__.__name__}" + f"flipAugEmbedding={self.flipAugEmbedding}, linear={self.linear}, activation={self.activation}"
        else:
            return f"{self.__class__.__name__}" + f"flipAugEmbedding={self.flipAugEmbedding}"