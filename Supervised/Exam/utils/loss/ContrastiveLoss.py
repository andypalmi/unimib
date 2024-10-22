import torch.nn.functional as F
import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
    """
    ContrastiveLoss is a custom loss function for contrastive learning.
    Args:
        temperature (float, optional): A scaling factor for the similarity scores. Default is 0.5.
    Methods:
        forward(z1, z2):
            Computes the contrastive loss between two sets of embeddings.
            Args:
                z1 (torch.Tensor): Embeddings from the first branch of the model.
                z2 (torch.Tensor): Embeddings from the second branch of the model.
            Returns:
                torch.Tensor: The computed contrastive loss.
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Normalize the embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate the embeddings from both branches
        embeddings = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.T) / self.temperature
        
        # Create labels: Positive pairs (diagonal) are 1, all others 0
        labels = torch.cat([torch.arange(z1.size(0)) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z1.device)
        
        # Mask to exclude similarity with itself
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
        labels = labels.masked_fill(mask, 0)
        
        # Calculate contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
