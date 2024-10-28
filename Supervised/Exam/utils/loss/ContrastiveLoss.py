import torch.nn.functional as F
import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
    """
    Improved contrastive loss with better numerical stability and efficiency for contrastive learning.

    Example Workflow:
    - Given a batch size of 4, the similarity matrix will be of shape [8, 8] (2 * batch_size),
      containing cosine similarity values between each pair of concatenated embeddings.
      
    - The resulting similarity matrix, after applying the mask, looks like this:
    
        similarity = 
        [   [-inf, -0.3, 0.1, -0.4, 1.2, 0.2, -0.1, 0.4],
            [-0.3, -inf, 0.3, 0.5, 0.7, 1.5, -0.2, -0.3],
            [0.1, 0.3, -inf, 1.4, -0.2, 0.3, 0.8, -0.5],
            [-0.4, 0.5, 1.4, -inf, -0.6, 0.2, 0.1, 0.9],
            [1.2, 0.7, -0.2, -0.6, -inf, 0.4, 0.5, 0.3],
            [0.2, 1.5, 0.3, 0.2, 0.4, -inf, -0.3, -0.5],
            [-0.1, -0.2, 0.8, 0.1, 0.5, -0.3, -inf, 0.6],
            [0.4, -0.3, -0.5, 0.9, 0.3, -0.5, 0.6, -inf]   ]

    - Each row represents the cosine similarity between a given embedding and all other embeddings.
      Diagonal entries are masked to prevent self-similarity from influencing the loss calculation.

    - The `labels` tensor defines which index in each row is the positive pair:
        labels = [4, 5, 6, 7, 0, 1, 2, 3]  
      indicating that, for example, row 0 has its positive pair at index 4, row 1 at index 5, etc.
    """
    
    def __init__(self, temperature=0.07, batch_size=None):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        
    def forward(self, z1, z2):
        """
        Args:
            z1, z2: Tensors of shape [batch_size, feature_dim] representing two sets of embeddings.
        
        Returns:
            A single scalar loss value for the batch.
        """
        # Define batch size based on z1 if not provided
        batch_size = z1.shape[0] if self.batch_size is None else self.batch_size
        
        # Normalize embeddings to have unit norm (important for cosine similarity)
        z1 = F.normalize(z1, dim=1)  # Shape: [batch_size, feature_dim]
        z2 = F.normalize(z2, dim=1)  # Shape: [batch_size, feature_dim]
        
        # Concatenate normalized embeddings to create a unified batch of size 2 * batch_size
        embeddings = torch.cat([z1, z2], dim=0)  # Shape: [2 * batch_size, feature_dim]
        
        # Compute similarity matrix with cosine similarity between all embeddings
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        # similarity.shape -> [2 * batch_size, 2 * batch_size]
        
        # Mask self-similarity values (diagonal) by setting them to -inf
        # `torch.eye()` produces an identity matrix (1s on the diagonal), and `~` flips it
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=similarity.device)
        similarity = similarity.masked_fill(~mask, -torch.inf)  # Ignore self-similarity
        
        # Create labels for positive pairs
        # Labels indicate the positions of positive pairs in similarity matrix
        labels = torch.arange(batch_size, device=similarity.device)
        labels = torch.cat([labels + batch_size, labels])  # Shape: [2 * batch_size]
        
        '''
        Explanation of `labels`:
            With batch size 4, `labels` is created as follows:
            - First half: indices [0, 1, 2, 3] are shifted by batch size to [4, 5, 6, 7]
            - Second half: same indices [0, 1, 2, 3] are used directly
            Result:
                labels = [4, 5, 6, 7, 0, 1, 2, 3]
                
            This means that, for instance:
            - Row 0 in the similarity matrix has its positive pair at index 4
            - Row 1 has its positive pair at index 5
            - Row 2 has its positive pair at index 6
            - Row 3 has its positive pair at index 7
        '''
        
        # Compute contrastive loss using cross-entropy on similarity and labels
        # Cross-entropy will maximize similarity for positive pairs and minimize for others
        loss = F.cross_entropy(similarity, labels)
        
        return loss
