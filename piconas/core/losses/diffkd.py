import torch 

def diffkendall(x, y, alpha=0.5):
    """
    Differentiable approximation of Kendall's rank correlation.
    
    Args:
        x: Tensor of shape (N, D) 
        y: Tensor of shape (N, D)
        alpha: Hyperparameter for sigmoid approximation
    
    Returns:
        diffkendall: Tensor of shape (N,) 
    """
    
    N, D = x.shape
    
    # Pairwise difference 
    x_diff = x[:, None, :] - x[:, :, None]  # (N, D, D) 
    y_diff = y[:, None, :] - y[:, :, None] 
    
    # Numerator  
    num = (torch.sigmoid(alpha*x_diff) - torch.sigmoid(-alpha*x_diff)) * \
          (torch.sigmoid(alpha*y_diff) - torch.sigmoid(-alpha*y_diff)) 
    
    # Denominator
    den = (torch.sigmoid(alpha*x_diff) + torch.sigmoid(-alpha*x_diff)) * \
          (torch.sigmoid(alpha*y_diff) + torch.sigmoid(-alpha*y_diff))
        
    # DiffKendall       
    diffkendall = (num.sum(-1).sum(-1) - num.diagonal(dim1=-2, dim2=-1).sum(-1)) / \
                  ((D-1) * D / 2)  
        
    return diffkendall