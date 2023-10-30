import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# Model to optimize 
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 

# Sample data
x = torch.rand(5, 10)
y = torch.rand(5, 10) 

# Forward pass  
y_pred = model(x)

# DiffKendall loss
loss_fn = torch.nn.MSELoss()
loss = loss_fn(diffkendall(y, y_pred), torch.ones(5)) 

# Backprop
loss.backward()

print(loss.item())

# Optimize
optimizer.step()