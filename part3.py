import torch
import matplotlib.pyplot as plt

def sierpinski(size):
    if size == 1: # Base case
        return torch.Tensor([[1]]) # If size is one return 1x1 tensor
    else:
        size //= 2 # Halved Size (Width and Height)
        top = torch.cat((sierpinski(size), sierpinski(size)), dim=1) # Add 2 to top
        bottom = torch.cat((sierpinski(size), torch.zeros((size, size))), dim=1) # Add 1 to bottom
        return torch.cat((top, bottom), dim=0) # Concatenate both top and bottom
    
# Generate and plot the Sierpinski triangle
sierpinski_triangle = sierpinski(128)
plt.imshow(sierpinski_triangle)
plt.show()
