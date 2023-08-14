import torch
import matplotlib.pyplot as plt

def sierpinski(size):
    if size == 1: # Base case
        return torch.Tensor([[1]]) # If size is one return 1x1 tensor
    else:
        size //= 2
        top = torch.cat((sierpinski(size), sierpinski(size)), dim=1)
        bottom = torch.cat((sierpinski(size), torch.zeros((size, size))), dim=1)
        return torch.cat((top, bottom), dim=0)
    
    # Generate and plot the Sierpinski triangle
sierpinski_triangle = sierpinski(128)
plt.imshow(sierpinski_triangle)
plt.show()
