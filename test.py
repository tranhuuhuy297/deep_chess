import numpy as np
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

a = torch.tensor([[1, 2, 3.], [4., 5, 6.]]).to(device)

np.save('huy.npy',np.array(a.cpu()))

data = np.load('huy.npy')

print(data)