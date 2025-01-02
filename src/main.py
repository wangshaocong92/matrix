import torch


torch.cuda.device("cuda")
print(torch.cuda.device_count())