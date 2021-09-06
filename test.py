import torch
from models import PSyncNet_color, PWav2Lip
import torch.nn as nn

device = "cuda:1"

psyncnet = PSyncNet_color().to(device)
pwav2lip = PWav2Lip().to(device)

audio = torch.randn(8,5,1,80,16).to(device)
video = torch.randn(8,6,5,64,64).to(device)

# a, v = pwav2lip(audio, video, 0)

output = pwav2lip(audio, video, 4)
# print(a.shape, v.shape)
print(output.shape)