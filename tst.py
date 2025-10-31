import torch

torch.backends.cudnn.enabled = False

x = torch.randn(1, 3, 512, 512).cuda()
conv = torch.nn.Conv2d(3, 64, 3).cuda()
y = conv(x)
print(y.shape)