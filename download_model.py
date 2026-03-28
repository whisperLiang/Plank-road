import torch
import torchvision.models as models
import os

model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(wweights="DEFAULT")
torch.save(model.state_dict(),  "fasterrcnn_mobilenet_v3_large_fpn.pth")
