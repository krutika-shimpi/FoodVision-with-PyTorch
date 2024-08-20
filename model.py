import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes :int =3,
                          seed : int = 42):
  """ Create an return a pretraiend model weights and transforms """
  # 1, 2, 3 Create EffNetB2 pretrained weights, transforms and model

  # 1. Setup pretrained model weights
  model_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT

  # 2. Get effnetb2 transforms
  effnetb2_transforms = model_weights.transforms()

  # 3. Create model instance
  model = torchvision.models.efficientnet_b2(weights = model_weights)

  # 4. Freeze all the layers in the base model
  for param in model.parameters():
    param.requires_grad = False

  # 5. Change the classifier head
  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features = 1408,
                out_features = num_classes))

  return model, effnetb2_transforms
