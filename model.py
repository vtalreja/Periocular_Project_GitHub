import torch
import torch.nn as nn
from torchvision import datasets, models, transforms



class classification_model(nn.Module):
    def __init__(self, model_name,num_classes,use_pretrained):
        super(classification_model, self).__init__()

        if model_name == "vgg":
            """ VGG16_bn
            """
            model_ft = models.vgg16_bn(pretrained=use_pretrained)
            for param in model_ft.parameters():
                param.requires_grad = False
            self.features = model_ft.features
            # self.avgpool = nn.AdaptiveAvgPool2d((12, 12))
            self.classifier = nn.Sequential(
                nn.Linear(512*12*15, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024, num_classes),
            )


    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

