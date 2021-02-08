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

def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, num_classes),
        # )
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft