import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import json
from collections import OrderedDict


class classification_model(nn.Module):
    def __init__(self, model_name, num_classes, use_pretrained):
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
                nn.Linear(512 * 12 * 15, 2048),
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


class attribute_model(nn.Module):
    def __init__(self, load_model_fname, model_name, num_classes_classification, use_pretrained, head_config):
        super(attribute_model, self).__init__()
        if model_name == "vgg":
            model_ft = classification_model(model_name, num_classes_classification, use_pretrained)
            # original saved file with DataParallel
            state_dict = torch.load(load_model_fname)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            # model.load_state_dict(new_state_dict)
            model_ft.load_state_dict(new_state_dict)
            for param in model_ft.parameters():
                param.requires_grad = False
            self.features = model_ft.features
            self.classifier = model_ft.classifier
            self.classifier[3] = nn.Linear(2048, 512)
            self.classifier[4] = nn.ReLU(True)
            self.classifier[5] = nn.Dropout()
            self.classifier[6] = nn.Linear(512, 256)

        # Define the heads attribute as a Module List as it will be easier to dynamically
        # assign the number of heads as nn.Linear modules depending on the configuration
        self.heads = nn.ModuleList()

        # Defining this will help to get the keys name in the forward function
        self.head_config = head_config

        self.Softmax = nn.Softmax(dim=1)  # Softmax to be used for evaluation

        # Append each head as a nn.Linear model
        for head_name, num_label in (head_config.items()):
            h = self._create_head(256, num_label)
            self.heads.append(h)
        # self.heads.to(self.device)

    def _create_head(self, in_features, num_of_labels):
        '''A function to creat the individual heads as a sequential model

        Args:
        in_features (int): The number of input features for the linear layer of the head
        num_of_labels (int): The number of labels for the given head, which will also be the number of output features for the linear layer


        Returns:
        A nn.Linear layer, which corresponds to a single head of the multi-head classifier

        '''
        return nn.Linear(in_features=in_features, out_features=num_of_labels)

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        output_dict = {}

        for i, (name, num_classes) in enumerate(
                self.head_config.items()):  # Enumerate through the head config to propagate the flattened tensor through the head classifiers
            output_dict[name] = self.heads[i](x)

        return output_dict


def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
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
