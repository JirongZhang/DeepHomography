from torch import nn
import resnet
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def build_model(model_name, pretrained=False):
    if model_name == 'resnet34':
        model = resnet.resnet34(pretrained=False)
    elif model_name == 'resnet50':
        model = resnet.resnet50(pretrained=False)
    elif model_name == 'resnet101':
        model = resnet.resnet101(pretrained=False)
    elif model_name == 'resnet152':
        model = resnet.resnet152(pretrained=False)

    if model_name == 'resnet18':
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.last_linear = nn.Linear(512, 8)
    elif model_name == 'resnet34':
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 8)  # Nx8
    else:
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 8)  # Nx8

    if pretrained == True:
        exclude_dict = ['conv1.weight','fc.weight','fc.bias']
        pretrained_dict = model_zoo.load_url(model_urls[model_name])
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in exclude_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


