import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo

model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'


class VGG16(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128,  3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),   
            nn.ReLU(inplace=True),          
            nn.Dropout(),                   
            nn.Linear(4096, 4096),          
            nn.ReLU(inplace=True),          
            nn.Dropout(),                   
            nn.Linear(4096, 2)              
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def vgg16(pretrained=False):
    '''
    MUST use this function to create the convolutional model
    :param pretrained: if true returns a model pre-trained on ImageNet
    '''

    model = VGG16()
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_url)
        model_dict = model.state_dict()

        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # print("Weights loaded:", len(pretrained_dict))
        # print(f"Pre-trained: {len(pretrained_dict)}/{len(model_dict)} tensors loaded")
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # load the new state dict
        model.load_state_dict(model_dict)
   

    return model
        