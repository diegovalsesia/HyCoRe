import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from hyptorch.nn import FromPoincare, ToPoincare, HypLinear
import geoopt
import models.nn as hypnn

c = 1.0
ball = geoopt.PoincareBall(c)

class ToPoincare(torch.nn.Module):
    def __init__(self, dim, ball):
        super().__init__()
        self.x = torch.zeros(dim)
        self.ball_ = ball
        
    def forward(self, u):
        device = u.device
        mapped = self.ball_.expmap0(u)
        return mapped
        
        
class HResNet(nn.Module):
    def __init__(self, base_model, out_dim, pretrained=False, freeze_base=False):
        super(HResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                            "resnet50": models.resnet50(pretrained=pretrained),
                            "resnet101": models.resnet101(pretrained=pretrained)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.to_poincare = hypnn.ToPoincare(c=c, train_x=0, train_c=0, ball_dim=256)
        self.embedding = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            self.to_poincare,
        )
        self._init_embedding_weights()    

    def _init_embedding_weights(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(
                    m.weight, -1e-4, 1e-4)
                nn.init.constant_(m.bias, 0.)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        try:
            h = self.encoder(x)
        except:
            print(x.shape)
            h = self.encoder(x)
        h = h.squeeze()
        x = self.embedding(h)
        return h, x
    

class HResNet2(nn.Module):
    def __init__(self, base_model, out_dim, pretrained=False, freeze_base=False):
        super(HResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                            "resnet50": models.resnet50(pretrained=pretrained),
                            "resnet101": models.resnet101(pretrained=pretrained)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.to_poincare = hypnn.ToPoincare(c=c, train_x=0, train_c=0, ball_dim=256)
        self.embedding = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            self.to_poincare,
            hypnn.HypLinear(256, 2, c=c)
        )
        self._init_embedding_weights()    

    def _init_embedding_weights(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(
                    m.weight, -1e-4, 1e-4)
                nn.init.constant_(m.bias, 0.)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        try:
            h = self.encoder(x)
        except:
            print(x.shape)
            h = self.encoder(x)
        h = h.squeeze()
        x = self.embedding(h)
        return h, x
