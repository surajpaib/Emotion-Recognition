import torch
import torch.nn as nn

import json
import logging

from .blocks import ConvBlock, NetworkHead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, model_json, initialize_weights=True):
        super(Model, self).__init__()

        self.network = nn.ModuleList()

        with open(model_json, "r") as fp:
            self.model_config = json.load(fp)


        logger.info("\n \n Creating Model from {} \n".format(model_json))



        for conv in self.model_config["ConvBlocks"]:
            self.network.append(ConvBlock(conv))
  

        self.network.append(nn.AdaptiveAvgPool2d((5, 5)))

        self.network.append(NetworkHead(self.model_config["NetworkHead"]))

       
        logger.info(self.network)


        if initialize_weights:
            self._initialize_weights()

    def forward(self, x):
        for layer in self.network:
            x = layer(x)

        return x


    def _initialize_weights(self):
        """
        Initialized similar to VGG Implementations from torchvision library
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = Model("models/Baseline.json")

    x = torch.randn((6, 1, 48, 48))

    y = model(x)

    print(y.shape)