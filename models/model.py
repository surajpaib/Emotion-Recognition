import torch
import torch.nn as nn

import json
import logging

from .blocks import ConvBlock, NetworkHead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, model_json, initialize_weights=True):
        """
        Initialize the model using different model blocks defined in blocks.py


        Args:
        model_json: The path to the json file where the model definition in blocks is defined.
        initialize_weights: Set to true if weights need to be initialized.
        """
        super(Model, self).__init__()


        # Create a module list to store all the blocks as a list. Easier forward pass definition with this.
        self.network = nn.ModuleList()

        # Load model configuration using the json
        with open(model_json, "r") as fp:
            self.model_config = json.load(fp)


        logger.info("\n \n Creating Model from {} \n".format(model_json))

        # Iterate over the items in the "ConvBlocks" key in the json 
        for conv in self.model_config["ConvBlocks"]:

            # For each item intialize a ConvBlock module from blocks.py 
            self.network.append(ConvBlock(conv))
  
        # After all the conv blocks add an adaptive average pooling layer
        self.network.append(nn.AdaptiveAvgPool2d((5, 5)))


        # Intialize the NetworkHead as defined in under the "NetworkHead" key in the json
        self.network.append(NetworkHead(self.model_config["NetworkHead"]))

        # Display Network structure
        logger.info(self.network)

        # Initialize weights if flag set
        if initialize_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        Model forward definition
        """

        # Super simple forward pass definition going over each element in the module list. 
        # The output at each layer is passed as input to the next. 
        for layer in self.network:
            x = layer(x)

        return x


    def _initialize_weights(self):
        """
        Initialized similar to VGG Implementations from torchvision library
        """
        for m in self.modules():

            # If CNN layer, kaiming normal initialization
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            # If batch norm, set alpha to zero and beta to 1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # If linear layer, set bias to 0 and initialize weights from a normal distribution with mean 0 and variance 0.01
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":

    # Test Model initialization and forward pass
    model = Model("models/Baseline.json")
    x = torch.randn((6, 1, 48, 48))
    y = model(x)
    print(y.shape)