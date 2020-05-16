import torch
import torch.nn as nn

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    def __init__(self, init_dict):
        super(ConvBlock, self).__init__()
        self.init_dict = init_dict

        self.conv = nn.Conv2d(self.init_dict["in_channels"], self.init_dict["out_channels"], \
                        padding=self.init_dict["padding"], stride=self.init_dict["stride"], kernel_size=self.init_dict["kernel_size"])


        if self.init_dict["normalization"]:
            normalization = eval("torch.nn.{}".format(self.init_dict["normalization"]))
            self.normalization = normalization( self.init_dict["out_channels"])

        self.activation = eval('torch.nn.{}'.format(self.init_dict["activation"]))(inplace=True)
        
        if self.init_dict["pooling"]:
            pooling_dict = self.init_dict["pooling"]
            pool = eval("torch.nn.{}".format(pooling_dict["type"]))
            self.pooling = pool(kernel_size=pooling_dict["kernel_size"], stride=pooling_dict["stride"])

      

        logger.info("\n \n Added Conv2D block with \n {}".format(self.init_dict))

    def forward(self, x):
        y = self.conv(x)

        if self.init_dict["normalization"]:
            y = self.normalization(y)
        
        y = self.activation(y)
        
        if self.init_dict["pooling"]:
            y = self.pooling(y)
    
        return y

        
class NetworkHead(nn.Module):
    def __init__(self, init_dict):
        super(NetworkHead, self).__init__()

        self.init_dict = init_dict


        if self.init_dict["dropout"]:
            self.dropout1 = nn.Dropout()

        self.fc1 = nn.Linear(self.init_dict["fc_input"], self.init_dict["fc1"])
        self.relu = nn.ReLU(inplace=True)

        if self.init_dict["dropout"]:
            self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(self.init_dict["fc1"], self.init_dict["fc2"])
        self.relu = nn.ReLU(inplace=True)
        self.final_layer = nn.Linear(self.init_dict["fc2"], self.init_dict["final_layer"])

        logger.info("\n \n Added fully connected network head with \n {}".format(self.init_dict))


    def forward(self, x):
        x = torch.flatten(x, 1)

        if self.init_dict["dropout"]:
            y = self.dropout1(x)

        y = self.fc1(y)
        y = self.relu(y)
        if self.init_dict["dropout"]:
            y = self.dropout2(y)

        y = self.fc2(y)
        y = self.relu(y)

        y = self.final_layer(y)

        return y
