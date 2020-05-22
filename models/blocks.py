import torch
import torch.nn as nn

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    def __init__(self, init_dict):
        """
        Convolution Block definition and forward pass based on the json structure
        At each convolution block, Conv2d, MaxPool2d, BatchNorm/InstanceNorm and ReLu can be initialized.

        Args:
        init_dict: Dictionary input to initialize elements of the convolution block

        Sample init_dict,
        {
            "in_channels": 1,
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 2,
            "activation": "ReLU",
            "normalization": {},
            "pooling": {
                "type": "MaxPool2d",
                "kernel_size": 2,
                "stride": 2
            }
        }

        This dictionary specifies all the neccessary parameters as keys
        """
        super(ConvBlock, self).__init__()
        self.init_dict = init_dict


        # Intialize Conv2d based on in_channels, out_channels, padding, stride and kernel size specified as keys
        self.conv = nn.Conv2d(self.init_dict["in_channels"], self.init_dict["out_channels"], \
                        padding=self.init_dict["padding"], stride=self.init_dict["stride"], kernel_size=self.init_dict["kernel_size"])

        # If "normalization" key is not empty, then initialize normalization based on the specified string
        if self.init_dict["normalization"]:

            # The string is evaluated based on the layer being present in torch.nn module. For example, if BatchNorm2d is provied, torch.nn.BatchNorm2d is applied
            normalization = eval("torch.nn.{}".format(self.init_dict["normalization"]))

            # The normalization is now defined
            self.normalization = normalization( self.init_dict["out_channels"])

        # Activation is defined by evaluating the provided "activation" string in the torch.nn module. For example, if ReLU is provided, torch.nn.ReLU is applied
        self.activation = eval('torch.nn.{}'.format(self.init_dict["activation"]))(inplace=True)
        
        # If "pooling" is not empty, initialize pooling similar to "normalization"
        if self.init_dict["pooling"]:
            pooling_dict = self.init_dict["pooling"]
            pool = eval("torch.nn.{}".format(pooling_dict["type"]))
            self.pooling = pool(kernel_size=pooling_dict["kernel_size"], stride=pooling_dict["stride"])

    
        logger.info("\n \n Added Conv2D block with \n {}".format(self.init_dict))

    def forward(self, x):

        # Forward pass over the conv
        y = self.conv(x)

        # If normalization is specified, include normalization in fw pass
        if self.init_dict["normalization"]:
            y = self.normalization(y)
        
        # Add activation in the forward pass
        y = self.activation(y)
        
        # If pooling is specified, include pooling in fw pass
        if self.init_dict["pooling"]:
            y = self.pooling(y)
    
        return y

        
class NetworkHead(nn.Module):
    def __init__(self, init_dict):
        """
        Network Head definition and forward pass based on the json structure
        At each network head block, 2 fully connected layers and dropout are defined.

        Args:
        init_dict: Dictionary input to initialize elements of the Network head

        Sample init_dict,
        {
        "fc_input": 1600,
        "fc1": 64,
        "final_layer": 7,
        "dropout": true
        }

        This dictionary specifies all the neccessary parameters as keys
        """
        super(NetworkHead, self).__init__()

        self.init_dict = init_dict

        # If dropout is provided as true, enable 50% probability dropout
        if self.init_dict["dropout"]:
            self.dropout1 = nn.Dropout()

        # Two Linear layers are initialized based on input and output neurons specified in the dictionary. 
        self.fc1 = nn.Linear(self.init_dict["fc_input"], self.init_dict["fc1"])
        self.relu = nn.ReLU(inplace=True)
        self.final_layer = nn.Linear(self.init_dict["fc1"], self.init_dict["final_layer"])
        logger.info("\n \n Added fully connected network head with \n {}".format(self.init_dict))


    def forward(self, x):
        """
        Forward pass definition for the network head block
        """

        # Flatten torch tensor starting from first dimension in the tensor
        x = torch.flatten(x, 1)

        # Include dropout in forward pass if enabled
        if self.init_dict["dropout"]:
            y = self.dropout1(x)

        # Forward pass for linear layers and relu activation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.final_layer(y)

        return y
