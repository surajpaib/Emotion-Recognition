import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()


        self.conv1 = ConvBlock({"in_channels": 1, 
                                "out_channels": 64,
                                "kernel_size": 3, 
                                "stride": 4, 
                                "padding": 2, 
                                "activation": "ReLU",
                                "normalization": None, 
                                "pooling": {
                                    "type": "MaxPool2d",
                                    "kernel_size": 3,
                                    "stride": 2
                                }})

        self.conv2 = ConvBlock({"in_channels": 64, 
                                        "out_channels": 128,
                                        "kernel_size": 3, 
                                        "stride": 1, 
                                        "padding": 2, 
                                        "activation": "ReLU",
                                        "normalization": None, 
                                        "pooling": {
                                            "type": "MaxPool2d",
                                            "kernel_size": 3,
                                            "stride": 2
                                        }})                                


        self.conv3 = ConvBlock({"in_channels": 128, 
                                        "out_channels": 256,
                                        "kernel_size": 3, 
                                        "stride": 1, 
                                        "padding": 1, 
                                        "activation": "ReLU",
                                        "normalization": None, 
                                        "pooling": None
                                        })      



        self.conv4 = ConvBlock({"in_channels": 256, 
                                        "out_channels": 256,
                                        "kernel_size": 3, 
                                        "stride": 1, 
                                        "padding": 1, 
                                        "activation": "ReLU",
                                        "normalization": None, 
                                        "pooling": {
                                            "type": "MaxPool2d",
                                            "kernel_size": 3,
                                            "stride": 2
                                        }})    


        self.network_head = NetworkHead({
            "fc_input": 256*1*1,
            "fc1": 4096,
            "fc2": 4096,
            "final_layer": 7,
            "dropout": True
        })                                         


    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)

        y = self.network_head(y)
        return y

class ConvBlock(nn.Module):
    def __init__(self, init_dict):
        super(ConvBlock, self).__init__()
        self.init_dict = init_dict

        self.conv = nn.Conv2d(self.init_dict["in_channels"], self.init_dict["out_channels"], \
                        padding=self.init_dict["padding"], stride=self.init_dict["stride"], kernel_size=self.init_dict["kernel_size"])

        print(eval('torch.nn.{}'.format(self.init_dict["activation"])))
        self.activation = eval('torch.nn.{}'.format(self.init_dict["activation"]))(inplace=True)
        
        if self.init_dict["pooling"]:
            pooling_dict = self.init_dict["pooling"]
            pool = eval("torch.nn.{}".format(pooling_dict["type"]))
            self.pooling = pool(kernel_size=pooling_dict["kernel_size"], stride=pooling_dict["stride"])

        if self.init_dict["normalization"]:
            normalization = eval("torch.nn.{}".format(self.init_dict["normalization"]))
            self.normalization = normalization( self.init_dict["out_channels"])



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

        self.fc1 = nn.Linear(self.init_dict["fc_input"], self.init_dict["fc2"])
        self.relu = nn.ReLU(inplace=True)

        if self.init_dict["dropout"]:
            self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(self.init_dict["fc1"], self.init_dict["fc2"])
        self.relu = nn.ReLU(inplace=True)
        self.final_layer = nn.Linear(self.init_dict["fc2"], self.init_dict["final_layer"])

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

if __name__ == "__main__":
    model = BaselineModel()

    x = torch.randn((6, 1, 48, 48))

    y = model.forward(x)

    print(y.shape)