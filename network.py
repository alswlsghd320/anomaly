import timm
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        print('Weight Init')

class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.encoder_name, pretrained=True,
                                    drop_path_rate=args.drop_path_rate, num_classes=args.num_classes)
    def forward(self, x):
        x = self.encoder(x)
        return x
    
