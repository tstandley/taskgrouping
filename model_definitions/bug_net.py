import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

from .ozan_rep_fun import ozan_rep_function,trevor_rep_function,OzanRepFunction,TrevorRepFunction

__all__ = ['bugnet_taskonomy']

hidden_dim = 3

class BugEncoder(nn.Module):
    # Input images are of size 256x256 and output should be 512
    def __init__(self):
        super(BugEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=hidden_dim,
                               kernel_size= 3,
                               stride = 1,
                               padding = 1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        return x


class BugDecoder(nn.Module):
    def __init__(self, out_channels):
        super(BugDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=hidden_dim,
                               out_channels=out_channels,
                               kernel_size= 3,
                               stride = 1,
                               padding = 1,
                               bias=True)

    def forward(self, x):
        x = self.conv1(x)
        return x


class BugNet(nn.Module):
    def __init__(self, tasks=None):
        super(BugNet, self).__init__()

        self.encoder = BugEncoder()
        
        self.tasks=tasks
        self.task_to_decoder = {}

        if tasks is not None:
            for task in tasks:
                if task == 'segment_semantic':
                    output_channels = 18
                if task == 'depth_zbuffer':
                    output_channels = 1
                if task == 'normal':
                    output_channels = 3
                if task == 'edge_occlusion':
                    output_channels = 1
                if task == 'reshading':
                    output_channels = 3
                if task == 'keypoints2d':
                    output_channels = 1
                if task == 'edge_texture':
                    output_channels = 1

                decoder = BugDecoder(output_channels)
                self.task_to_decoder[task]=decoder
        
        self.decoders = nn.ModuleList(self.task_to_decoder.values())

    def forward(self, input):
        rep = self.encoder(input)
        outputs={'rep':rep}

        # rep = ozan_rep_function(rep)
        rep = trevor_rep_function(rep)

        for i, (task,decoder) in enumerate(zip(self.task_to_decoder.keys(), self.decoders)):
            outputs[task]=decoder(rep)
        
        return outputs


def bugnet_taskonomy(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _tinynet('tinynet', Bottleneck, [3, 8, 36, 3], pretrained,
    #                **kwargs)
    return BugNet(**kwargs)
