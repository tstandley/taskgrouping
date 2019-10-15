""" 
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
from .ozan_rep_fun import ozan_rep_function,trevor_rep_function,OzanRepFunction,TrevorRepFunction

__all__ = ['xception_taskonomy_joined_decoder','xception_taskonomy_joined_decoder_fifth','xception_taskonomy_joined_decoder_quad','xception_taskonomy_joined_decoder_half','xception_taskonomy_joined_decoder_80','xception_taskonomy_joined_decoder_ozan']

# model_urls = {
#     'xception_taskonomy':'file:///home/tstand/Dropbox/taskonomy/xception_taskonomy-a4b32ef7.pth.tar'
# }


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,groupsize=1):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=max(1,in_channels//groupsize),bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        #self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
        #self.pointwise=lambda x:x
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters=out_filters

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            #rep.append(nn.AvgPool2d(3,strides,1))
            rep.append(nn.Conv2d(filters,filters,2,2))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x+=skip
        return x

class Encoder(nn.Module):
    def __init__(self, sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, sizes[0], 3,2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(sizes[0],sizes[1],3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(sizes[1])
        #do relu here

        self.block1=Block(sizes[1],sizes[2],2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(sizes[2],sizes[3],2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(sizes[3],sizes[4],2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(sizes[4],sizes[5],3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(sizes[5],sizes[6],3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(sizes[6],sizes[7],3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(sizes[7],sizes[8],3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(sizes[8],sizes[9],3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(sizes[9],sizes[10],3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(sizes[10],sizes[11],3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(sizes[11],sizes[12],3,1,start_with_relu=True,grow_first=True)

        #self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        #self.conv3 = SeparableConv2d(768,512,3,1,1)
        #self.bn3 = nn.BatchNorm2d(512)
        #self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        #self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        #self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        #self.bn4 = nn.BatchNorm2d(2048)
    def forward(self,input):
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        #x = self.block12(x)
        
        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = self.relu(x)

        
        #x = self.conv4(x)
        #x = self.bn4(x)

        representation = self.relu2(x)

        return representation



def interpolate(inp,size):
    t = inp.type()
    inp = inp.float()
    out = nn.functional.interpolate(inp,size=size,mode='bilinear',align_corners=False)
    if out.type()!=t:
        out = out.half()
    return out



class Decoder(nn.Module):
    def __init__(self, output_channels=32,num_classes=None):
        super(Decoder, self).__init__()
        
        self.output_channels = output_channels
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)
        if num_classes is not None:
            self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

            self.conv3 = SeparableConv2d(1024,1536,3,1,1)
            self.bn3 = nn.BatchNorm2d(1536)

            #do relu here
            self.conv4 = SeparableConv2d(1536,2048,3,1,1)
            self.bn4 = nn.BatchNorm2d(2048)

            self.fc = nn.Linear(2048, num_classes)
        else:
            self.upconv1 = nn.ConvTranspose2d(512,128,2,2)
            self.bn_upconv1 = nn.BatchNorm2d(128)
            self.conv_decode1 = nn.Conv2d(128, 128, 3,padding=1)
            self.bn_decode1 = nn.BatchNorm2d(128)
            self.upconv2 = nn.ConvTranspose2d(128,64,2,2)
            self.bn_upconv2 = nn.BatchNorm2d(64)
            self.conv_decode2 = nn.Conv2d(64, 64, 3,padding=1)
            self.bn_decode2 = nn.BatchNorm2d(64)
            self.upconv3 = nn.ConvTranspose2d(64,48,2,2)
            self.bn_upconv3 = nn.BatchNorm2d(48)
            self.conv_decode3 = nn.Conv2d(48, 48, 3,padding=1)
            self.bn_decode3 = nn.BatchNorm2d(48)
            self.upconv4 = nn.ConvTranspose2d(48,32,2,2)
            self.bn_upconv4 = nn.BatchNorm2d(32)
            self.conv_decode4 = nn.Conv2d(32, output_channels, 3,padding=1)



    def forward(self,representation):
        if self.num_classes is None:
            x = self.upconv1(representation)
            x = self.bn_upconv1(x)
            x = self.relu(x)
            x = self.conv_decode1(x)
            x = self.bn_decode1(x)
            x = self.relu(x)
            x = self.upconv2(x)
            x = self.bn_upconv2(x)
            x = self.relu(x)
            x = self.conv_decode2(x)
            
            x = self.bn_decode2(x)
            x = self.relu(x)
            x = self.upconv3(x)
            x = self.bn_upconv3(x)
            x = self.relu(x)
            x = self.conv_decode3(x)
            x = self.bn_decode3(x)
            x = self.relu(x)
            x = self.upconv4(x)
            x = self.bn_upconv4(x)
            x = self.relu(x)
            x = self.conv_decode4(x)

        else:
            x = self.block12(representation)
        
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)

            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x




class XceptionTaskonomy(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self,size=1, tasks=None,num_classes=None, ozan=False):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(XceptionTaskonomy, self).__init__()
        pre_rep_size=728
        sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
        if size == 1:
            sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
        elif size==.2:
            sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
        elif size==.3:
            sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
        elif size==.4:
            sizes=[32,64,128,256,728,728,728,728,728,728,728,728,728]
        elif size==.5:
            sizes=[24,48,96,192,512,512,512,512,512,512,512,512,512]
        elif size==.8:
            sizes=[32,64,128,248,648,648,648,648,648,648,648,648,648]
        elif size==2:
            sizes=[32,64, 128,256, 728, 728, 728, 728, 728, 728, 728, 728, 728]
        elif size==4:
            sizes=[64,128,256,512,1456,1456,1456,1456,1456,1456,1456,1456,1456]
            

        self.encoder=Encoder(sizes=sizes)
        pre_rep_size=sizes[-1]

        self.tasks=tasks
        self.ozan=ozan
        self.task_to_decoder = {}



        if tasks is not None:
            
            self.final_conv = SeparableConv2d(pre_rep_size,512,3,1,1)
            self.final_conv_bn = nn.BatchNorm2d(512)
            output_channels=0
            self.channels_per_task = {'segment_semantic':18,
                                      'depth_zbuffer':1,
                                      'normal':3,
                                      'edge_occlusion':1,
                                      'reshading':3,
                                      'keypoints2d':1,
                                      'edge_texture':1,
                                     }
            for task in tasks:
                output_channels+=self.channels_per_task[task]
            self.decoder=Decoder(output_channels)
            
        else:
            self.decoder=Decoder(output_channels=0,num_classes=1000)

        
        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------
    

    def forward(self, input):
        rep = self.encoder(input)


        if self.tasks is None:
            return self.decoder(rep)
        
        rep = self.final_conv(rep)
        rep = self.final_conv_bn(rep)

        outputs = {}
        raw_output=self.decoder(rep)

        range_start = 0
        #print(raw_output.shape)
        for task in self.tasks:
            outputs[task]=raw_output[:,range_start:range_start+self.channels_per_task[task],:,:]
            range_start+=self.channels_per_task[task]
        
        return outputs



def xception_taskonomy_joined_decoder(**kwargs):
    """
    Construct Xception.
    """
    
    model = XceptionTaskonomy(**kwargs,size=1)

    return model

def xception_taskonomy_joined_decoder_fifth(**kwargs):
    """
    Construct Xception.
    """
    
    model = XceptionTaskonomy(**kwargs,size=.2)

    return model

def xception_taskonomy_joined_decoder_quad(**kwargs):
    """
    Construct Xception.
    """
    
    model = XceptionTaskonomy(**kwargs,size=4)

    return model

def xception_taskonomy_joined_decoder_half(**kwargs):
    """
    Construct Xception.
    """
    
    model = XceptionTaskonomy(**kwargs,size=.5)

    return model

def xception_taskonomy_joined_decoder_80(**kwargs):
    """
    Construct Xception.
    """
    
    model = XceptionTaskonomy(**kwargs,size=.8)

    return model

def xception_taskonomy_joined_decoder_ozan(**kwargs):
    """
    Construct Xception.
    """
    
    model = XceptionTaskonomy(ozan=True,**kwargs)

    return model
