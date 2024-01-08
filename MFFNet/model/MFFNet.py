 # coding=utf-8
from inspect import classify_class_attrs
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.conv import Conv2d
#from model.resattention import res_cbam
import torchvision.models as models
#from model.res2fg import res2net
import torch.nn.functional as F
import math
from model.vgg import B2_VGG
class SelfAttention_1(nn.Module):
    def __init__(self,in_channels):
        super(SelfAttention_1, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels
        #max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        #bn = nn.BatchNorm2d

        self.g = nn.Conv2d(in_channels,in_channels,1)
        self.theta = nn.Conv2d(in_channels,in_channels,1)
        self.phi = nn.Conv2d(in_channels,in_channels,1)
    def forward(self,x3):

        batch_size = x3.size(0)

        g_x = self.g(x3).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)#[b,wh,c]

        theta_x = self.theta(x3).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x3).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x3.size()[2:])
        out = y + x3
        
        return out

class GC(nn.Module):
    def __init__(self,in_channels):
        super(GC,self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_channels,in_channels),requires_grad=True)
        self.reset_para()
    def reset_para(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
    def forward(self,x):
        batch_size = x.size(0)
        channel = x.size(1)
        #print(channel)
        g_x = x.view(batch_size, channel, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)#[b,wh,c]
        theta_x = x.view(batch_size, channel, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = x.view(batch_size, channel, -1)
        
        f = torch.matmul(theta_x, phi_x)

        adj = F.softmax(f, dim=-1)
        #print(g_x.size())
        #print(self.weight.size())
        support = torch.matmul(g_x,self.weight)
        y = torch.matmul(adj,support)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, *x.size()[2:])
        return y+x

class SE(nn.Module):
    def __init__(self):
        super(SE,self).__init__()
        self.conv_1x1_0 = nn.Conv2d(128,64,kernel_size=1)
        self.conv_1x1_1 = nn.Conv2d(128,64,kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(128,64,kernel_size=1)
        self.CA_4_1 = ChannelAttention(128)
        self.CA_1_0 = ChannelAttention(128)
        self.conv_3x3_1 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.conv_3x3_0 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.conv_1x1_out = nn.Conv2d(64,1,kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
    def forward(self,x0,x1,x4):
        f0 = self.conv_1x1_0(x0)
        f1 = self.conv_1x1_1(x1)
        f4 = self.conv_1x1_4(x4)
        S1 = self.conv_3x3_1(self.CA_4_1(torch.cat((f1,self.upsample3(f4)),1)))
        S0 = self.conv_3x3_0(self.CA_1_0(torch.cat((f0,self.upsample1(S1)),1)))
        out = F.sigmoid(self.conv_1x1_out(S0))
        return out
class TDF(nn.Module):
    def __init__(self,in_channels):
        super(TDF,self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.conv_2 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv_3 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x1,x2,x3):
        m2_c = self.ca(x1)*x2
        m3_c = self.ca(x1)*x3

        m2_s = self.sa(x1)*x2
        m3_s = self.sa(x1)*x3

        out = x1+self.conv_2(torch.cat((m2_c,m2_s),1))+self.conv_3(torch.cat((m3_c,m3_s),1))
        return out
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.P4 = PFM(128)
        self.P3 = PFM(128)
        self.conv_3x3_4_3 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_3_2 = nn.Conv2d(512,128,kernel_size=3,padding=1)
        self.conv_3x3_2_1 = nn.Conv2d(512,128,kernel_size=3,padding=1)
        self.conv_3x3_1_0 = nn.Conv2d(512,128,kernel_size=3,padding=1)

        self.conv_1x1_0 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_1x1_1 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_1x1_2 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_1x1_3 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(128,1,kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
    def forward(self,x0,x1,x2,x3,x4):
        p4 = self.P4(x4)
        p3 = self.P3(x3)
        f3 = self.conv_3x3_4_3(torch.cat((p3,self.upsample1(p4)),1))
        f2 = self.conv_3x3_3_2(torch.cat((x2,self.upsample1(f3),self.upsample1(p3),self.upsample2(p4)),1))
        f1 = self.conv_3x3_2_1(torch.cat((x1,self.upsample1(f2),self.upsample2(p3),self.upsample3(p4)),1))
        f0 = self.conv_3x3_1_0(torch.cat((x0,self.upsample1(f1),self.upsample3(p3),self.upsample4(p4)),1))

        #s0 = F.sigmoid(self.conv_1x1_0(f0))
        #s1 = F.sigmoid(self.conv_1x1_1(f1))
        #s2 = F.sigmoid(self.conv_1x1_2(f2))
        #s3 = F.sigmoid(self.conv_1x1_3(f3))
        #s4 = F.sigmoid(self.conv_1x1_4(p4))
        return f0,f1,f2,f3,p4
class MMI(nn.Module):
    def __init__(self,in_channels):
        super(MMI,self).__init__()
        self.CA_r = ChannelAttention_2(in_channels)
        self.SA_r = SpatialAttention()
        self.CA_d = ChannelAttention_2(in_channels)
        self.CA_t = ChannelAttention_2(in_channels)
        self.SA_dt = SpatialAttention()
        self.conv1 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(3*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,r,d,t):
        sa_r = self.SA_r(self.CA_r(r)*r)
        d_r = d+sa_r*d
        t_r = t+sa_r*t
        dt = self.conv1(torch.cat((self.CA_d(d_r)*d_r,self.CA_t(t_r)*t_r),1))
        sa_dt = self.SA_dt(dt)
        out = r*sa_dt+dt*sa_dt
        return out
class MF(nn.Module):
    def __init__(self):
        super(MF,self).__init__()
        self.sa =SpatialAttention()
        self.conv3 = nn.Conv2d(128*3,128,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(128*3,128,kernel_size=3,padding=1)
        self.conv1 = nn.Conv2d(128*3,128,kernel_size=3,padding=1)
    def forward(self,h,l,e):
        a1 = self.conv1(torch.cat((h*e,h,e),1))
        a2 = self.conv2(torch.cat((l*e,l,e),1))
        a3 =self.conv3(torch.cat((a1*a2,a1,a2),1))
        out = a3+a3*self.sa(a3)
        return out
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)*x

class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_1, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class ChannelAttention_2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_2, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)#dim=1是因为传入数据是多加了一维unsqueeze(0)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)
class SpatialAttention_m(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_m, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(x.size())
        max_out = torch.mean(x, dim=1, keepdim=True)#dim=1是因为传入数据是多加了一维unsqueeze(0)
        x=max_out
        #print(x.size())
        x = self.conv1(x)
        
        return self.sigmoid(x)


#mid level
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.att1 = ChannelAttention_1(in_channels)
        self.att2 = ChannelAttention_2(in_channels)
        self.conv = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        a1 = x*self.att1(x)
        a2 = x*self.att2(x)
        out = self.conv(torch.cat((a1,a2),1))
        return out
class WCF(nn.Module):# Pyramid Feature Module
    def __init__(self,in_channels):
        super(WCF,self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=4, dilation=4)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=6, dilation=6)
        )



        self.conv = nn.Conv2d(4*in_channels,in_channels,kernel_size=3,padding=1)
        self.sa = SpatialAttention()
    def forward(self,x):
        #x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.branch0(x)
        y1 = self.branch1(x+y0)
        y2 = self.branch2(x+y1+y0)
        y3 = self.branch3(x+y2+y1+y0)

        y = self.conv(torch.cat((y0,y1,y2,y3),1))

        out = y+y*self.sa(y)+x
        return out


class BP(nn.Module):
    def __init__(self,in_channels):
        super(BP,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv11 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv22 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv33 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)

        self.conv4 = nn.Conv2d(3*in_channels,in_channels,kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv_out = nn.Conv2d(in_channels,1,kernel_size=1)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
    def forward(self,x1,x2,x3):
        a = self.conv1(x1)
        b = self.conv2(x2)
        c = self.conv3(x3)

        b1 = self.conv22(torch.cat((b,c),1))
        a1 = self.conv33(torch.cat((a,b1),1))

        y3 = c*self.sa3(c)+c
        y2 = b1*self.sa2(b1)+b1
        y1 = a1*self.sa1(a1)+a1
        out = self.conv5(self.conv4(torch.cat((a,b,c),1)))
        #out = self.conv_out(out)
        
        return out
class RO(nn.Module):
    def __init__(self,in_channels):
        super(RO,self).__init__()    
        self.sa_max = SpatialAttention()
        self.sa_mea = SpatialAttention_m()
        self.conv = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.sa_max_1 = SpatialAttention()
        self.sa_mea_1 = SpatialAttention_m()
        self.conv_1 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        y0 = self.conv(torch.cat((self.sa_max(x)*x,self.sa_mea(x)*x),1))+x
        y1 = self.conv_1(torch.cat((self.sa_max_1(y0)*y0,self.sa_mea_1(y0)*y0),1))+y0
        return y1
class CBR(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CBR,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.Ba = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.Ba(self.conv(x)))

        return out
class MFFNet(nn.Module):#输入三通道
    def __init__(self, in_channels):
        super(MFFNet, self).__init__()
        #resnet_RGB = models.resnet18(pretrained=True)
        #resnet_D = models.resnet18(pretrained=True)
        #resnet_T = models.resnet18(pretrained=True)

        resnet_RGB = B2_VGG()
        resnet_D = B2_VGG()
        resnet_T = B2_VGG()
        #resnet2 = models.resnet18(pretrained=True)
        #self.weight=nn.Parameter(torch.FloatTensor(1))
        #
        #reanet = res2net()
        #res2n
        # ************************* Encoder ***************************
        # input conv3*3,64
        self.conv_RGB = resnet_RGB.conv1
        #self.bn_RGB = resnet_RGB.bn1
        #self.relu_RGB = resnet_RGB.relu
        #self.maxpool_RGB = resnet_RGB.maxpool#计算得到112.5 但取112 向下取整
        # Extract Features
        self.encoder1_RGB = resnet_RGB.conv2
        self.encoder2_RGB = resnet_RGB.conv3
        self.encoder3_RGB = resnet_RGB.conv4_1
        self.encoder4_RGB = resnet_RGB.conv5_1
        
        self.conv_D = nn.Conv2d(1,64,kernel_size=3,padding=1)
        #self.bn_D = resnet_D.bn1
        #self.relu_D = resnet_D.relu
        #self.maxpool_D = resnet_D.maxpool#计算得到112.5 但取112 向下取整
        # Extract Features
        self.encoder1_D = resnet_D.conv2
        self.encoder2_D = resnet_D.conv3
        self.encoder3_D = resnet_D.conv4_1
        self.encoder4_D = resnet_D.conv5_1

        self.conv_T = resnet_T.conv1
        #self.bn_T = resnet_T.bn1
        #self.relu_T = resnet_T.relu
        #self.maxpool_T = resnet_T.maxpool#计算得到112.5 但取112 向下取整
        # Extract Features
        self.encoder1_T = resnet_T.conv2
        self.encoder2_T = resnet_T.conv3
        self.encoder3_T = resnet_T.conv4_1
        self.encoder4_T = resnet_T.conv5_1

        self.conv_1x1_0_RGB = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_RGB = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_RGB = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_RGB = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_RGB = nn.Conv2d(512,128,kernel_size=1)
        
        self.conv_1x1_0_D = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_D = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_D = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_D = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_D = nn.Conv2d(512,128,kernel_size=1)

        self.conv_1x1_0_T = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_T = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_T = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_T = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_T = nn.Conv2d(512,128,kernel_size=1)

        
        self.tdf1_r = TDF(64)
        self.tdf2_r = TDF(128)
        self.tdf3_r = TDF(256)
        self.tdf4_r = TDF(512)

        self.tdf1_d = TDF(64)
        self.tdf2_d = TDF(128)
        self.tdf3_d = TDF(256)
        self.tdf4_d = TDF(512)

        self.tdf1_t = TDF(64)
        self.tdf2_t = TDF(128)
        self.tdf3_t = TDF(256)
        self.tdf4_t = TDF(512)

        self.conv_en_r_0 = nn.Conv2d(64,128,kernel_size=1)
        self.conv_en_r_1 = nn.Conv2d(128,256,kernel_size=1)
        self.conv_en_r_2 = nn.Conv2d(256,512,kernel_size=1)

        self.conv_en_t_0 = nn.Conv2d(64,128,kernel_size=1)
        self.conv_en_t_1 = nn.Conv2d(128,256,kernel_size=1)
        self.conv_en_t_2 = nn.Conv2d(256,512,kernel_size=1)

        self.conv_en_d_0 = nn.Conv2d(64,128,kernel_size=1)
        self.conv_en_d_1 = nn.Conv2d(128,256,kernel_size=1)
        self.conv_en_d_2 = nn.Conv2d(256,512,kernel_size=1)

        self.conv_E0 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_E1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_E2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_E3 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_E4 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        # ************************* Decoder ***************************
        #self.Se_D = SE()
        #self.Se_T = SE()
        self.conv_rd1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_rd0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_rt1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_rt0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_dt1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_dt0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_rdt1 = nn.Conv2d(384,128,kernel_size=1)
        self.conv_rdt0 = nn.Conv2d(384,128,kernel_size=1)

        self.conv4_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv3_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv2_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv1_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv0_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)

        self.conv4_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv1_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv0_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)

        self.bp = BP(128)
        
        self.wcf_0 = WCF(128)
        self.wcf_1 = WCF(128)
        self.wcf_2 = WCF(128)
        self.wcf_3 = WCF(128)
        self.wcf_4 = WCF(128)
        


        self.conv_3x3_0_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_1_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_2_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_3_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_4_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)

        self.conv_3x3_0_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_1_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_2_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_3_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_4_T = nn.Conv2d(256,128,kernel_size=3,padding=1)

        #self.decoder_RGB = Decoder()
        #self.decoder_T = Decoder()

        #self.loc = location(128)
        #self.Fuse = FF()
        #self.fff_4 = FF(128)
        #self.fff_3 = FFF(128)
        #self.fff_2 = FFF(128)
        #self.fff_1 = FFF(128)
        #self.fff_0 = FFF(128)
        self.conv_a2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_a1 = nn.Conv2d(512,128,kernel_size=3,padding=1)
        self.conv_a0 = nn.Conv2d(640,128,kernel_size=3,padding=1)
        self.ro4 = RO(128)
        self.ro3 = RO(128)
        self.ro2 = RO(128)
        self.ro1 = RO(128)
        self.ro0 = RO(128)
        
        #self.fu4 = FF()
        self.mf3 = MF()
        self.mf2 = MF()
        self.mf1 = MF()
        self.mf0 = MF()
        self.conv_s3 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_s2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_s1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_s0 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.at2 = Attention(128)
        self.at1 = Attention(128)
        self.at0 = Attention(128)
        self.conv_2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_1 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_0 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_e = nn.Conv2d(128,1,kernel_size=1)

        # ************************* Feature Map Upsample ***************************
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.downsample3 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.downsample4 = nn.Upsample(scale_factor=0.0625, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.conv_out_4 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_3 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_2 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_1 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_0 = nn.Conv2d(128,1,kernel_size=1)
        
    def forward(self, x_rgb,x_d,x_t):
        # ************************* Encoder ***************************
        #D>>R
        sx_d = self.conv_D(x_d)
        sx_t = self.conv_T(x_t)
        sx_r = self.conv_RGB(x_rgb)

        sx_d_e = self.tdf1_d(sx_d,sx_r,sx_t)
        sx_t_e = self.tdf1_t(sx_t,sx_r,sx_d)
        sx_r_e = self.tdf1_r(sx_r,sx_d,sx_t)
        #sx_d = self.bn_D(sx_d)
        #s0_d = self.relu_D(sx_d)#112
        #sx_d = self.maxpool_D(s0_d)#112
        
        s1_d = self.encoder1_D(sx_d_e)#64
        s1_t = self.encoder1_T(sx_t_e)#64
        s1_r = self.encoder1_RGB(sx_r_e)#112
        
        s1_d_e = self.tdf2_d(s1_d,s1_r,s1_t)
        s1_t_e = self.tdf2_t(s1_t,s1_r,s1_d)
        s1_r_e = self.tdf2_r(s1_r,s1_d,s1_t)
        
        s2_d = self.encoder2_D(s1_d_e+self.downsample(self.conv_en_d_0(sx_d)))#32
        s2_t = self.encoder2_T(s1_t_e+self.downsample(self.conv_en_t_0(sx_t)))#32
        s2_r = self.encoder2_RGB(s1_r_e+self.downsample(self.conv_en_r_0(sx_r)))#56
        s2_d_e = self.tdf3_d(s2_d,s2_r,s2_t)
        s2_t_e = self.tdf3_t(s2_t,s2_r,s2_d)
        s2_r_e = self.tdf3_r(s2_r,s2_d,s2_t)
        
        s3_d = self.encoder3_D(s2_d_e+self.downsample(self.conv_en_d_1(s1_d)))#16
        s3_t = self.encoder3_T(s2_t_e+self.downsample(self.conv_en_t_1(s1_t)))#16
        s3_r = self.encoder3_RGB(s2_r_e+self.downsample(self.conv_en_r_1(s1_r)))#28
        s3_d_e = self.tdf4_d(s3_d,s3_r,s3_t)
        s3_t_e = self.tdf4_t(s3_t,s3_r,s3_d)
        s3_r_e = self.tdf4_r(s3_r,s3_d,s3_t)
        
        s4_d = self.encoder4_D(s3_d_e+self.downsample(self.conv_en_d_2(s2_d)))#8
        s4_t = self.encoder4_T(s3_t_e+self.downsample(self.conv_en_t_2(s2_t)))#8
        s4_r = self.encoder4_RGB(s3_r_e+self.downsample(self.conv_en_r_2(s2_r)))#14
        
        s0_d = self.conv_1x1_0_D(sx_d)
        s1_d = self.conv_1x1_1_D(s1_d)
        s2_d = self.conv_1x1_2_D(s2_d)
        s3_d = self.conv_1x1_3_D(s3_d)
        s4_d = self.conv_1x1_4_D(s4_d)


        
        
        
        
        s0_t = self.conv_1x1_0_T(sx_t)
        s1_t = self.conv_1x1_1_T(s1_t)
        s2_t = self.conv_1x1_2_T(s2_t)
        s3_t = self.conv_1x1_3_T(s3_t)
        s4_t = self.conv_1x1_4_T(s4_t)

       

        s0_r = self.conv_1x1_0_RGB(sx_r)
        s1_r = self.conv_1x1_1_RGB(s1_r)
        s2_r = self.conv_1x1_2_RGB(s2_r)
        s3_r = self.conv_1x1_3_RGB(s3_r)
        s4_r = self.conv_1x1_4_RGB(s4_r)
        ##################################
        
        ###1
        E0 = self.conv_E0(torch.cat((s0_r,s0_d,s0_t),1))
        E1 = self.conv_E1(torch.cat((s1_r,s1_d,s1_t),1))
        E2 = self.conv_E2(torch.cat((s2_r,s2_d,s2_t),1))
        E3 = self.conv_E3(torch.cat((s3_r,s3_d,s3_t),1))
        E4 = self.conv_E4(torch.cat((s4_r,s4_d,s4_t),1))

        E0 = self.wcf_0(E0)
        E1 = self.wcf_1(E1)
        E2 = self.wcf_2(E2)
        E3 = self.wcf_3(E3)
        E4 = self.wcf_4(E4)
        

        e = self.bp(E0,self.upsample1(E1),self.upsample2(E2))


        s4 = self.ro4(E4)
        s3 = self.ro3(self.conv_s3(torch.cat((E3,self.upsample1(s4)),1)))
        s2 = self.ro2(self.conv_s2(torch.cat((E2,self.upsample1(s3),self.upsample2(s4)),1)))
        s1 = self.ro1(self.conv_s1(torch.cat((E1,self.upsample1(s2),self.upsample2(s3)),1)))
        s0 = self.ro0(self.conv_s0(torch.cat((E0,self.upsample1(s1),self.upsample2(s2)),1)))

        b4 = s4
        b3 = self.mf3(s3,self.upsample1(b4),self.downsample3(e))
        b2 = self.mf2(s2,self.upsample1(b3),self.downsample2(e))
        b1 = self.mf1(s1,self.upsample1(b2),self.downsample(e))
        b0 = self.mf0(s0,self.upsample1(b1),e)
        
        
        Sal0 = F.sigmoid(self.conv_out_0(b0))
        Sal1 = F.sigmoid(self.conv_out_1(b1))
        Sal2 = F.sigmoid(self.conv_out_2(b2))
        Sal3 = F.sigmoid(self.conv_out_3(b3))
        Sal4 = F.sigmoid(self.conv_out_4(b4))
        E = F.sigmoid(self.conv_e(e))
        return Sal0,self.upsample1(Sal1),self.upsample2(Sal2),self.upsample3(Sal3),self.upsample4(Sal4),E#,G_d,G_t


