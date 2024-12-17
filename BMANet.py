import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2

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

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)
        self.conv6=nn.Conv2d(3*channel, channel, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        high_global=self.conv6(x)
        x = self.conv5(x)

        return x,high_global

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
        return self.sigmoid(out)
    

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale
    
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out
    
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv(nn.Module):
    def __init__(self, in_channels,inter_channel, out_channels):
        super(FusionConv, self).__init__()
        dim = inter_channel
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        
        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out

class BAM(nn.Module):
    def __init__(self, in_channels,inter_channel, out_channels):
        super(BAM, self).__init__()
        self.fusion_conv = FusionConv(in_channels ,inter_channel,out_channels)

    def forward(self, x1, x2):
        x_fused = self.fusion_conv(x1, x2)
        return x_fused
    
class BMA(nn.Module):
    def __init__(self, in_channels):#256
        super(BMA, self).__init__()
        self.conv_pred=nn.Conv2d(1,1,1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(in_channels)
        self.pred=nn.Conv2d(in_channels,1,1)

    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        pred = F.interpolate(pred, size=xsize, mode='bilinear', align_corners=True)
        pred = torch.sigmoid(pred)

        background_att = 1 - pred
        background_x= x * background_att
        
        pred_feature=self.conv_pred(pred)
        pred_feature = x * pred_feature

        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)#(1, 1, 22, 22)
        edge_feature = x * edge_input#(1, 256, 22, 22)

        fusion_feature = torch.cat([background_x, pred_feature, edge_feature], dim=1)#(1, 768, 22, 22)
        fusion_feature = self.fusion_conv(fusion_feature)# (1, 256, 22, 22)

        attention_map = self.attention(fusion_feature)#(1, 1, 22, 22)
        fusion_feature = fusion_feature * attention_map# (1, 256, 22, 22)

        out = fusion_feature + residual#(1, 256, 22, 22)
        out = self.cbam(out)#(1, 256, 22, 22)
        pred= self.pred(out)
        return pred
    
class CBR(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(CBR, self).__init__()
        self.cbr=nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3 ,stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
    
    def forward(self,x):
        x=self.cbr(x)
        return x

class BMANet(nn.Module):
    def __init__(self, channel=64):
        super(BMANet, self).__init__()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/home/wzh/Project/SUN_code/BMANet/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # ---- Receptive Field Block like module ----
        self.rfb1_1 = RFB_modified(64, channel)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)

        self.agg = aggregation(channel)

        self.x4_dem_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(320, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.BMA4=BMA(64)
        self.BMA3=BMA(64)
        self.BMA2=BMA(64)

        self.CBR4=CBR(channel,channel)
        self.CBR3=CBR(channel,channel)
        self.CBR2=CBR(channel,channel)
        self.CBR1=CBR(channel,channel)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.BAM=BAM(128,32,1)

    def forward(self, x):
        pvt = self.backbone(x) #(1, 3, 352, 352)
        x1 = pvt[0] #(1, 64, 88, 88)
        x2 = pvt[1] #(1, 128, 44, 44)
        x3 = pvt[2] #(1, 320, 22, 22)
        x4 = pvt[3] #(1, 512, 11, 11)
        
        x1_rfb = self.rfb1_1(x1) 
        x2_rfb = self.rfb2_1(x2) 
        x3_rfb = self.rfb3_1(x3) 
        x4_rfb = self.rfb4_1(x4) 

        global_map,high_global=self.agg(x4_rfb,x3_rfb,x2_rfb)
        x4_rfb=self.upsample4(x4_rfb)
        x3_rfb=self.upsample2(x3_rfb)
        
        high_global=self.CBR4(high_global)
        high_boundary_4=high_global+x4_rfb
        high_boundary_4=self.CBR3(high_boundary_4)

        high_boundary_3=high_boundary_4+x3_rfb
        high_boundary_3=self.CBR2(high_boundary_3)

        high_boundary_2=high_boundary_3+x2_rfb
        high_boundary_2=self.CBR1(high_boundary_2)

        high_boundary_2=self.upsample2(high_boundary_2)
        edge=self.BAM(x1_rfb,high_boundary_2)

        side_out4=self.BMA4(edge,x4_rfb,global_map)

        side_out3=self.BMA3(edge,x3_rfb,side_out4)

        main_out=self.BMA2(edge,x2_rfb,side_out3)

        return main_out, edge, global_map, side_out4, side_out3

if __name__ == '__main__':
    model = BMANet().cuda()
    
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    output=model(input_tensor)
    for out in output:
        print(out.shape)

    # iterations = 300
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # for _ in range(50):
    #     _ = model(input_tensor)

    # times = torch.zeros(iterations)
    # with torch.no_grad():
    #     for iter in range(iterations):
    #         starter.record()
    #         _ = model(input_tensor)
    #         ender.record()
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         times[iter] = curr_time

    # mean_time = times.mean().item()
    # print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))

    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(model, (3, 352, 352), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # tensor = (torch.rand(1, 3, 352, 352).cuda(),)
    # flops = FlopCountAnalysis(model, tensor)
    # print("FLOPs(G): ", flops.total()/1e9)