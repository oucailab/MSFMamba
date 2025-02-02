import torch
import torch.nn as nn
import yaml
from modules.Mamba_v3 import *

class Syn_layer(nn.Module):
    def __init__(self,hc1,hc2,sc1,sc2,hsi_N,img_size,d_state,lay_n,expand):
        super().__init__()
        N = 9 - lay_n * 2
        N1 = hsi_N - N + 1
        self.hsi_conv = self._create_hsi_conv_layer(hc1, hc2 , kernel_size=(N, 3, 3))
        self.sar_conv = self._create_conv2d_layer(sc1, sc2, kernel_size=3)
        self.MBh = MSBlock(hidden_dim = hc2 * N1, d_state=d_state,expand=expand)
        self.SpBh = SpecMambaBlock(hidden_dim = (img_size-2)**2, d_state=d_state,expand=expand)
        self.MBx = MSBlock(hidden_dim=sc2, d_state=d_state,expand=expand)
        self.FSSBlock = FSSBlock(hidden_dim1=hc2*(hsi_N - N + 1), hidden_dim2=sc2, d_state=d_state,expand=expand)
    
    def _create_hsi_conv_layer(self, in_channels, out_channels, kernel_size, conv2d=False):
        if conv2d:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0),
                nn.BatchNorm3d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )

    def _create_conv2d_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, hsi, sar):
        hsi = self.hsi_conv(hsi)
        sar = self.sar_conv(sar)
        hsi = self.SpBh(self.MBh(hsi))
        sar = self.MBx(sar)
        out_hsi,out_sar = self.FSSBlock(hsi,sar)
        return out_hsi,out_sar

class Net(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        with open('dataset_info.yaml', 'r') as file:
            data = yaml.safe_load(file)
        data = data[dataset]
        
        self.out_features = data['num_classes']
        d_state = data['d_state']
        img_size = data['window_size']
        cha_h = [8, 16, 24, 32]
        cha_x = [64, 128, 192, 256]

        expand = data['expand']
        pca_num = data['pca_num']
        sar_num = data['slar_channel_num']
        self.layer_num = data['num_layer']
        self.layers = nn.ModuleList()
        for i in range(self.layer_num): 
            if(i==0):
                self.layers.append(Syn_layer(1,cha_h[i],sar_num,cha_x[i],pca_num,img_size,d_state,i,expand))
            else:
                self.layers.append(Syn_layer(cha_h[i-1],cha_h[i],cha_x[i-1],cha_x[i],pca_num,img_size,d_state,i,expand))
            pca_num = pca_num - (9-2*(i+1)+1)
            img_size = img_size - 2

        self.fusionlinear_fusion = nn.Linear(in_features=(cha_x[self.layer_num - 1] + cha_h[self.layer_num - 1] * pca_num ) * (img_size ** 2), out_features=self.out_features)

    def forward(self, hsi, sar):
        for layer in self.layers:
            hsi,sar = layer(hsi,sar)          
        
        B,C,N,H,W = hsi.size()
        hsi = hsi.reshape(B,C*N,H,W)

        fusion_feat_m = torch.cat((hsi,sar),dim=1)
        B,C,H,W = fusion_feat_m.size()
        fusion_feat = fusion_feat_m.reshape(B , -1)
        output_fusion = self.fusionlinear_fusion(fusion_feat)
        
        return fusion_feat_m,output_fusion