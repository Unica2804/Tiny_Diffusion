import torch
import torch.nn as nn

class Resblock3D(nn.Module):
    def __init__(self,in_ch,out_ch,t_emb_dim):
        super().__init__()

        self.time_mlp = nn.Linear(t_emb_dim,out_ch)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch,out_ch,kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch,out_ch,kernel_size=(3,1,1), padding=(1,0,0))
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch,out_ch,kernel_size=(1,3,3), padding=(0,1,1)),
            nn.Conv3d(out_ch,out_ch,kernel_size=(3,1,1), padding=(1,0,0))
        )
        self.reLU=nn.ReLU()
        self.res_conv = nn.Conv3d(in_ch, out_ch,1) if in_ch != out_ch else nn.Identity()

    def forward(self,x,t):
        h= self.conv1(x)
        h+= self.time_mlp(t)[:,:,None,None,None]
        h= self.conv2(h)
        return self.reLU(h+self.res_conv(x))
    

class tiny3Dunet(nn.Module):
    def __init__(self):
        super().__init__()
        t_dim=256

        self.time_mlp = nn.Sequential(
            nn.Linear(1,t_dim),
            nn.ReLU(),
            nn.Linear(t_dim,t_dim)
        )

        self.inc = nn.Conv3d(2,32,kernel_size=3,padding=1)
        self.down1 = Resblock3D(32,64,t_dim)
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2))
        self.down2 = Resblock3D(64,128,t_dim)

        self.mid = Resblock3D(128,128,t_dim)

        self.up = nn.Upsample(scale_factor=(1,2,2),mode='trilinear',align_corners=False)
        self.up_block1 = Resblock3D(256,64,t_dim)
        self.up_block2 = Resblock3D(128,32,t_dim)

        self.outc = nn.Conv3d(32,1,kernel_size=1)
    def forward(self,x,t):
        t= t.float().view(-1,1)
        t_emb= self.time_mlp(t)

        x0=self.inc(x)
        x1=self.down1(x0,t_emb)
        p1=self.pool(x1)

        x2=self.down2(p1,t_emb)
        p2=self.pool(x2)

        m= self.mid(p2,t_emb)

        u1=self.up(m)
        u1=torch.cat([u1,x2],dim=1)
        u1=self.up_block1(u1,t_emb)

        u2=self.up(u1)
        u2=torch.cat([u2,x1],dim=1)
        u2=self.up_block2(u2,t_emb)

        return self.outc(u2) 