import torch
import torch.nn as nn
import numpy as np

affine=True
def my_crop(x,y):
    #crop center y 
    h,w = x.size(2),x.size(3)
    h1,w1 = y.size(2),y.size(3)
    a = int(round((h1-h)/2))
    b = int(round((w1-w)/2))
    return y[:,:,a:a+h,b:b+w]



class DSSNet(nn.Module):
    
   
    def __init__(self):
        super(DSSNet,self).__init__()
        #vgg mdoel
        self.conv1_1 = nn.Conv2d(3,64,3,padding=5)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64,64,3,padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64,128,3,padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128,128,3,padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256,256,3,padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256,256,3,padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256,512,3,padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512,512,3,padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512,512,3,padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(3,stride=2,padding=1,ceil_mode=True)
 
        self.conv5_1 = nn.Conv2d(512,512,3,padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512,512,3,padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512,512,3,padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(3,stride=2,padding=1,ceil_mode=True)       
        self.pool5a = nn.AvgPool2d(3,stride=1,padding=1)

        #DSN conv
        self.conv1_dsn6 = nn.Conv2d(512,512,7,padding=3)
        self.relu1_dsn6 = nn.ReLU(inplace=True)
        self.conv2_dsn6 = nn.Conv2d(512,512,7,padding=3)
        self.relu2_dsn6 = nn.ReLU(inplace=True)
        self.conv3_dsn6 = nn.Conv2d(512,1,1)
        self.upsample32_in_dsn6 = nn.ConvTranspose2d(1,1,64,stride=32,bias=False)

        self.conv1_dsn5 = nn.Conv2d(512,512,5,padding=2)
        self.relu1_dsn5 = nn.ReLU(inplace=True)
        self.conv2_dsn5 = nn.Conv2d(512,512,5,padding=2)
        self.relu2_dsn5 = nn.ReLU(inplace=True)
        self.conv3_dsn5 = nn.Conv2d(512,1,1)
        self.upsample16_in_dsn5 = nn.ConvTranspose2d(1,1,32,stride=16,bias=False)

        self.conv1_dsn4 = nn.Conv2d(512,256,5,padding=2)
        self.relu1_dsn4 = nn.ReLU(inplace=True)
        self.conv2_dsn4 = nn.Conv2d(256,256,5,padding=2)
        self.relu2_dsn4 = nn.ReLU(inplace=True)
        self.conv3_dsn4 = nn.Conv2d(256,1,1)
        self.upsample4_dsn6 = nn.ConvTranspose2d(1,1,8,stride=4,bias=False)#conv3_dsn6
        self.upsample2_dsn5 = nn.ConvTranspose2d(1,1,4,stride=2,bias=False)#conv3_dsn5
        self.conv4_dsn4 = nn.Conv2d(3,1,1)
        self.upsample8_in_dsn4 = nn.ConvTranspose2d(1,1,16,stride=8,bias=False)

        self.conv1_dsn3 = nn.Conv2d(256,256,5,padding=2)
        self.relu1_dsn3 = nn.ReLU(inplace=True)
        self.conv2_dsn3 = nn.Conv2d(256,256,5,padding=2)
        self.relu2_dsn3 = nn.ReLU(inplace=True)
        self.conv3_dsn3 = nn.Conv2d(256,1,1)
        self.upsample8_dsn6 = nn.ConvTranspose2d(1,1,16,stride=8,bias=False)#conv3_dsn6
        self.upsample4_dsn5 = nn.ConvTranspose2d(1,1,8,stride=4,bias=False)#conv3_dsn5
        self.conv4_dsn3 = nn.Conv2d(3,1,1)
        self.upsample4_in_dsn3 = nn.ConvTranspose2d(1,1,8,stride=4,bias=False)

        self.conv1_dsn2 = nn.Conv2d(128,128,3,padding=1)
        self.relu1_dsn2 = nn.ReLU(inplace=True)
        self.conv2_dsn2 = nn.Conv2d(128,128,3,padding=1)
        self.relu2_dsn2 = nn.ReLU(inplace=True)
        self.conv3_dsn2 = nn.Conv2d(128,1,1)
        self.upsample16_dsn6 = nn.ConvTranspose2d(1,1,32,stride=16,bias=False)#conv3_dsn6
        self.upsample8_dsn5 = nn.ConvTranspose2d(1,1,16,stride=8,bias=False)#conv3_dsn5
        self.upsample4_dsn4 = nn.ConvTranspose2d(1,1,8,stride=4,bias=False)#conv4_dsn4
        self.upsample2_dsn3 = nn.ConvTranspose2d(1,1,4,stride=2,bias=False)#conv4_dsn3
        self.conv4_dsn2 = nn.Conv2d(5,1,1)
        self.upsample2_in_dsn2 = nn.ConvTranspose2d(1,1,4,stride=2,bias=False)

        self.conv1_dsn1 = nn.Conv2d(64,128,3,padding=1)
        self.relu1_dsn1 = nn.ReLU(inplace=True)
        self.conv2_dsn1 = nn.Conv2d(128,128,3,padding=1)
        self.relu2_dsn1 = nn.ReLU(inplace=True)
        self.conv3_dsn1 = nn.Conv2d(128,1,1)
        self.upsample32_dsn6 = nn.ConvTranspose2d(1,1,64,stride=32,bias=False)
        self.upsample16_dsn5 = nn.ConvTranspose2d(1,1,32,stride=16,bias=False)
        self.upsample8_dsn4 = nn.ConvTranspose2d(1,1,16,stride=8,bias=False)
        self.upsample4_dsn3 = nn.ConvTranspose2d(1,1,8,stride=4,bias=False)
        self.conv4_dsn1 = nn.Conv2d(5,1,1)

        self.conv_fuse = nn.Conv2d(6,1,1)
        self.init_weight()

    def forward(self,x):
        out = []
        h = x
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        h1 = x
        x = self.pool1(x)

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        h2 = x
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        h3 = x
        x = self.pool3(x)

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        h4 = x
        x = self.pool4(x)

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        h5 = x
        x = self.pool5(x)
        x = self.pool5a(x)
        
        h6 = x
        h6 = self.relu1_dsn6(self.conv1_dsn6(h6))
        h6 = self.relu2_dsn6(self.conv2_dsn6(h6))
        h6 = self.conv3_dsn6(h6)  #conv3_dsn6
        score6 = my_crop(h,self.upsample32_in_dsn6(h6))

        h5 = self.relu1_dsn5(self.conv1_dsn5(h5))
        h5 = self.relu2_dsn5(self.conv2_dsn5(h5))
        h5 = self.conv3_dsn5(h5)
        score5 = my_crop(h,self.upsample16_in_dsn5(h5))

        h4 = self.relu1_dsn4(self.conv1_dsn4(h4))
        h4 = self.relu2_dsn4(self.conv2_dsn4(h4))
        h4 = self.conv3_dsn4(h4)
        h4 = torch.cat((h4,my_crop(h4,self.upsample4_dsn6(h6)),my_crop(h4,self.upsample2_dsn5(h5))),1)
        h4 = self.conv4_dsn4(h4)
        score4 = my_crop(h,self.upsample8_in_dsn4(h4))

        h3 = self.relu1_dsn3(self.conv1_dsn3(h3))
        h3 = self.relu2_dsn3(self.conv2_dsn3(h3))
        h3 = self.conv3_dsn3(h3)
        h3 = torch.cat((h3,my_crop(h3,self.upsample8_dsn6(h6)), my_crop(h3,self.upsample4_dsn5(h5))),1) 
        h3 = self.conv4_dsn3(h3)
        score3 = my_crop(h,self.upsample4_in_dsn3(h3))

        h2 = self.relu1_dsn2(self.conv1_dsn2(h2))
        h2 = self.relu2_dsn2(self.conv2_dsn2(h2))
        h2 = self.conv3_dsn2(h2)
        h2 = torch.cat((h2, my_crop(h2,self.upsample8_dsn5(h5)),
            my_crop(h2,self.upsample4_dsn4(h4)),my_crop(h2,self.upsample16_dsn6(h6)),my_crop(h2,self.upsample2_dsn3(h3))),1)
        h2 = self.conv4_dsn2(h2)
        score2 = my_crop(h,self.upsample2_in_dsn2(h2))

        h1 = self.relu1_dsn1(self.conv1_dsn1(h1))
        h1 = self.relu2_dsn1(self.conv2_dsn1(h1))
        h1 = self.conv3_dsn1(h1)
        h1 = torch.cat((h1,my_crop(h1,self.upsample16_dsn5(h5)),my_crop(h1,self.upsample8_dsn4(h4)),my_crop(h1,self.upsample32_dsn6(h6)),my_crop(h1,self.upsample4_dsn3(h3))),1)
        h1 = self.conv4_dsn1(h1)
        score1 = my_crop(h,h1)
        score_fuse = self.conv_fuse(torch.cat((score1,score2,score3,score4,score5,score6),1))
        out.append(score1)
        out.append(score2)
        out.append(score3)
        out.append(score4)
        out.append(score5)
        out.append(score6)
        out.append(score_fuse)
        return out
    

    def init_weight(self):
        add_conv=[
        self.conv1_dsn6,
        self.conv2_dsn6,  
        self.conv3_dsn6,
        self.conv1_dsn5,
        self.conv2_dsn5,
        self.conv3_dsn5,
        self.conv1_dsn4,
        self.conv2_dsn4,
        self.conv3_dsn4,
        self.conv4_dsn4, 
        self.conv1_dsn3,
        self.conv2_dsn3,
        self.conv3_dsn3,
        self.conv4_dsn3,
        self.conv1_dsn2,
        self.conv2_dsn2,
        self.conv3_dsn2,
        self.conv4_dsn2,
        self.conv1_dsn1,
        self.conv2_dsn1,
        self.conv3_dsn1,
        self.conv4_dsn1,
                ]
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                if m in add_conv:
                    m.weight.data.normal_(0,0.01)
                    m.bias.data.zero_()
                elif m is self.conv_fuse:
                    #ck = torch.ones(1,6,1,1)*0.1667
                    m.weight.data.fill_(0.1667)
                    m.bias.data.zero_()
                else:
                    m.weight.data.zero_()
                    m.bias.data.zero_()
            if isinstance(m,nn.ConvTranspose2d):
                initial_weight = self.get_upsampling_weight(m.in_channels,
                        m.out_channels,m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
                m.weight.requires_grad = False

    def get_upsampling_weight(self,in_channels,out_channels,kernel_size):
        factor = (kernel_size +1)//2
        if kernel_size % 2 ==1:
            center = factor -1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size,:kernel_size]
        filt = (1-abs(og[0]-center)/factor)*\
                (1-abs(og[1]-center)/factor)
        weight = np.zeros((in_channels,out_channels,kernel_size,kernel_size),
                dtype=np.float64)
        weight[range(in_channels),range(out_channels),:,:]=filt
        return torch.from_numpy(weight).float()

    def copy_from_vgg(self,vgg):
        features = [
                self.conv1_1,self.relu1_1,
                self.conv1_2,self.relu1_2,
                self.pool1,
                self.conv2_1,self.relu2_1,
                self.conv2_2,self.relu2_2,
                self.pool2,
                self.conv3_1,self.relu3_1,
                self.conv3_2,self.relu3_2,
                self.conv3_3,self.relu3_3,
                self.pool3,
                self.conv4_1,self.relu4_1,
                self.conv4_2,self.relu4_2,
                self.conv4_3,self.relu4_3,
                self.pool4,
                self.conv5_1,self.relu5_1,
                self.conv5_2,self.relu5_2,
                self.conv5_3,self.relu5_3,
                self.pool5,
                ]
        for l1,l2 in zip(vgg.features,features):
            if isinstance(l1,nn.Conv2d) and isinstance(l2,nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

