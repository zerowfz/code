import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os.path
import glob
from DSSNet import DSSNet
from test import evaluate
from dataset import MSRA10K
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
def get_parameter(model,bias=True,scale=False):
    #different params for weight and bias
    b=[
       model.conv3_dsn6,
       model.conv3_dsn4,
       model.conv4_dsn4,
       model.conv3_dsn5,
       model.conv3_dsn3,
       model.conv4_dsn3,
       model.conv3_dsn2,
       model.conv4_dsn2,
       model.conv3_dsn1,
       model.conv4_dsn1,
       model.conv_fuse,
    ]
    if scale:
        for m in b:
            if bias:
                if m.bias is not None:
                    yield m.bias
            else:
                yield m.weight
    else:
        for m in model.modules():
            if isinstance(m,nn.Conv2d) and m not in b:
                if bias:
                    if m.bias is not None:
                        yield m.bias
                else:
                    yield m.weight
    

def criterion(predict,label):
    #predict:1*1*h*w,label:1*h*w
    #predict = F.sigmoid(predict)
    #x1 = torch.log(predict)[0]
    #x2 = torch.log(1-predict)[0]
    #loss = x1[label==1].sum()+x2[label==0].sum()
    #grad_predict = predict.register_hook(lambda grad:grad)
    predict = predict[0][0]
    label = label[0].float()
    tem = (predict>=0).float()
    #grad_tem = tem.register_hook(lambda grad:grad)
    #print(tem.requires_grad)
    #print(predict.requires_grad)
    loss =-(predict*(label-tem)-torch.log(1+torch.exp(predict-2*predict*tem))).sum()
    #print(predict.grad)
    assert ~np.isnan(loss.data.cpu().numpy()),'nan'
    #x = torch.log(torch.cat((1-predict,predict),1))
    #m = nn.NLLLoss2d()
    #label = label.unsqueeze(0)
    #loss=-predict[label==1].mean()
    
    return loss

rootpath = '/home/wfz/Documents/code/saliency/MSRA10K_Imgs_GT'
trainset = MSRA10K(root=rootpath)
dataloader = DataLoader(trainset,shuffle=True,batch_size=1)
testset = MSRA10K(root=rootpath,split='val')
testloader = DataLoader(testset,batch_size=1)

model = DSSNet()
vgg = torchvision.models.vgg16(pretrained=False)
vgg.load_state_dict(torch.load('vgg16_from_caffe.pth'))
model.copy_from_vgg(vgg)
#model.load_state_dict(torch.load('./result/pth/0.01_10000.pth'))
model.cuda()
#lr 'step' update
lr_base = 1e-8
gamma=0.1
iter_size=10
stepsize=8000
#max_iter
max_iter=20000
iter_=0
optimizer = torch.optim.SGD([{'params':get_parameter(model,bias=False),'lr':lr_base},                         {'params':get_parameter(model,bias=True),'lr':2*lr_base,'weight_decay':0},
          {'params':get_parameter(model,bias=False,scale=True),'lr':0.1*lr_base},
        {'params':get_parameter(model,scale=True),'lr':0.2*lr_base,'weight_decay':0}],
                             lr=lr_base,
                             momentum=0.90,
                             weight_decay=0.0005)
optimizer.zero_grad()
maxepoch=100
for epoch in range(0,maxepoch):
    for data,label in dataloader:
        model.train()
        iter_ +=1
        data,label = Variable(data),Variable(label)
        data,label = data.cuda(),label.cuda()
        out = model(data)
        #grad_o = out[6].register_hook(lambda grad:grad)
        #print('before loss')
        #print(label[(label!=0)+(label!=1)-1])
        loss = criterion(out[6],label)
        for i in range(6):
            #1*1*h*w saliency map
            loss += criterion(out[i],label)
        loss /= iter_size
        #test2
        #print(loss.data[0])
        #print(loss)
        #print(iter_)
        loss.backward()
        #print(out[6].grad)
        #print('after loss')
        if iter_ % iter_size==0:
            #print(model.conv_fuse.weight.grad.mean())
            #print(model.conv1_1.weight.mean())
            #print(model.conv3_dsn1.weight.grad.mean())
            #print(model.conv3_dsn2.weight.grad.mean())
            #print(model.conv3_dsn3.weight.grad.mean())
            #print(model.conv3_dsn4.weight.grad.mean())
            #print(model.conv3_dsn5.weight.grad.mean())
            #print(model.conv3_dsn6.weight.grad.mean())
            #print(model.conv3_dsn5.weight)
            #print(F.sigmoid(out[6]))
            #print(F.sigmoid(out[5]))
            optimizer.step()
            #print(optimizer.state_dict())
            lr_ = lr_base*(gamma**(np.floor(iter_/stepsize)))
            optimizer = torch.optim.SGD([{'params':get_parameter(model,bias=False),'lr':lr_},
                         {'params':get_parameter(model,bias=True),'lr':2*lr_,'weight_decay':0},                                 {'params':get_parameter(model,bias=False,scale=True),'lr':0.1*lr_},
                         {'params':get_parameter(model,scale=True),'lr':0.2*lr_,'weight_decay':0}],
                             lr=lr_,
                             momentum=0.90,
                             weight_decay=0.0005)
            optimizer.zero_grad()
            print('epoch:',epoch,'iter:',iter_,'loss:',loss.data[0],'lr',lr_)

        if iter_%2000==0:
            print('test:')
            #print(F.sigmoid(out[5]))
            torch.save(model.state_dict(),f'result/9_10{lr_base}_{iter_}.pth')
            #evaluate(model,testloader)
        if iter_>max_iter:
            break
    if iter_>max_iter:
        break


