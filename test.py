import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from DSSNet import DSSNet
from dataset import MSRA10K
from dataset import ECSSD
import pydensecrf.densecrf as dcrf
import time
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
thresholds=256
E=1e-4
EPSION=1e-8
def untransform(img):
    #input :img :n*c*h*w
    img = img[0]
    img = img.transpose(1,2,0)
    img += np.array([104.00699,116.66877,122.67892])
    img = img[:,:,::-1]
    return img.astype(np.uint8)

def mapminmax(sal,a=0,b=255):
    max_ = np.max(np.max(sal))
    min_ = np.min(np.min(sal))
    mul = (b-a)/(max_-min_)
    sal = mul*(sal-min_)+a
    return sal
def compute_pr(sal,lbl,pre,rec):
    #lbl :1 :saliency 0:background
    l = lbl>0
    for i in range(thresholds):
        p = sal>i
        ab =p[l].sum()
        a = p.sum()
        b = l.sum()
        pre[i] += (ab+E)/(a+E)
        rec[i] += (ab+E)/(b+E)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def crf_compute(img,ann):
    tau = 1.05
    img = img.copy(order='C')
    d = dcrf.DenseCRF2D(img.shape[1],img.shape[0],2)
    U = np.zeros((2,img.shape[0]*img.shape[1]),dtype=np.float32)
    U[1,:] = (-np.log(ann+EPSION)/(tau*sigmoid(ann))).flatten()
    U[0,:] = (-np.log(1-ann+EPSION)/(tau*sigmoid(1-ann))).flatten()
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3,compat=3)
    d.addPairwiseBilateral(sxy=60,srgb=5,rgbim=img,compat=5)
    infer = np.array(d.inference(5)).astype(np.float32)
    return infer[1].reshape(img.shape[:2])


def evaluate(model,dataset):
    #model.conv_fuse.weight.data[0,0,0,0]=0
    #model.conv_fuse.weight.data[0,4,0,0]=0
    #model.conv_fuse.weight.data[0,5,0,0]=0
    model.eval()
    save_image=False
    #sal_list  = []
    #lbl_list = []
    pre = np.zeros(thresholds)
    rec = np.zeros(thresholds)
    time1=[]
    time2=[]
    for iter_,(data,label) in enumerate(dataset):
        time_1 = time.time()
        img = data.numpy()
        img = untransform(img)
        data = Variable(data).cuda()
        out = model(data)
        fin = F.sigmoid(out[6])
        for i in [2,3,4]:
            fin += F.sigmoid(out[i])
        #fin = fin.data[0][0].cpu().numpy()
        label = label[0].numpy()
        fin = (fin/4).data[0][0].cpu().numpy()
        time_2 = time.time()
        time1.append(time_2-time_1)
        fin_before = (fin*255).astype(np.uint8)
        fin=crf_compute(img,fin)
        fin *= 255
        time_3 = time.time()
        time2.append(time_3-time_2)
        #fin = mapminmax(fin)
        fin = fin.astype(np.int32)
        label = label.astype(np.int32)
        compute_pr(fin,label,pre,rec)
        if save_image:
            fin  = fin.astype(np.uint8)
            label = label.astype(np.uint8)
            #sal = Image.fromarray(fin)
            #sal_name = f'./result/img1/sal_{iter_}.png'
            #sal.save(sal_name)
            label[label==1]=255
            #lbl = Image.fromarray(label)
            #lbl_name = f'./result/img1/lbl_{iter_}.png'
            #lbl.save(lbl_name)
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(fin,cmap = cm.Greys_r)
            plt.subplot(2,2,2)
            plt.imshow(label,cmap=cm.Greys_r)
            plt.subplot(2,2,3)
            plt.imshow(fin_before,cmap=cm.Greys_r)
            plt.subplot(2,2,4)
            plt.imshow(img)
            plt.savefig(f'./result/img2/result_{iter_}.png')
            plt.close()
    pre = pre/len(dataset)
    rec = rec/len(dataset)
    print('F:',max(pre*rec*(1+0.3)/(0.3*pre+rec)))
    print(np.array(time1).mean())
    print(np.array(time2).mean())
    with open('./result/pr.txt','a+') as f:
        f.write('precision:\n')
        f.write(str(pre)+'\n')
        f.write('recall:\n')
        f.write(str(rec)+'\n')
                

def main():
    model = DSSNet()
    for i in range(1):
        ex = 10000
        print(ex)
        model.load_state_dict(torch.load('converted.pth'))
        model.cuda()
        root = '/home/wfz/Documents/code/saliency/MSRA10K_Imgs_GT' 
        testdata = MSRA10K(root=root,split='val',transform=True)
        testloader = DataLoader(testdata,batch_size=1)
        evaluate(model,testloader)
    
if __name__=='__main__':
    main()


        
    
