import sys 
import torch
from DSSNet import DSSNet
from PIL import Image
import numpy as np
from torch.autograd import Variable
root = '/home/wfz/Documents/code/saliency/'
caffe_root = root +'caffe_dss-master/python'
sys.path.insert(0,caffe_root)
import caffe
from collections import OrderedDict
def convert(caf,tor):
    params_caf = []
    for i in caf.params.keys():
        params_caf.append(caf.params[i][0].data)
        if not caf.params[i][1].data.all():
            continue
            print('bia_name:',i)
        params_caf.append(caf.params[i][1].data)
    i=0
    for n,m in tor.named_parameters():
        tem = torch.from_numpy(params_caf[i])
        #print(tem)
        #print(m.data)
        i+=1
        assert tem.size()==m.data.size(),f"problem{n}"
        m.data.copy_(tem)

    return tor


caf_root = root+'caffe_dss-master/examples/dss_final'
caffe.set_mode_cpu()
net = caffe.Net(caf_root+'/deploy.prototxt','dss_model_released.caffemodel',caffe.TEST)
net.forward()
model = DSSNet()
c=convert(net,model)
img = Image.open('100061.jpg')
img = np.array(img)
img = img.astype(np.float32)
img = img[:,:,::-1]
img -= np.array((104.00698793,116.66876762,122.67891643))
img = img.transpose((2,0,1))
net.blobs['data'].reshape(1,*img.shape)
net.blobs['data'].data[...]=img
net.forward()
o = []
def hook(module,input,output):
    o.append(input[0].data.numpy())
def dist_(caffe_tensor,th_tensor):
    t = th_tensor[0]
    c = caffe_tensor[0]
    if t.shape != c.shape:
        print ("t.shape",t.shape)
        print ("c.shape",c.shape)
    d = np.linalg.norm(t-c)
    print("d",d)

c.conv1_1.register_forward_hook(hook)
c.relu1_1.register_forward_hook(hook)
c.conv2_1.register_forward_hook(hook)
c.conv3_1.register_forward_hook(hook)
c.conv4_1.register_forward_hook(hook)
c.conv5_1.register_forward_hook(hook)
c.conv1_dsn6.register_forward_hook(hook)
c.conv1_dsn5.register_forward_hook(hook)
c.conv1_dsn4.register_forward_hook(hook)
c.conv1_dsn3.register_forward_hook(hook)
c.conv1_dsn2.register_forward_hook(hook)
c.conv1_dsn1.register_forward_hook(hook)
out = c(Variable(torch.FloatTensor(img.copy()).unsqueeze(0)))
dist_(net.blobs['data'].data,o[0])
dist_(net.blobs['conv1_1'].data,o[1])
dist_(net.blobs['pool1'].data,o[2])
dist_(net.blobs['pool2'].data,o[3])
dist_(net.blobs['pool3'].data,o[4])
dist_(net.blobs['pool4'].data,o[5])
dist_(net.blobs['pool5a'].data,o[6])
dist_(net.blobs['conv5_3'].data,o[7])
dist_(net.blobs['conv4_3'].data,o[8])
dist_(net.blobs['conv3_3'].data,o[9])
dist_(net.blobs['conv2_2'].data,o[10])
dist_(net.blobs['conv1_2'].data,o[11])



#torch.save(c.state_dict(),'converted2.pth')




