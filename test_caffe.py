import sys
import test
import numpy as np
print('1')
caffe_root  = '../caffe_dss-master/'
code_root  = '../caffe_dss-master/examples/dss_final/'
sys.path.insert(0,caffe_root+'python')
import caffe
print('2')
import os 
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
print('3')
caffe.set_mode_gpu()
net = caffe.Net(code_root+'train_val.prototxt',code_root+'snapshot/dss_iter_12000.caffemodel',caffe.TEST)
pre = np.zeros(256)
rec = np.zeros(256)
for i in range(2000):
    net.forward()
    out=net.blobs['sigmoid-fuse'].data
    for j in range(3):
        out += net.blobs[f'sigmoid-dsn{j+2}'].data
    fin = out/4
    fin = fin[0][0]
    img = test.untransform(net.blobs['data'].data)
    fin = test.crf_compute(img,fin)
    fin *=255
    test.compute_pr(fin,net.blobs['label'].data[0][0],pre,rec)
pre = pre/2000
rec = rec/2000
print('F:',max(pre*rec*(1+0.3)/(0.3*pre+rec)))

