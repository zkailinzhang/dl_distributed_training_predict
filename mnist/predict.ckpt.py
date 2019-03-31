
# coding: utf-8


import os
import tensorflow as tf
import struct 
import numpy as np

from PIL import Image
#import matplotlib.pyplot as plt
tf.enable_eager_execution()



test_images = './mnist1/'

os.system('gunzip -c {} > {}'.format(test_images+ 't10k-images-idx3-ubyte.gz',test_images +'t10k-images-idx3-ubyte'))
os.system('gunzip -c {} > {}'.format(test_images + 't10k-labels-idx1-ubyte.gz',test_images +'t10k-labels-idx1-ubyte'))






import struct

def load_mnist_train(path, kind='train'):    
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_mnist_test(images_path,labels_path):

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels 




test_images_path= os.path.join(test_images,'t10k-images-idx3-ubyte')
test_labels_path= os.path.join(test_images, 't10k-labels-idx1-ubyte')




test_images=[]
test_labels=[]
test_images,test_labels=load_mnist_test(test_images_path,test_labels_path)
print(np.shape(test_images),np.shape(test_labels))



#reshape 和加一唯独有什么区别
img_random = test_images[0,:].reshape(1,784)
img_random =img_random.astype(np.float32)
print(img_random.dtype,np.shape(img_random))
#plt.imshow(img_random,cmap=plt.cm.gray)
#plt.title(test_labels[0])
#plt.Text(0.5,1,'7')


# 模型路径存在虚拟主机路径下： /outputs/root/quick-start1/experiments/5
# 


modelpathpre = '/outputs/root//'
experiment  = 'quick-start1/'
exp_num = 'experiments/5/'

modelpath = modelpathpre + experiment  + exp_num
print(modelpath)

modelpath = './modelp/'
# #### 重点来了，加载模型，读取模型


#经典的加载

#

with tf.Session() as sess:
    #加载网络图
    saver = tf.train.import_meta_graph(modelpath+'model.ckpt-430.meta')
    #加载参数
    saver.restore(sess,tf.train.latest_checkpoint(modelpath))
    #tf.summary.
    #input_img = {'images:0':img_random }

    img_random2 = img_random.reshape([1,28,28,1])
    
    input_img = {'network/Reshape:0':img_random2 }

    graph = tf.get_default_graph()
    #print(graph.get_all_collection_keys())
    #print([n.name for n in graph.as_graph_def(). node])
    #这个竟然ok
    #ye = graph.get_tensor_by_name('network/dense/bias/Adam_1:0') #加：0 是tensor
    
    ye = graph.get_tensor_by_name('network/dense_1/bias/Adam_1:0') #加：0 是tensor
    
    #ye = graph.get_operation_by_name('network/dense_1/bais')  # 不加 op
    
    sess.run([ye],feed_dict=input_img)
    
    print(ye)
    aa = tf.argmax(ye, axis=1)
    print(aa)
    #print(sess.run(aa))
    tf.estimator.EstimatorSpec(mode= tf.estimator.ModeKeys.PREDICT, predictions=ye)
    print(ye)
    bb = sess.run(ye)
    type(bb)
    print('{}, zuida: {}'.format(bb,np.max(bb.val(sess),axis=0)))

'''
To construct input pipelines, use the `tf.data` module.
Tensor("network/dense_1/bias/Adam_1:0", shape=(10,), dtype=float32_ref)
Tensor("ArgMax_1:0", shape=(), dtype=int64)
Tensor("network/dense_1/bias/Adam_1:0", shape=(10,), dtype=float32_ref)
[1.6545158e-05 3.7704060e-05 6.4401545e-05 4.7269867e-05 2.3644099e-05
 2.8983388e-05 2.3744109e-05 3.1365456e-05 5.8853399e-05 4.8405072e-05]
zuida: Tensor("network/dense_1/bias/Adam_1:0", shape=(10,), dtype=float32_ref)
'''

