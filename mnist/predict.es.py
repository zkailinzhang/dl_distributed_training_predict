
# coding: utf-8


import os
import tensorflow as tf
import struct 
import numpy as np

from PIL import Image
#import matplotlib.pyplot as plt




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

modelpath = './modelp'
# #### 重点来了，加载模型，读取模型


#经典的加载

#
'''
with tf.Session() as sess:
    #加载网络图
    saver = tf.train.import_meta_graph(modelpath+'model.ckpt-430.meta')
    #加载参数
    saver.restore(sess,tf.train.latest_checkpoint(modelpath))
    
    input_img = {'images':img_random }
    graph = tf.get_default_graph()
    print(graph.get_all_collection_keys())
    print([n.name for n in graph.as_graph_def(). node])
    y_pre = graph.get_tensor_by_name("y:0")
    
    sess.run(y_pre,feed_dict=input_img)
    
    print(y_pre)
'''    



#用estimator加载
mode = tf.estimator.ModeKeys.PREDICT


def get_model_fn(dropout):
    """Create a `model_fn` compatible with tensorflow estimator based on hyperparams."""

    def get_network(x_dict, is_training):
        with tf.variable_scope('network'):
            x = x_dict['images']
            x = tf.reshape(x, shape=[-1, 28, 28, 1])
            conv1 = tf.layers.conv2d(x, 32, 5, activation='relu')
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation='relu')
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            fc1 = tf.contrib.layers.flatten(conv2)
            fc1 = tf.layers.dense(fc1, 1024)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
            out = tf.layers.dense(fc1, 10)
        return out

    def model_fn(features, labels, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        #is_training = 0
        results = get_network(features, is_training=is_training)

        predictions = tf.argmax(results, axis=1)

        # Return prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

                # Define loss
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=results, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluation metrics
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        precision = tf.metrics.precision(labels=labels, predictions=predictions)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': accuracy, 'precision': precision})

    return model_fn



estimator = tf.estimator.Estimator(get_model_fn(dropout=0.),model_dir=modelpath)
#estimator = tf.estimator.EstimatorSpec(mode=mode,model_dir=modelpath)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': img_random},
    y=None,
    batch_size=1,
    shuffle=False
)
 
#返回的是个迭代器
#estimator.predict(input_fn=input_fn,predict_keys=predicts,checkpoint_path=modelpath)
for predicts in estimator.predict(input_fn=input_fn):
    print(predicts)