from __future__ import absolute_import, division, print_function

import argparse
import tensorflow as tf
#from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths, get_outputs_path
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def set_logging(log_level=None):
    if log_level == 'INFO':
        log_level = tf.logging.INFO
    elif log_level == 'DEBUG':
        log_level = tf.logging.DEBUG
    elif log_level == 'WARN':
        log_level = tf.logging.WARN
    else:
        log_level = 'INFO'

    tf.logging.set_verbosity(log_level)
 
set_logging('DEBUG')


data_paths = "./mnist1"
mnist = input_data.read_data_sets(data_paths, one_hot=False)
#(55000, 784) float32
print(np.shape(mnist.train.images),mnist.train.images.dtype)

def get_model_fn(learning_rate, dropout, activation):
    """Create a `model_fn` compatible with tensorflow estimator based on hyperparams."""

    def get_network(x_dict, is_training):
        with tf.variable_scope('network'):
            x = x_dict['images']
            x = tf.reshape(x, shape=[-1, 28, 28, 1])
            conv1 = tf.layers.conv2d(x, 32, 5, activation=activation)
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=activation)
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            fc1 = tf.contrib.layers.flatten(conv2)
            fc1 = tf.layers.dense(fc1, 1024)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
            out = tf.layers.dense(fc1, 10)
        return out

    def model_fn(features, labels, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        results = get_network(features, is_training=is_training)

        predictions = tf.argmax(results, axis=1)

        # Return prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Define loss
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=results, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        default=128,
        type=int
    )
    parser.add_argument(
        '--num_steps',
        default=800,
        type=int
    )
    parser.add_argument(
        '--num_iterations',
        default=1,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float
    )
    parser.add_argument(
        '--dropout',
        default=0.25,
        type=float
    )
    parser.add_argument(
        '--num_epochs',
        default=1,
        type=int
    )
    parser.add_argument(
        '--activation',
        default='relu',
        type=str
    )
    parser.add_argument(
        '--distributed',
        default=False,
        type=bool
    )

    args = parser.parse_args()
    arguments = args.__dict__

    batch_size = arguments.pop('batch_size')
    num_steps = arguments.pop('num_steps')
    learning_rate = arguments.pop('learning_rate')
    dropout = arguments.pop('dropout')
    num_epochs = arguments.pop('num_epochs')
    activation = arguments.pop('activation')
    distributed = arguments.pop('distributed')
    num_iterations = arguments.pop('num_iterations')
    if activation == 'relu':
        activation = tf.nn.relu
    elif activation == 'sigmoid':
        activation = tf.nn.sigmoid
    elif activation == 'linear':
        activation = None

    
    estimator = tf.estimator.Estimator(
        get_model_fn(learning_rate=learning_rate, dropout=dropout, activation=activation),
        model_dir='./log')

    # Train the Model
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images},
        y=mnist.train.labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=True)

    for i in range(num_iterations):
        estimator.train(input_fn, steps=num_steps)

        # Evaluate the Model
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': mnist.test.images},
            y=mnist.test.labels,
            batch_size=batch_size,
            shuffle=False)

        metrics = estimator.evaluate(input_fn)

        print("Testing metrics: {}", metrics)
        print(loss=metrics['loss'],
                               accuracy=metrics['accuracy'],
                               precision=metrics['precision'])


    #estimator.predict()

'''

(tf12) zkl@zkl-MACH-WX9:~/123/polyaxon-quick-start$ cd /home/zkl/123/polyaxon-quick-start ; env PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1 /home/zkl/anaconda3/envs/tf12/bin/python /home/zkl/.vscode/extensions/ms-python.python-2019.3.6139/pythonFiles/ptvsd_launcher.py --default --client --host localhost --port 44649 /home/zkl/123/polyaxon-quick-start/model_tf.py 
WARNING:tensorflow:From /home/zkl/123/polyaxon-quick-start/model_tf.py:24: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
WARNING:tensorflow:From /home/zkl/.local/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From /home/zkl/.local/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting ./mnist1/train-images-idx3-ubyte.gz
WARNING:tensorflow:From /home/zkl/.local/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting ./mnist1/train-labels-idx1-ubyte.gz
Extracting ./mnist1/t10k-images-idx3-ubyte.gz
Extracting ./mnist1/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:From /home/zkl/.local/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_model_dir': './log', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
graph_options {
  rewrite_options {
    meta_optimizer_iterations: ONE
  }
}
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_service': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f6106ac3f28>, '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
WARNING:tensorflow:From /home/zkl/.local/lib/python3.6/site-packages/tensorflow/python/estimator/inputs/queues/feeding_queue_runner.py:62: QueueRunner.__init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
WARNING:tensorflow:From /home/zkl/.local/lib/python3.6/site-packages/tensorflow/python/estimator/inputs/queues/feeding_functions.py:500: add_queue_runner (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Graph was finalized.
2019-03-27 23:36:17.596771: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
WARNING:tensorflow:From /home/zkl/.local/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py:804: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.
INFO:tensorflow:Saving checkpoints for 0 into ./log/model.ckpt.
INFO:tensorflow:loss = 2.3226657, step = 1
INFO:tensorflow:global_step/sec: 13.9386
INFO:tensorflow:loss = 0.22335765, step = 101 (7.181 sec)
INFO:tensorflow:global_step/sec: 13.4861
INFO:tensorflow:loss = 0.16281345, step = 201 (7.408 sec)
INFO:tensorflow:global_step/sec: 15.3163
INFO:tensorflow:loss = 0.046344183, step = 301 (6.529 sec)
INFO:tensorflow:global_step/sec: 13.3481
INFO:tensorflow:loss = 0.03438069, step = 401 (7.493 sec)
INFO:tensorflow:Saving checkpoints for 430 into ./log/model.ckpt.
INFO:tensorflow:Loss for final step: 0.048366968.
INFO:tensorflow:Calling model_fn.
INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2019-03-27-15:44:52
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from ./log/model.ckpt-430
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2019-03-27-15:44:55
INFO:tensorflow:Saving dict for global step 430: accuracy = 0.9845, global_step = 430, loss = 0.049345534, precision = 0.99922323
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 430: ./log/model.ckpt-430
Testing metrics: {} {'accuracy': 0.9845, 'loss': 0.049345534, 'precision': 0.99922323, 'global_step': 430}

'''