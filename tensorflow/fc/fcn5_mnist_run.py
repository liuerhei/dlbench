import tensorflow as tf
import fcn_model as models
import time
import os
import numpy as np
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1024, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('epochs', 40, """Max epochs for training.""")
tf.app.flags.DEFINE_integer('log_step', 10, """Log step""")
tf.app.flags.DEFINE_integer('eval_step', 1, """Evaluate step of epoch""")
tf.app.flags.DEFINE_integer('device_id', 0, """Device id.""")
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/csshshi/data/tensorflow/MNIST_data/',
tf.app.flags.DEFINE_string('data_dir', os.environ['HOME']+'/data/tensorflow/MNIST_data/',
#tf.app.flags.DEFINE_string('data_dir', '/home/comp/pengfeixu/Data/tensorflow/MNIST_data/',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """How many GPUs to use.""")

EPOCH_SIZE = 60000
TEST_SIZE = 10000

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

def train(model='fcn5'):
    with tf.Graph().as_default(), tf.Session() as sess:
        #feature_dim = models.feature_dim        #784
        #label_dim = models.label_dim            #10
        label_dim = models.label_dim            #10
        feature_dim = models.image_dim        #784
        images = tf.placeholder(tf.float32, [None, feature_dim])   #图片数据
        labels = tf.placeholder(tf.float32, [None, label_dim])     #用于保存test数据标签

        logits = None                                   
        if model == 'fcn5':    
            logits = models.model_fcn5(images)
        else:
            logits = models.model_fcn8(images)
        #建立网络模型 logits就是计算之后的结果，[60000,10]
        loss = models.loss(logits, labels)    
        #相当于mnist里的cross_entropy,交叉熵

        predictionCorrectness = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(predictionCorrectness, "float"))
        #将计算结果与标签进行比对，计算正确率

        lr = 0.05
        #optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        #batch_size_per_epoch = int((EPOCH_SIZE + FLAGS.batch_size - 1)/ FLAGS.batch_size)
        #iterations = FLAGS.epochs * batch_size_per_epoch 
        iterations = 2360
        for step in range(iterations):
            start_time = time.time()
            imgs, labs = mnist.train.next_batch(FLAGS.batch_size)
            _, loss_value = sess.run([optimizer, loss], feed_dict={images:imgs,labels:labs})
            if step % FLAGS.log_step == 0:
                print ("time %s, step %d, loss = %.4f" % (datetime.now(), step, loss_value))
            if step > 0 and step % 59 == 0:
                accuracy_value = accuracy.eval(feed_dict={images: mnist.test.images, labels: mnist.test.labels})
                print("test accuracy %g"%accuracy_value)


def main(argv=None):
    train(model='fcn5')


if __name__ == '__main__':
    tf.app.run()
