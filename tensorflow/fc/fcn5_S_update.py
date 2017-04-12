import tensorflow as tf
import fcn_model as models
import time
from datetime import datetime
import numpy as np
import os

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1024, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('log_step', 10, """Log step""")

def createFakeData(count, feature_dim, label_dim):
        features = np.random.randn(count, models.features_dim)
        label = np.zeros((count, label_dim))
        for i in range(count):
                j = np.random.randint(0, 9, size = 1)
                label[i, j] = 1
        return features, label

features, label = createFakeData(60000, 8096, 10)

def getFakeData(batch_size):
        k = np.random.randint(0, 58977)
        feat = features[k:k+1023]
        lab = label[k:k+1023]
        #feat = features[1:1024]
        #lab = label[1:1024]
        return feat, lab
#def getFakeData(count, feature_dim, label_dim):
#        features = np.random.randn(count, feature_dim)
#        label = np.zeros((count, label_dim))
#        for i in range(count):
#                j = np.random.randint(0, 9, size = 1)
#                label[i, j] = 1
#        return features, label

def train(model='fcn8'):
    #num_threads = 32 
    num_threads = os.getenv('OMP_NUM_THREADS', 1)
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=int(num_threads))
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        label_dim = models.label_dim            #10
        features_dim = models.features_dim       #32 * 32 * 8
        images = tf.placeholder(tf.float32, [None, features_dim])   
        labels = tf.placeholder(tf.float32, [None, 10])    

        logits = models.model_fcn8(images)
        loss = models.loss(logits, labels)    

        lr = 0.05
        #optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss)

        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        iterations = 2360
        avg_batch_time = 0.0
        for step in range(iterations):
            #feat, lab = getFakeData(1024, features_dim, label_dim)
            feat, lab = getFakeData(1024)
            start_time = time.time()
            _, loss_value = sess.run([optimizer, loss], feed_dict={images:feat, labels:lab})
            sec_per_batch = time.time() - start_time
            avg_batch_time += float(sec_per_batch)
            if step % FLAGS.log_step == 0:
                example_per_sec = FLAGS.batch_size / sec_per_batch
                strfor = ('%s, step %4d, loss = %4.2f (%6.1f example/sec; %4.3f sec/batch)')
                print (strfor % (datetime.now(), step, loss_value,example_per_sec, float(sec_per_batch)))
        avg_batch_time /= iterations
        print("the average batch time is ", avg_batch_time)

def main(argv=None):
    train(model='fcn8')


if __name__ == '__main__':
    tf.app.run()
