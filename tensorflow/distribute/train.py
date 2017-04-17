import tensorflow as tf
import time
import input_data 

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

FLAGS = tf.app.flags.FLAGS
# define cluster
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222",
#tf.app.flags.DEFINE_string("ps_hosts", "114.214.166.246:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223",
#tf.app.flags.DEFINE_string("worker_hosts", "114.214.166.247:2222",
                           "Comma-separated list of hostname:port pairs")

# define server
tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

def main(_):
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")
        
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

        server = tf.train.Server(cluster,
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index)

        if FLAGS.job_name == "ps":
                server.join()
        elif FLAGS.job_name == "worker":
                with tf.device(tf.train.replica_device_setter(
                        worker_device="/job:worker/task:%d" %FLAGS.task_index, 
                        #worker_device="/job:worker/task:0",
                        cluster=cluster)):
                        hid_w = tf.Variable(tf.truncated_normal([784,128], stddev=1.0 / 28), name='hid_w')
                        hid_b = tf.Variable(tf.ones([128]), name='hid_b')

                        sm_w = tf.Variable(tf.truncated_normal([128, 10], stddev=1.0 / 8), name='sm_w')
                        sm_b = tf.Variable(tf.ones([10]), name='sm_b')

                        images = tf.placeholder(tf.float32, [None, 784], name='images')
                        labels = tf.placeholder(tf.float32, [None, 10], name='labels')

                        hid_layer = tf.nn.relu(tf.nn.xw_plus_b(images, hid_w, hid_b))
                        y = tf.nn.softmax(tf.nn.xw_plus_b(hid_layer, sm_w, sm_b))

                        #global_step = 0
                        #loss = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(y, 1e-1, 1.0)))
                        loss = -tf.reduce_sum(labels * tf.log(y))

                        #train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
                        train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

                        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                        tf.summary.scalar("cost", loss)
                        tf.summary.scalar("accuracy", accuracy)
                        init = tf.global_variables_initializer()
                        saver = tf.train.Saver()
                        summarlabelsop = tf.summary.merge_all()

                sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                         logdir="/public/home/jnliu/Dl-Tensorflow/distribute/train_logs",
                                         init_op=init,
                                         summary_op=summarlabelsop,
                                         saver=saver,
                                         #global_step=global_step,
                                         save_model_secs=10)
                with sv.managed_session(server.target) as sess:
                        step = 0
                        while not sv.should_stop() and step < 100:
                                imgs, labs = mnist.train.next_batch(100)
                                #_, summary, loss, step = sess.run([train_op, summarlabelsop, loss''', global_step'''] , feed_dict={x:batch_xs, y_:batch_ys})
                                _, summary, loss = sess.run([train_op, summarlabelsop, loss], feed_dict={images:imgs, labels:labs})
                                print("{0} step: {1}, loss = {2}".format(time.time(), step, loss))
                
                sv.stop()

if __name__ == "__main__":
        tf.app.run()




