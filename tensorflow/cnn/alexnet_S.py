from datetime import datetime
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 24, """batch size""")
tf.app.flags.DEFINE_integer('batch_num', 500, """batch number""")
tf.app.flags.DEFINE_boolean('forword_only', False, """only run forword pass""")
tf.app.flags.DEFINE_boolean('forword_backword_only', False, """only run forword-backword pass""")
tf.app.flags.DEFINE_string('data_format', 'NHWC', """NHWC or NCHW""")

conv_counter = 1
pool_counter = 1
affine_counter = 1
parameters = []

def _conv(inputOP, in_ch, out_ch, kh, kw, strh, strw, padType):
#(inputOP, in_channel, out_channel, ker_heigth, ker_width, strides_height, strides_width, padType):
        global conv_counter
        global pool_counter
        global parameters
        name = 'conv' + str(conv_counter)
        conv_counter += 1
        with tf.name_scope(name) as scope:
                kernal = tf.Variable(tf.truncated_normal([kh, kw, in_ch, out_ch], dtype=tf.float32, stddev=1e-1), name='weight')
                if FLAGS.data_format == 'NCHW':
                        strides = [1, 1, strh, strw]
                else:
                        strides = [1, strh, strw, 1]
                conv = tf.nn.conv2d(inputOP, kernal, strides, padding=padType, data_format=FLAGS.data_format)
                biases = tf.Variable(tf.constant(0.0, shape=[out_ch], dtype=tf.float32), trainable=True, name='biases')
                bias = tf.nn.bias_add(conv, biases, data_format=FLAGS.data_format)
                conv1 = tf.nn.relu(bias, name=scope)
                parameters += [kernal, biases]
                return conv1

def _mpool(inputOP, kh, kw, strh, strw):
        global pool_counter
        global parameters
        name = 'pool' + str(pool_counter)
        pool_counter += 1
        if FLAGS.data_format == 'NCHW':
                strides = [1, 1, strh, strw]
                ksize = [1, 1, kh, kw]
        else:
                ksize = [1, kh, kw, 1]
                strides = [1, strh, strw, 1]
        return tf.nn.max_pool(inputOP, 
                              ksize=ksize, 
                              strides=strides, 
                              padding='VALID', 
                              data_format=FLAGS.data_format,
                              name=name)

def _affine(inputOP, in_ch, out_ch):
        global affine_counter
        global parameters
        name = 'affine' + str(affine_counter)
        affine_counter += 1
        with tf.name_scope(name) as scope:
                kernal = tf.Variable(tf.truncated_normal([in_ch, out_ch],
                                                         dtype=tf.float32,
                                                         stddev=1e-1),
                                                         name='weights')
                biases = tf.Variable(tf.constant(0.0, shape=[out_ch], dtype=tf.float32), trainable=True, name='biases')
                affine1 = tf.nn.relu_layer(inputOP, kernal, biases, name=name)
                parameters += [kernal, biases]
                return affine1

def loss(logits, labels):
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated = tf.constant([indices, labels], 1)
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 1000]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels,
                                                                name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss
def inference(images):
        conv1 = _conv(images, 3, 96, 11, 11, 4, 4, 'VALID')
        #print(conv1.get_shape())
        pool1 = _mpool(conv1, 3, 3, 2, 2)
        #print(pool1.get_shape())
        #conv2 = _conv(pool1, 96, 256, 5, 5, 1, 1, 'SAME')
        conv2 = tf.layers.conv2d(pool1, 256, 5, padding='SAME')
        #print(conv2.get_shape())
        pool2 = _mpool(conv2, 3, 3, 2, 2, )
        #print(pool2.get_shape())

        conv3 = _conv(pool2, 256, 384, 3, 3, 1, 1, 'SAME')
        conv4 = _conv(conv3, 384, 256, 3, 3, 1, 1, 'SAME')
        conv5 = _conv(conv4, 256, 256, 3, 3, 1, 1, 'SAME')
        pool5 = _mpool(conv5, 3, 3, 2, 2)
        resh1 = tf.reshape(pool5, [-1, 256*6*6])
        aff1 = _affine(resh1, 256*6*6, 4096)
        aff2 = _affine(aff1, 4096, 4096)
        aff3 = _affine(aff2, 4096, 1000)
        return aff3

def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  if not isinstance(target, list):
    target = [target]
  target_op = tf.group(*target)
  for i in range(FLAGS.batch_num + num_steps_burn_in):
  #for i in range(1):
    start_time = time.time()
    _ = session.run(target_op)
    duration = time.time() - start_time
    if i > num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
                        (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))

def run_benchmark():
  global parameters
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    if FLAGS.data_format == 'NCHW':
      image_shape = [FLAGS.batch_size, 3, image_size + 3, image_size + 3]
    else:
      image_shape = [FLAGS.batch_size, image_size + 3, image_size + 3, 3]
    images = tf.Variable(tf.random_normal(image_shape,
                                          dtype=tf.float32,
                                          stddev=1e-1))

    labels = tf.Variable(tf.ones([FLAGS.batch_size],
                                 dtype=tf.int32))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    last_layer = inference(images)
    #shape = [-1, 1000]
    # Build an initialization operation.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session('')
    sess.run(init)

    run_forword = True
    run_forword_backword = True
    if FLAGS.forword_only and FLAGS.forword_backword_only:
      raise ValueError("Cannot specify --forword_only and "
                       "--forword_backword_only at the same time.")
    if FLAGS.forword_only:
      run_forword_backword = False
    elif FLAGS.forword_backword_only:
      run_forword = False

    if run_forword:
      # Run the forword benchmark.
      print("I am here")
      time_tensorflow_run(sess, last_layer, "Forward")

    if run_forword_backword:
      # Add a simple objective so we can calculate the backword pass.
      objective = loss(last_layer, labels)
      # Compute the gradient with respect to all the parameters.
      grad = tf.gradients(objective, parameters)
      # Run the backword benchmark.
      time_tensorflow_run(sess, grad, "Forward-backword")


def main(_):
  program_start_time = time.time()
  run_benchmark()
  program_end_time = time.time()
  print('Program finished, Total seconds: %s' % (program_end_time - program_start_time))


if __name__ == '__main__':
  tf.app.run()









                
