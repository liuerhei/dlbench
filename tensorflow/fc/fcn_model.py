import tensorflow as tf

features_dim = 32 * 32 * 8
image_dim = 28 * 28
label_dim = 10

def get_variable(name, shape, is_bias=False):
        #return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(-0.5, 0.5))
        if is_bias:
                #return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(0.1, 0.5))
                return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))
                #return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

#def get_variable(name, shape, is_bias=False):
#        if is_bias:
#                return tf.get_variable(name, shape, initializer=tf.zeros(shape))
#                #return tf.get_variable(name, shape, initializer=tf.random_uniform(-0.5, 0.5))
#        return tf.get_variable(name, shape, initializer=tf.ones(shape))

def sigmoid_layer(layer_index, x, input_dim, output_dim):
        W = get_variable("W" + str(layer_index), [input_dim, output_dim])
        B = get_variable("B" + str(layer_index), [output_dim], is_bias=True)
        return tf.nn.sigmoid(tf.nn.xw_plus_b(x, W, B))

def model_fcn5(images):
        HL0 = sigmoid_layer(0, images, image_dim, 2048)
        HL1 = sigmoid_layer(1, HL0, 2048, 4096)
        HL2 = sigmoid_layer(2, HL1, 4096, 1024)

        FinalLayerW = get_variable("W5", [1024, label_dim])
        FinalLayerB = get_variable("B5", [label_dim], is_bias=True)
        #FinalLayer = tf.nn.softmax(tf.nn.xw_plus_b(HL2, FinalLayerW, FinalLayerB))
        FinalLayer = tf.nn.xw_plus_b(HL2, FinalLayerW, FinalLayerB)
        return FinalLayer

def model_fcn8(features):
        HL_dim = 2048
        HL0 = sigmoid_layer(0, features, features_dim, HL_dim)
        HL1 = sigmoid_layer(1, HL0, HL_dim, HL_dim)
        HL2 = sigmoid_layer(2, HL1, HL_dim, HL_dim)
        HL3 = sigmoid_layer(3, HL2, HL_dim, HL_dim)
        HL4 = sigmoid_layer(4, HL3, HL_dim, HL_dim)
        HL5 = sigmoid_layer(5, HL4, HL_dim, HL_dim)

        FinalLayerW = get_variable("W8", [HL_dim, 10])
        FinalLayerB = get_variable("B8", 10)
        FinalLayer = tf.nn.xw_plus_b(HL5, FinalLayerW, FinalLayerB)
        return FinalLayer

def loss(logits, labels):
        labels = tf.cast(labels, tf.float32)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        return loss

def train(images, labels, batch_size):
        logits = model_fcn5(images)
        loss_value = loss(logits, labels)
        lr = 0.5
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss_value, global_step=global_step)
        return optimizer
