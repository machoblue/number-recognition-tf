import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('output_graph', '../model.pb', '''File name of ouptut graph def.''')
tf.app.flags.DEFINE_integer('batch_size', 600,
                            '''Batch size for training.''')
tf.app.flags.DEFINE_integer('training_steps', 6000,
                            '''How many training steps to run.''')

train_size = x_train.shape[0]

def inference(x, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        x = tf.identity(x)
        w0 = tf.Variable(tf.truncated_normal([784, 50]))
        b0 = tf.Variable(tf.zeros([50]))
        
        z = tf.nn.relu(tf.matmul(x, w0) + b0)
        
        w1 = tf.Variable(tf.zeros([50, 10])) # truncated_normalにすると、accuracyが上がらない
        b1 = tf.Variable(tf.zeros([10]))
        
        y = tf.matmul(z, w1) + b1
    return tf.identity(y, name='inference_1') # sugyanのスクリプトではinference

def main(argv=None):
    batch_mask = np.random.choice(train_size, FLAGS.batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    f = inference(x)
    p = tf.nn.softmax(f)

    t = tf.placeholder(tf.float32, [None, 10])
    loss = -tf.reduce_sum(t * tf.log(p))
    train_step = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#    sess = tf.InteractiveSession()
#    sess.run(tf.initialize_all_variables()) # deprecated
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        i = 0
        for _ in range(FLAGS.training_steps):
            i += 1
            sess.run(train_step, feed_dict={x: x_batch, t: t_batch})
            if i % 100 == 0:
                loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:x_test, t: t_test})
                print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
        
        # 学習結果保存
        output = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['inference_1'])
        with open(FLAGS.output_graph, 'wb') as f:
            f.write(output.SerializeToString())

if __name__ == '__main__':
    tf.app.run()