# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from numpy import array

import tensorflow as tf
import numpy as np

FLAGS = None
SLOT_COUNT = 7 * 4
FEATURE_COUNT = 33
INPUT_SIZE = SLOT_COUNT * FEATURE_COUNT
OUTPUT_SIZE = SLOT_COUNT
NORMALIZE_FACTOR = 600
PATH = '/Users/louisliu/vagrant/github/kkv_data_game/'
OUTPUT_FILENAME = PATH + 'out.ans'

ans_header = [
    'user_id',
    'time_slot_0', 'time_slot_1', 'time_slot_2', 'time_slot_3',
    'time_slot_4', 'time_slot_5', 'time_slot_6', 'time_slot_7',
    'time_slot_8', 'time_slot_9', 'time_slot_10', 'time_slot_11',
    'time_slot_12', 'time_slot_13', 'time_slot_14', 'time_slot_15',
    'time_slot_16', 'time_slot_17', 'time_slot_18', 'time_slot_19',
    'time_slot_20', 'time_slot_21', 'time_slot_22', 'time_slot_23',
    'time_slot_24', 'time_slot_25', 'time_slot_26', 'time_slot_27'
]


def write_out_answer(output_filename, y_data):
    assert(len(y_data[0]) == SLOT_COUNT)
    f = open(output_filename, 'w')

    for item in ans_header[0:-1]:
        f.write('%s,' % item)
    f.write('%s\n' % ans_header[-1])

    entry_count = len(y_data)
    start_index = 57159
    print(entry_count)
    for i in range(entry_count):
        user_id = start_index + i
        f.write('%d,' % user_id)
        for j in range(SLOT_COUNT - 1):
            if y_data[i][j] >= NORMALIZE_FACTOR:
                print(y_data[i][j])
            assert(y_data[i][j] <= NORMALIZE_FACTOR)
            f.write('%f,' % (y_data[i][j] / NORMALIZE_FACTOR))
        if y_data[i][-1] >= NORMALIZE_FACTOR:
            print(y_data[i][-1])
        assert(y_data[i][-1] <= NORMALIZE_FACTOR)
        f.write('%f\n' % (y_data[i][-1] / NORMALIZE_FACTOR))

    f.close()


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # load my data
  print('loading train data...')
  x_data_train = np.loadtxt('x_data_train_1_36.txt')
  x_data_train = array(x_data_train).T
  y_data_train = np.loadtxt('y_data_train.txt')
  print('loading test data...')
  x_data_test = np.loadtxt('x_data_test.txt')
  x_data_test = array(x_data_test).T

  # Create the model
  x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
  W = tf.Variable(tf.zeros([INPUT_SIZE, OUTPUT_SIZE]))
  b = tf.Variable(tf.zeros([OUTPUT_SIZE]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  PER_SIZE = 45
  for _ in range(1000):
    # batch_xs, batch_ys = mnist.train.next_batch(1)
    batch_xs, batch_ys = x_data_train[_*PER_SIZE:(_+1)*PER_SIZE], y_data_train[_*PER_SIZE:(_+1)*PER_SIZE]
    # batch_xs = [[0. for i in range(INPUT_SIZE)]]
    # batch_ys = [[0. for i in range(OUTPUT_SIZE)]]
    # print(batch_xs.shape)
    # print(batch_xs)
    # print(batch_ys)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
  #                                  y_: mnist.test.labels}))

  y_data_test = sess.run(y, feed_dict={x: x_data_test[0:]})
  write_out_answer(OUTPUT_FILENAME, y_data_test)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
