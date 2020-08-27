# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Multi-resolution input data pipeline."""

import os
import re
import numpy as np
import tensorflow as tf
import tflex
import dnnlib
import dnnlib.tflib as tflib

from tensorflow.python.platform import gfile
from tensorflow.python.framework import errors_impl

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files.
        resolution      = None,     # Dataset resolution, None = autodetect.
        label_file      = None,     # Relative path of the labels file, None = autodetect.
        max_label_size  = None,     # 0 = no labels, 'full' = full labels, <int> = N first label components.
        max_images      = None,     # Maximum number of images to use, None = use all images.
        repeat          = True,     # Repeat dataset indefinitely?
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        batch_size      = None,     # Fixed batch size
        num_threads     = 2):       # Number of concurrent threads.

        if max_label_size is None:
          if 'LABEL_SIZE' not in os.environ:
            max_label_size = 0
          else:
            try:
              max_label_size = int(os.environ['LABEL_SIZE'])
            except:
              max_label_size = os.environ['LABEL_SIZE']
        assert max_label_size == 'full' or isinstance(max_label_size, int) and max_label_size >= 0

        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []        # [channels, height, width]
        self.dtype              = 'uint8'
        self.dynamic_range      = [0, 255]
        self.label_file         = label_file
        self.label_size         = None      # components
        self.label_dtype        = None
        self._np_labels         = None
        self._tf_minibatch_in   = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1

        # List tfrecords files and inspect their shapes.
        assert gfile.IsDirectory(self.tfrecord_dir)
        tfr_files = sorted(tf.io.gfile.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) >= 1

        # To avoid parsing the entire dataset looking for the max
        # resolution, just assume that we can extract the LOD level
        # from the tfrecords file name.
        tfr_shapes = []
        lod = -1
        for tfr_file in tfr_files:
          match = re.match('.*?([0-9]+).tfrecords', tfr_file)
          if match:
            level = int(match.group(1))
            res = 2**level
            tfr_shapes.append((3, res, res))

        # Autodetect label filename.
        if self.label_file is None:
            guess = sorted(tf.io.gfile.glob(os.path.join(self.tfrecord_dir, '*.labels')))
            if len(guess):
                self.label_file = guess[0]
        elif not tf.io.gfile.exists(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if tf.io.gfile.exists(guess):
                self.label_file = guess

        # Determine shape and resolution.
        max_shape = max(tfr_shapes, key=np.prod)
        self.resolution = resolution if resolution is not None else max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))
        assert all(lod in tfr_lods for lod in range(self.resolution_log2 - 1))

        # Load labels.
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<16, 0], dtype=np.float32)
        if self.label_file is not None and max_label_size != 0:
            with gfile.GFile(self.label_file, 'rb') as f:
              self._np_labels = np.load(f)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        if max_images is not None and self._np_labels.shape[0] > max_images:
            self._np_labels = self._np_labels[:max_images]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name
        self.tfr = list(zip(tfr_files, tfr_shapes, tfr_lods))

        def finalize():
          # Build TF expressions.
          with tf.name_scope('Dataset'), tflex.device('/cpu:0'):
              self._tf_minibatch_in = batch_size if batch_size is not None or batch_size <= 0 else tf.placeholder(tf.int64, name='minibatch_in', shape=[])
              self._tf_labels_var, self._tf_labels_init = tflib.create_var_with_large_initial_value2(self._np_labels, name='labels_var')
              with tf.control_dependencies([self._tf_labels_init]):
                  self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)
              for tfr_file, tfr_shape, tfr_lod in self.tfr:
                  if tfr_lod < 0:
                      continue
                  dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
                  if max_images is not None:
                      dset = dset.take(max_images)
                  dset = dset.map(self.parse_tfrecord_tf, num_parallel_calls=num_threads)
                  dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
                  bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
                  if shuffle_mb > 0:
                      dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
                  if repeat:
                      dset = dset.repeat()
                  if prefetch_mb > 0:
                      dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
                  if batch_size is None or batch_size > 0:
                      dset = dset.batch(self._tf_minibatch_in)
                  self._tf_datasets[tfr_lod] = dset
              self._tf_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
              self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) if batch_size is None else tf.no_op() for lod, dset in self._tf_datasets.items()}
        self.finalize = finalize

    def close(self):
        pass

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            #self._tf_init_ops[lod].run(*[{self._tf_minibatch_in: minibatch_size}] if not isinstance(self._tf_minibatch_in, int) and self._tf_minibatch_in is not None else [])
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images, labels
        return self._tf_iterator.get_next()

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => images, labels
        while True:
            self.configure(minibatch_size, lod)
            with tf.name_scope('Dataset'):
                if self._tf_minibatch_np is None:
                    self._tf_minibatch_np = self.get_minibatch_tf()
                try:
                    return tflib.run(self._tf_minibatch_np)
                except errors_impl.AbortedError:
                    import traceback
                    traceback.print_exc()

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        with tf.name_scope('Dataset'):
            if self.label_size > 0:
                with tflex.device('/cpu:0'):
                    return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.int32))
            return tf.zeros([minibatch_size, 0], self.label_dtype)

    # Get random labels as NumPy array.
    def get_random_labels_np(self, minibatch_size): # => labels
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        return np.zeros([minibatch_size, 0], self.label_dtype)

    # Parse individual image from a tfrecords file into TensorFlow expression.
    @staticmethod
    def parse_tfrecord_tf(record):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features['data'], tf.uint8)
        return tf.reshape(data, features['shape'])

    # Parse individual image from a tfrecords file into TensorFlow expression.
    @staticmethod
    def parse_tfrecord_tf_float(record):
        img = TFRecordDataset.parse_tfrecord_tf(record)
        return tf.cast(img, dtype=tf.float32)

    # Parse individual image from a tfrecords file into NumPy array.
    @staticmethod
    def parse_tfrecord_np(record):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value # pylint: disable=no-member
        data = ex.features.feature['data'].bytes_list.value[0] # pylint: disable=no-member
        return np.fromstring(data, np.uint8).reshape(shape)

#----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.

def load_dataset(class_name=None, data_dir=None, verbose=False, **kwargs):
    kwargs = dict(kwargs)
    if 'tfrecord_dir' in kwargs:
        if class_name is None:
            class_name = __name__ + '.TFRecordDataset'
        if data_dir is not None:
            kwargs['tfrecord_dir'] = os.path.join(data_dir, kwargs['tfrecord_dir'])

    assert class_name is not None
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = dnnlib.util.get_obj_by_name(class_name)(**kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset

#----------------------------------------------------------------------------
