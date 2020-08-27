# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""ImageNet preprocessing for ResNet."""

from absl import flags
import tensorflow as tf

#FLAGS = flags.FLAGS
class Namespace:
  pass

FLAGS = Namespace()
FLAGS.cache_decoded_image = False


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    cropped image `Tensor`
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
    if FLAGS.cache_decoded_image:
      shape = tf.shape(image_bytes)
    else:
      shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    if FLAGS.cache_decoded_image:
      image = tf.image.crop_to_bounding_box(image_bytes, offset_y, offset_x,
                                            target_height, target_width)
    else:
      image = tf.image.decode_and_crop_jpeg(
          image_bytes, crop_window, channels=3)

    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  return tf.image.resize_area([image], [image_size, image_size])[0]


def _decode_and_center_crop_image(image_bytes, image_size, crop_padding=32):
  """Crops to center of image with padding then scales image_size."""
  image = tf.io.decode_image(image_bytes, channels=3)
  shape = tf.shape(image)
  #shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]
  channels = shape[2]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  if 'RANDOM_CROP' in os.environ:
    image = tf.image.random_crop(image, [padded_center_crop_size, padded_center_crop_size, channels])
  else:
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, padded_center_crop_size, padded_center_crop_size)
  image = tf.image.resize_area([image], [image_size, image_size])[0]

  return image

def _decode_and_center_crop(image_bytes, image_size, crop_padding=None):
  if 'NO_CENTER_CROP' in os.environ:
    image = tf.io.decode_image(image_bytes, channels=3)
    image = tf.image.resize_area([image], [image_size, image_size])[0]
    return image
  else:
    if crop_padding is None and 'CROP_PADDING' in os.environ:
      crop_padding = int(os.environ['CROP_PADDING'])
    if crop_padding is None:
      crop_padding = 32
    return _decode_and_center_crop_image(image_bytes, image_size, crop_padding=crop_padding)

def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_for_train(image_bytes, use_bfloat16, image_size):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes, image_size)
  image = _flip(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def preprocess_for_eval(image_bytes, use_bfloat16, image_size):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def preprocess_image(image_bytes,
                     image_size,
                     is_training=False,
                     use_bfloat16=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor` with value range of [0, 255].
  """
  if is_training and False: # a bit of a weird hack; we override the training pipeline in the eval functions.
    img = preprocess_for_train(image_bytes, use_bfloat16, image_size)
  else:
    img = preprocess_for_eval(image_bytes, use_bfloat16, image_size)
  #img = tf.transpose(img, [2, 0, 1])
  return img


# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

import abc
from collections import namedtuple
import functools
import math
import os
from absl import flags
import tensorflow as tf

#FLAGS = flags.FLAGS


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = preprocess_image(
        image_bytes=image_bytes, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label):
  """Build an Example proto for an example.

  Args:
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network

  Returns:
    Example proto
  """

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/class/label': _int64_feature(label),
              'image/encoded': _bytes_feature(image_buffer)
          }))
  return example


class ImageNetTFExampleInput(object):
  """Base class for ImageNet input_fn generator.

  Args:
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick
    num_cores: `int` for the number of TPU cores
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               num_cores=8,
               image_size=224,
               prefetch_depth_auto_tune=False,
               transpose_input=False):
    self.image_preprocessing_fn = preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.num_cores = num_cores
    self.transpose_input = transpose_input
    self.image_size = image_size
    self.prefetch_depth_auto_tune = prefetch_depth_auto_tune

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      if FLAGS.train_batch_size // FLAGS.num_cores > 8:
        shape = [None, None, None, batch_size]
      else:
        shape = [None, None, batch_size, None]
      images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
      images = tf.reshape(images, [-1])
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))

    return images, labels

  def dataset_parser(self, value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    # Subtract one so that labels are in [0, 1000).
    bias = int(os.environ['LABEL_BIAS']) if 'LABEL_BIAS' in os.environ else 0
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) + bias

    # Return all black images for padded data.
    image = tf.cond(
        label < 0, lambda: self._get_null_input(None), lambda: self.  # pylint: disable=g-long-lambda
        image_preprocessing_fn(
            image_bytes=image_bytes,
            is_training=self.is_training,
            image_size=self.image_size,
            use_bfloat16=self.use_bfloat16))

    return image, label

  def dataset_parser_static(self, value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

       This only decodes the image, which is prepared for caching.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image_bytes = tf.io.decode_image(image_bytes, channels=3)

    # Subtract one so that labels are in [0, 1000).
    bias = int(os.environ['LABEL_BIAS']) if 'LABEL_BIAS' in os.environ else 0
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) + bias
    return image_bytes, label

  def dataset_parser_dynamic(self, image_bytes, label):
    return self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16), label

  def pad_dataset(self, dataset, num_hosts):
    """Pad the eval dataset so that eval can have the same batch size as training."""
    num_dataset_per_shard = int(
        math.ceil(FLAGS.num_eval_images / FLAGS.eval_batch_size) *
        FLAGS.eval_batch_size / num_hosts)
    example_string = 'dummy_string'
    padded_example = _convert_to_example(
        str.encode(example_string), -1).SerializeToString()
    padded_dataset = tf.data.Dataset.from_tensors(
        tf.constant(padded_example, dtype=tf.string))
    padded_dataset = padded_dataset.repeat(num_dataset_per_shard)

    dataset = dataset.concatenate(padded_dataset).take(num_dataset_per_shard)
    return dataset

  @abc.abstractmethod
  def make_source_dataset(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.

    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.

    If self.is_training, the dataset should be infinite.

    Args:
      index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      if 'dataset_index' in params:
        current_host = params['dataset_index']
        num_hosts = params['dataset_num_shards']
      else:
        current_host = 0
        num_hosts = 1

    dataset = self.make_source_dataset(current_host, num_hosts)

    if not self.is_training and False:
      # Padding for eval.
      dataset = self.pad_dataset(dataset, num_hosts)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    if self.is_training and FLAGS.cache_decoded_image:
      dataset = dataset.apply(
          tf.contrib.data.map_and_batch(
              self.dataset_parser_dynamic,
              batch_size=batch_size,
              num_parallel_batches=self.num_cores,
              drop_remainder=True))
    else:
      dataset = dataset.apply(
          tf.contrib.data.map_and_batch(
              self.dataset_parser,
              batch_size=batch_size,
              num_parallel_batches=self.num_cores,
              drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      if FLAGS.train_batch_size // FLAGS.num_cores > 8:
        transpose_array = [1, 2, 3, 0]
      else:
        transpose_array = [1, 2, 0, 3]
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, transpose_array), labels
                                 ),
          num_parallel_calls=self.num_cores)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    if self.prefetch_depth_auto_tune:
      dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    else:
      dataset = dataset.prefetch(4)

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)
    return dataset


class ImageNetInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:

      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               data_dir,
               is_training=True,
               use_bfloat16=False,
               transpose_input=False,
               image_size=224,
               num_parallel_calls=64,
               num_cores=8,
               prefetch_depth_auto_tune=False,
               cache=False):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data;
          if 'null' (the literal string 'null') or implicitly False
          then construct a null pipeline, consisting of empty images
          and blank labels.
      image_size: size of input images
      num_parallel_calls: concurrency level to use when reading data from disk.
      num_cores: Number of prefetch threads
      prefetch_depth_auto_tune: Auto-tuning prefetch depths in input pipeline
      cache: if true, fill the dataset by repeating from its cache
    """
    super(ImageNetInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        num_cores=num_cores,
        prefetch_depth_auto_tune=prefetch_depth_auto_tune,
        transpose_input=transpose_input)
    self.data_dir = data_dir
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces
          the same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3], tf.bfloat16
                    if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(ImageNetInput, self).dataset_parser(value)

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Shuffle the filenames to ensure better randomization.
    file_patterns = [x.strip() for x in self.data_dir.split(',') if len(x.strip()) > 0]

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = None
    for pattern in file_patterns:
      x = tf.data.Dataset.list_files(pattern, shuffle=False)
      if dataset is None:
        dataset = x
      else:
        dataset = dataset.concatenate(x)
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      # We shuffle only during training, and during training, we must produce an
      # infinite dataset, so apply the fused shuffle_and_repeat optimized
      # dataset transformation.
      dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(1024 * 16))
      #dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_parallel_calls, sloppy=True))

    if self.is_training and FLAGS.cache_decoded_image:
      dataset = dataset.map(
          self.dataset_parser_static,
          num_parallel_calls=self.num_parallel_calls)

    if self.cache:
      dataset = dataset.cache()
    if self.is_training:
      # We shuffle only during training, and during training, we must produce an
      # infinite dataset, so apply the fused shuffle_and_repeat optimized
      # dataset transformation.
      dataset = dataset.apply(
          tf.contrib.data.shuffle_and_repeat(1024 * 16))
    return dataset


# Defines a selection of data from a Cloud Bigtable.
BigtableSelection = namedtuple('BigtableSelection',
                               ['project',
                                'instance',
                                'table',
                                'prefix',
                                'column_family',
                                'column_qualifier'])


class ImageNetBigtableInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a Bigtable for training or evaluation.
  """

  def __init__(self, is_training, use_bfloat16, transpose_input, selection):
    """Constructs an ImageNet input from a BigtableSelection.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      selection: a BigtableSelection specifying a part of a Bigtable.
    """
    super(ImageNetBigtableInput, self).__init__(
        is_training=is_training,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input)
    self.selection = selection

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    data = self.selection
    client = tf.contrib.cloud.BigtableClient(data.project, data.instance)
    table = client.table(data.table)
    ds = table.parallel_scan_prefix(data.prefix,
                                    columns=[(data.column_family,
                                              data.column_qualifier)])
    # The Bigtable datasets will have the shape (row_key, data)
    ds_data = ds.map(lambda index, data: data)

    if self.is_training:
      ds_data = ds_data.repeat()

    return ds_data
