#!/usr/bin/env python3
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
import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tensorflow.compat.v1 as tf

class Namespace():
  pass

me = Namespace()

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label_int, label_str, height,
                        width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label_int: integer, identifier for ground truth (0-based)
    label_str: string, identifier for ground truth, e.g., 'daisy'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height': _int64_feature(height),
              'image/width': _int64_feature(width),
              'image/colorspace': _bytes_feature(colorspace),
              'image/channels': _int64_feature(channels),
              'image/class/label': _int64_feature(label_int +
                                                  1),  # model expects 1-based
              'image/class/synset': _bytes_feature(label_str),
              'image/format': _bytes_feature(image_format),
              'image/filename': _bytes_feature(os.path.basename(filename)),
              'image/encoded': _bytes_feature(image_buffer)
          }))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, session=None):
    # Create a single Session to run all image coding calls.
    session = tf.get_default_session() if session is None else session
    self._sess = tf.Session() if session is None else session

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def __del__(self):
    self._sess.close()

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


# Parse individual image from a tfrecords file into TensorFlow expression.
def parse_tfrecord_tf(record):
  features = tf.parse_single_example(record, features={ 'shape': tf.FixedLenFeature([3], tf.int64), 'data': tf.FixedLenFeature([], tf.string)})
  data = tf.decode_raw(features['data'], tf.uint8)
  return tf.reshape(data, features['shape'])

def parse_tfrecord_file(tfr_file, num_threads=8):
  dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
  dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
  return dset

def init_dataset(dset):
  tf_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
  tf_init_ops = tf_iterator.make_initializer(dset)
  return tf_iterator, tf_init_ops

from tensorflow.python.framework import errors_impl

def iterate_stylegan_records(tfr_file, session=None):
  if session is None:
    session = tf.get_default_session()
  dset = parse_tfrecord_file(tfr_file)
  it, init = init_dataset(dset)
  session.run(init)
  with session.graph.as_default():
    op = tf.transpose(it.get_next(), [1, 2, 0])
  try:
    while True:
      yield sess.run(op)
  except errors_impl.OutOfRangeError:
    pass

def _is_png(filename):
  return filename.endswith('.png') or filename.endswith('.PNG')

def _is_png_data(image_data):
  return image_data.startswith(b'\x89PNG')

def _is_cmyk(filename):
  return False # TODO

def get_coder(coder=None):
  coder = me.coder = me.coder if hasattr(me, 'coder') else ImageCoder()
  return coder

import requests

def getfile(path):
  if path.startswith('http'):
    r = requests.get(path, stream=True)
    if r.status_code == 200:
      return r.raw.read()
    else:
      raise Exception("Couldn't open URL: %s" % path)
  with tf.gfile.FastGFile(path, 'rb') as f:
    return f.read()

def _process_image(filename, coder=None):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  coder = get_coder(coder)
  # Read the image file.
  image_data = getfile(filename)
  # Clean the dirty data.
  if _is_png_data(image_data):
    # 1 image is a PNG.
    tf.logging.info('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  elif _is_cmyk(filename):
    # 22 JPEG images are in CMYK colorspace.
    tf.logging.info('Converting CMYK to RGB for %s' % filename)
    image_data = coder.cmyk_to_rgb(image_data)
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3
  return image_data, height, width, image

def tf_randi(*args, **kws):
  assert len(args) > 0
  if len(args) == 1:
    lo, hi = [0] + [x for x in args]
  else:
    lo, hi = args
  return tf.random.uniform((), minval=lo, maxval=hi, dtype=tf.int32, **kws)

def tf_rand(*args, **kws):
  if len(args) == 0:
    lo, hi = 0.0, 1.0
  elif len(args) == 1:
    lo, hi = [0] + [x for x in args]
  else:
    lo, hi = args
  return tf.random.uniform((), minval=lo, maxval=hi, **kws)

def tf_biased_rand(*args, bias=1, **kws):
  x = tf_rand(*args, **kws)
  # simple technique to bias the result towards the center.
  for i in range(bias-1):
    x += tf_rand(*args, **kws)
  dtype = kws.pop('dtype') if 'dtype' in kws else tf.float32
  x = tf.cast(x, tf.float32) / bias
  x = tf.cast(x, dtype)
  return x

def tf_between(*args, bias=1, **kws):
  if bias <= 0:
    return tf_randi(*args, **kws)
  else:
    return tf_biased_rand(*args, dtype=tf.int32, bias=bias, **kws)
  
def random_crop(image_bytes, scope=None, resize=None, method=tf.image.ResizeMethod.AREA, seed=None):
  with tf.name_scope(scope, 'random_crop', [image_bytes]):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    w = shape[0]
    h = shape[1]
    channels = shape[2]
    x, y = 0, 0
    n = 3
    image = tf.cond(w > h,
        lambda: tf.image.decode_and_crop_jpeg(image_bytes, tf.stack([x + tf_between(w - h, seed=seed), y, h, h]), channels=n),
        lambda: tf.cond(h > w,
          lambda: tf.image.decode_and_crop_jpeg(image_bytes, tf.stack([x, y + tf_between(h - w, seed=seed), w, w]), channels=n),
          lambda: tf.image.decode_jpeg(image_bytes, channels=n)))
    if resize:
      image_size = [resize, resize] if isinstance(resize, int) or isinstance(resize, float) else resize
      image = tf.image.resize([image], image_size, method=method)[0]
    return image
    
    
# with open('test.jpg', 'wb') as f: f.write(tf.get_default_session().run(tf.io.encode_jpeg(sess.run(random_crop(open(np.random.choice(list(glob('*.jpg'))), 'rb').read())))))



IMAGE_SIZE = 224
CROP_PADDING = 32

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
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)

def _decode_and_random_crop(image_bytes, image_size, resize=True, method=tf.image.ResizeMethod.AREA, aspect_ratio_range=(3. / 4, 4. / 3), area_range=(0.08, 1.0)): # AREA is much higher quality than BICUBIC
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=10,
      scope=None)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)
  if resize:
    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size, resize=resize, method=method),
        lambda: tf.image.resize([image], [image_size, image_size], method=method)[0])
  else:
    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size, resize=resize, method=method),
        lambda: image)
  return image

def _decode_and_center_crop(image_bytes, image_size, resize=True, method=tf.image.ResizeMethod.AREA): # AREA is much higher quality than BICUBIC
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]
  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)
  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  if resize:
    image = tf.image.resize([image], [image_size, image_size], method=method)[0]
  return image

import os
from tensorflow.compat.v1.distribute.cluster_resolver import TPUClusterResolver

def get_target(target=None):
  if target is None and 'COLAB_TPU_ADDR' in os.environ:
    target = os.environ['COLAB_TPU_ADDR']
  if target is None and 'TPU_NAME' in os.environ:
    target = os.environ['TPU_NAME']
    if not target.startswith('grpc://'):
      target = TPUClusterResolver(target).get_master()
  if target is not None:
    print('Using target %s' % target)
  return target

def main():
  from tensorflow.python.framework.ops import disable_eager_execution
  disable_eager_execution()

  parser = argparse.ArgumentParser()
  parser.add_argument('infile')
  parser.add_argument('outfile', nargs='?')
  parser.add_argument('-r', '--resize', type=int, default=0)
  args = me.args = parser.parse_args()
  with tf.Session(get_target()) as sess:
    image_data, width, height, image = _process_image(args.infile)
    image_out = sess.run(tf.io.encode_jpeg(sess.run(random_crop(image_data, resize=args.resize))))
    if not args.outfile:
      args.outfile = os.path.basename(args.infile)
      if args.outfile == args.infile:
        print('Implicitly overwriting infile not supported; pass `%s %s` to confirm' % (args.outfile, args.outfile))
        sys.exit(1)
    with open(args.outfile, 'wb') as f:
      f.write(image_out)

if __name__ == '__main__':
  main()

def convert_to_example(csvline, categories):
  """Parse a line of CSV file and convert to TF Record.

  Args:
    csvline: line from input CSV file
    categories: list of labels
  Yields:
    serialized TF example if the label is in categories
  """
  filename, label = csvline.encode('ascii', 'ignore').split(',')
  if label in categories:
    # ignore labels not in categories list
    coder = ImageCoder()
    image_buffer, height, width, image = _process_image(filename, coder)
    del coder
    example = _convert_to_example(filename, image_buffer,
                                  categories.index(label), label, height, width)
    yield example.SerializeToString()

def main2():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_csv',
      # pylint: disable=line-too-long
      help=
      'Path to input.  Each line of input has two fields  image-file-name and label separated by a comma',
      required=True)
  parser.add_argument(
      '--validation_csv',
      # pylint: disable=line-too-long
      help=
      'Path to input.  Each line of input has two fields  image-file-name and label separated by a comma',
      required=True)
  parser.add_argument(
      '--labels_file',
      help='Path to file containing list of labels, one per line',
      required=True)
  parser.add_argument(
      '--project_id',
      help='ID (not name) of your project. Ignored by DirectRunner',
      required=True)
  parser.add_argument(
      '--runner',
      help='If omitted, uses DataFlowRunner if output_dir starts with gs://',
      default=None)
  parser.add_argument(
      '--output_dir', help='Top-level directory for TF Records', required=True)

  args = parser.parse_args()
  arguments = args.__dict__

  JOBNAME = (
      'preprocess-images-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

  PROJECT = arguments['project_id']
  OUTPUT_DIR = arguments['output_dir']

  # set RUNNER using command-line arg or based on output_dir path
  on_cloud = OUTPUT_DIR.startswith('gs://')
  if arguments['runner']:
    RUNNER = arguments['runner']
  else:
    RUNNER = 'DataflowRunner' if on_cloud else 'DirectRunner'

  # clean-up output directory since Beam will name files 0000-of-0004 etc.
  # and this could cause confusion if earlier run has 0000-of-0005, for eg
  if on_cloud:
    try:
      subprocess.check_call('gsutil -m rm -r {}'.format(OUTPUT_DIR).split())
    except subprocess.CalledProcessError:
      pass
  else:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)

  # read list of labels
  with tf.gfile.FastGFile(arguments['labels_file'], 'r') as f:
    LABELS = [line.rstrip() for line in f]
  print('Read in {} labels, from {} to {}'.format(
      len(LABELS), LABELS[0], LABELS[-1]))
  if len(LABELS) < 2:
    print('Require at least two labels')
    sys.exit(-1)

  # set up Beam pipeline to convert images to TF Records
  options = {
      'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
      'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
      'job_name': JOBNAME,
      'project': PROJECT,
      'teardown_policy': 'TEARDOWN_ALWAYS',
      'save_main_session': True
  }
  opts = beam.pipeline.PipelineOptions(flags=[], **options)

  with beam.Pipeline(RUNNER, options=opts) as p:
    # BEAM tasks
    for step in ['train', 'validation']:
      _ = (
          p
          | '{}_read_csv'.format(step) >> beam.io.ReadFromText(
              arguments['{}_csv'.format(step)])
          | '{}_convert'.format(step) >>
          beam.FlatMap(lambda line: convert_to_example(line, LABELS))
          | '{}_write_tfr'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
              os.path.join(OUTPUT_DIR, step)))
