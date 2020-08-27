# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Helper for adding automatically tracked values to Tensorboard.

Autosummary creates an identity op that internally keeps track of the input
values and automatically shows up in TensorBoard. The reported value
represents an average over input components. The average is accumulated
constantly over time and flushed when save_summaries() is called.

Notes:
- The output tensor must be used as an input for something else in the
  graph. Otherwise, the autosummary op will not get executed, and the average
  value will not get accumulated.
- It is perfectly fine to include autosummaries with the same name in
  several places throughout the graph, even if they are executed concurrently.
- It is ok to also pass in a python scalar or numpy array. In this case, it
  is added to the average immediately.
"""

from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tflex
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2

from . import tfutil
from .tfutil import TfExpression
from .tfutil import TfExpressionEx

# Enable "Custom scalars" tab in TensorBoard for advanced formatting.
# Disabled by default to reduce tfevents file size.
enable_custom_scalars = False

_dtype = tf.float64
_vars = OrderedDict()  # name => [var, ...]
_immediate = OrderedDict()  # name => update_op, update_value
_finalized = False
_merge_op = None


def _create_var(name: str, value_expr: TfExpression) -> TfExpression:
    """Internal helper for creating autosummary accumulators."""
    assert not _finalized
    name_id = name.replace("/", "_")
    v = tf.cast(value_expr, _dtype)

    if v.shape.is_fully_defined():
        size = np.prod(v.shape.as_list())
        size_expr = tf.constant(size, dtype=_dtype)
    else:
        size = None
        size_expr = tf.reduce_prod(tf.cast(tf.shape(v), _dtype))

    if size == 1:
        if v.shape.ndims != 0:
            v = tf.reshape(v, [])
        v = [size_expr, v, tf.square(v)]
    else:
        v = [size_expr, tf.reduce_sum(v), tf.reduce_sum(tf.square(v))]
    v = tf.cond(tf.is_finite(v[1]), lambda: tf.stack(v), lambda: tf.zeros(3, dtype=_dtype))

    with tfutil.absolute_name_scope("Autosummary/" + name_id), tf.control_dependencies(None):
        var = tf.Variable(tf.zeros(3, dtype=_dtype), trainable=False)  # [sum(1), sum(x), sum(x**2)]
    update_op = tf.cond(tf.is_variable_initialized(var), lambda: tf.assign_add(var, v), lambda: tf.assign(var, v))

    if name in _vars:
        _vars[name].append(var)
    else:
        _vars[name] = [var]
    return update_op

import os
from . import tpu_summaries

tpu_summary = None

def get_tpu_summary(model_dir=None):
    global tpu_summary
    if tpu_summary is None:
        if model_dir is None:
            model_dir = os.path.join(os.environ['MODEL_DIR'], 'autosummary')
        tpu_summary = tpu_summaries.TpuSummaries(model_dir)
    else:
        assert model_dir is None
    return tpu_summary

# TODO(joelshor): Make this a special case of `image_reshaper`.
def image_grid(input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
  """Arrange a minibatch of images into a grid to form a single image.

  Args:
    input_tensor: Tensor. Minibatch of images to format, either 4D
        ([batch size, height, width, num_channels]) or flattened
        ([batch size, height * width * num_channels]).
    grid_shape: Sequence of int. The shape of the image grid,
        formatted as [grid_height, grid_width].
    image_shape: Sequence of int. The shape of a single image,
        formatted as [image_height, image_width].
    num_channels: int. The number of channels in an image.

  Returns:
    Tensor representing a single image in which the input images have been
    arranged into a grid.

  Raises:
    ValueError: The grid shape and minibatch size don't match, or the image
        shape and number of channels are incompatible with the input tensor.
  """
  if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
    raise ValueError("Grid shape %s incompatible with minibatch size %i." %
                     (grid_shape, int(input_tensor.shape[0])))
  if len(input_tensor.shape) == 2:
    num_features = image_shape[0] * image_shape[1] * num_channels
    if int(input_tensor.shape[1]) != num_features:
      raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor.")
  elif len(input_tensor.shape) == 4:
    if (int(input_tensor.shape[1]) != image_shape[0] or
        int(input_tensor.shape[2]) != image_shape[1] or
        int(input_tensor.shape[3]) != num_channels):
      raise ValueError("Image shape and number of channels incompatible with "
                       "input tensor. %s vs %s" % (
                           input_tensor.shape, (image_shape[0], image_shape[1],
                                                num_channels)))
  else:
    raise ValueError("Unrecognized input tensor format.")
  height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
  input_tensor = tf.reshape(
      input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,))
  input_tensor = tf.transpose(a=input_tensor, perm=[0, 1, 3, 2, 4])
  input_tensor = tf.reshape(
      input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
  input_tensor = tf.transpose(a=input_tensor, perm=[0, 2, 1, 3])
  input_tensor = tf.reshape(input_tensor, [1, height, width, num_channels])
  return input_tensor

num_replicas = None

def get_num_replicas():
    assert num_replicas is not None
    return num_replicas

def set_num_replicas(n):
    global num_replicas
    num_replicas = n

def _i(x): return tf.transpose(x, [0, 2, 3, 1])
def _o(x): return tf.transpose(x, [0, 3, 1, 2])

def autoimages(summary_name, images, grid_shape=None, res=None):
    """Called from model_fn() to add a grid of images as summary."""
    # convert from [batch, channels, width, height] to [batch, width, height, channels]
    images = _i(images)
    # All summary tensors are synced to host 0 on every step. To avoid sending
    # more images then needed we transfer at most `sampler_per_replica` to
    # create a 8x8 image grid.
    batch_size_per_replica = images.shape[0].value
    num_replicas = get_num_replicas()
    total_num_images = batch_size_per_replica * num_replicas
    if callable(grid_shape):
        grid_shape = grid_shape(total_num_images)
    if grid_shape is None:
        w = int(os.environ['GRID_WIDTH']) if 'GRID_WIDTH' in os.environ else 3
        h = int(os.environ['GRID_HEIGHT']) if 'GRID_HEIGHT' in os.environ else 3
        grid_shape = (w, h)
    sample_num_images = np.prod(grid_shape)
    if total_num_images >= sample_num_images:
        # This can be more than 64. We slice all_images below.
        samples_per_replica = int(np.ceil(sample_num_images / num_replicas))
    else:
        samples_per_replica = batch_size_per_replica
    if res is None:
        res = int(os.environ['RESOLUTION']) if 'RESOLUTION' in os.environ else 64
    num_channels = int(os.environ["NUM_CHANNELS"]) if "NUM_CHANNELS" in os.environ else 3
    image_shape = [res, res, num_channels]
    sample_res = int(os.environ['GRID_RESOLUTION']) if 'GRID_RESOLUTION' in os.environ else 200
    sample_shape = [sample_res, sample_res, num_channels]
    tf.logging.info("autoimages(%s, %s): Creating %dx%d images grid at resolution %dx%d channel count %d across %d replicas",
                    repr(summary_name), repr(images),
                    grid_shape[0], grid_shape[1], image_shape[0], image_shape[1], image_shape[2],
                    num_replicas)
    images = images[:samples_per_replica]

    def _merge_images_to_grid(all_images):
        all_images = all_images[:np.prod(grid_shape)]
        shape = image_shape
        if shape[0] > sample_shape[0] or shape[1] > sample_shape[1]:
            tf.logging.info('autoimages(%s, %s): Downscaling sampled images from %dx%d to %dx%d',
                            repr(summary_name), repr(all_images),
                            shape[0], shape[1],
                            sample_shape[0], sample_shape[1])
            all_images = tf.image.resize(all_images, sample_shape[0:2], method=tf.image.ResizeMethod.AREA)
            shape = sample_shape
        return image_grid(
            all_images,
            grid_shape=grid_shape,
            image_shape=shape[:2],
            num_channels=shape[2])
    get_tpu_summary().image(summary_name,
                            images,
                            reduce_fn=_merge_images_to_grid)

def autosummary(name: str, value: TfExpressionEx, passthru: TfExpressionEx = None, condition: TfExpressionEx = True) -> TfExpressionEx:
    """Create a new autosummary.

    Args:
        name:     Name to use in TensorBoard
        value:    TensorFlow expression or python value to track
        passthru: Optionally return this TF node without modifications but tack an autosummary update side-effect to this node.

    Example use of the passthru mechanism:

    n = autosummary('l2loss', loss, passthru=n)

    This is a shorthand for the following code:

    with tf.control_dependencies([autosummary('l2loss', loss)]):
        n = tf.identity(n)
    """
    tf.logging.info('autosummary(%s, %s)', repr(name), repr(value))
    get_tpu_summary().scalar(name, value)
    return value
    tfutil.assert_tf_initialized()
    name_id = name.replace("/", "_")

    if tfutil.is_tf_expression(value):
        with tf.name_scope("summary_" + name_id), tflex.device(value.device):
            condition = tf.convert_to_tensor(condition, name='condition')
            update_op = tf.cond(condition, lambda: tf.group(_create_var(name, value)), tf.no_op)
            with tf.control_dependencies([update_op]):
                return tf.identity(value if passthru is None else passthru)

    else:  # python scalar or numpy array
        assert not tfutil.is_tf_expression(passthru)
        assert not tfutil.is_tf_expression(condition)
        if condition:
            if name not in _immediate:
                with tfutil.absolute_name_scope("Autosummary/" + name_id), tflex.device(None), tf.control_dependencies(None):
                    update_value = tf.placeholder(_dtype)
                    update_op = _create_var(name, update_value)
                    _immediate[name] = update_op, update_value
            update_op, update_value = _immediate[name]
            tfutil.run(update_op, {update_value: value})
        return value if passthru is None else passthru


def finalize_autosummaries() -> None:
    """Create the necessary ops to include autosummaries in TensorBoard report.
    Note: This should be done only once per graph.
    """
    global _finalized
    tfutil.assert_tf_initialized()

    if _finalized:
        return None

    _finalized = True
    tfutil.init_uninitialized_vars([var for vars_list in _vars.values() for var in vars_list])

    # Create summary ops.
    with tflex.device(None), tf.control_dependencies(None):
        for name, vars_list in _vars.items():
            name_id = name.replace("/", "_")
            with tfutil.absolute_name_scope("Autosummary/" + name_id):
                moments = tf.add_n(vars_list)
                moments /= moments[0]
                with tf.control_dependencies([moments]):  # read before resetting
                    reset_ops = [tf.assign(var, tf.zeros(3, dtype=_dtype)) for var in vars_list]
                    with tf.name_scope(None), tf.control_dependencies(reset_ops):  # reset before reporting
                        mean = moments[1]
                        std = tf.sqrt(moments[2] - tf.square(moments[1]))
                        tf.summary.scalar(name, mean)
                        if enable_custom_scalars:
                            tf.summary.scalar("xCustomScalars/" + name + "/margin_lo", mean - std)
                            tf.summary.scalar("xCustomScalars/" + name + "/margin_hi", mean + std)

    # Setup layout for custom scalars.
    layout = None
    if enable_custom_scalars:
        cat_dict = OrderedDict()
        for series_name in sorted(_vars.keys()):
            p = series_name.split("/")
            cat = p[0] if len(p) >= 2 else ""
            chart = "/".join(p[1:-1]) if len(p) >= 3 else p[-1]
            if cat not in cat_dict:
                cat_dict[cat] = OrderedDict()
            if chart not in cat_dict[cat]:
                cat_dict[cat][chart] = []
            cat_dict[cat][chart].append(series_name)
        categories = []
        for cat_name, chart_dict in cat_dict.items():
            charts = []
            for chart_name, series_names in chart_dict.items():
                series = []
                for series_name in series_names:
                    series.append(layout_pb2.MarginChartContent.Series(
                        value=series_name,
                        lower="xCustomScalars/" + series_name + "/margin_lo",
                        upper="xCustomScalars/" + series_name + "/margin_hi"))
                margin = layout_pb2.MarginChartContent(series=series)
                charts.append(layout_pb2.Chart(title=chart_name, margin=margin))
            categories.append(layout_pb2.Category(title=cat_name, chart=charts))
        layout = summary_lib.custom_scalar_pb(layout_pb2.Layout(category=categories))
    return layout

def save_summaries(file_writer, global_step=None):
    """Call FileWriter.add_summary() with all summaries in the default graph,
    automatically finalizing and merging them on the first call.
    """
    return
    global _merge_op
    tfutil.assert_tf_initialized()

    if _merge_op is None:
        layout = finalize_autosummaries()
        if layout is not None:
            file_writer.add_summary(layout)
        with tflex.device(None), tf.control_dependencies(None):
            _merge_op = tf.summary.merge_all()

    if _merge_op is not None:
        file_writer.add_summary(_merge_op.eval(), global_step)
