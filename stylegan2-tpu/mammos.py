import sys
import os

sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pymen/bin')]

if 'COLAB_TPU_ADDR' in os.environ:
  os.environ['TPU_NAME'] = 'grpc://' + os.environ['COLAB_TPU_ADDR']

os.environ['NOISY'] = '1'
# --- set resolution and label size here:
label_size = int(os.environ['LABEL_SIZE'])
resolution = int(os.environ['RESOLUTION'])
num_channels = int(os.environ['NUM_CHANNELS'])
model_dir = os.environ['MODEL_DIR']
count = int(os.environ['COUNT']) if 'COUNT' in os.environ else 1
grid_image_size = int(os.environ['GRID_SIZE']) if 'GRID_SIZE' in os.environ else 9
batch_size = 1
# ------------------------


import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tqdm
from training.networks_stylegan2 import *
from training import misc

import dnnlib
from dnnlib import EasyDict

import tensorflow as tf
import tflex
import os
import numpy as np

tflex.self = globals()

train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
G_args    = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
D_args    = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
G_loss    = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')      # Options for generator loss.
D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
sched     = EasyDict()                                                     # Options for TrainingSchedule.
grid      = EasyDict(size='8k', layout='random')                           # Options for setup_snapshot_image_grid().
sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
tf_config = {'rnd.np_random_seed': 1000}    
label_dtype = np.int64
sched.minibatch_gpu = 1

def with_session(sess, f, *args, **kws):
  with sess.as_default():
    return f(*args, **kws)

async def with_session_async(sess, f, *args, **kws):
  with sess.as_default():
    return await f(*args, **kws)

def init(session=None, num_channels=None, resolution=None, label_size=None):
  label_size = int(os.environ['LABEL_SIZE']) if label_size is None else label_size
  resolution = int(os.environ['RESOLUTION']) if resolution is None else resolution
  num_channels = int(os.environ['NUM_CHANNELS']) if num_channels is None else num_channels
  dnnlib.tflib.init_tf()

  session = tflex.get_session(session)
  pprint(session.list_devices())

  tflex.set_override_cores(tflex.get_cores())
  with tflex.device('/gpu:0'):
    tflex.G = tflib.Network('G', num_channels=num_channels, resolution=resolution, label_size=label_size, **G_args)
    tflex.G.print_layers()
    tflex.Gs, tflex.Gs_finalize = tflex.G.clone2('Gs')
    tflex.Gs_finalize()
    tflex.D = tflib.Network('D', num_channels=num_channels, resolution=resolution, label_size=label_size, **D_args)
    tflex.D.print_layers()
  tflib.run(tf.global_variables_initializer())
  return session

def load_checkpoint(path, session=None, var_list=None):
  if var_list is None:
    var_list = tflex.Gs.trainables
  ckpt = tf.train.latest_checkpoint(path) or path
  assert ckpt is not None
  print('Loading checkpoint ' + ckpt)
  saver = tf.train.Saver(var_list=var_list)
  saver.restore(tflex.get_session(session), ckpt)
  return ckpt

init()
load_checkpoint(model_dir)

import PIL.Image
import numpy as np
import requests
from io import BytesIO

def get_images(tags, seed = 0, mu = 0, sigma = 0, truncation=None):
    print("Generating mammos...")

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation is not None:
        Gs_kwargs.truncation_psi = truncation
    rnd = np.random.RandomState(seed)

    all_seeds = [seed] * batch_size
    all_z = np.stack([np.random.RandomState(seed).randn(*tflex.Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    print(all_z.shape)

    drange_net = [-1, 1]
    with tflex.device('/gpu:0'):
      result = tflex.Gs.run(all_z, None, is_validation=True, randomize_noise=False, minibatch_size=sched.minibatch_gpu)
      if result.shape[1] > 3:
        final = result[:, 3, :, :]
      else:
        final = None
      result = result[:, 0:3, :, :]
      img = misc.convert_to_pil_image(misc.create_image_grid(result, (1, 1)), drange_net)
      img.save('mammos.png')
      return result, img
