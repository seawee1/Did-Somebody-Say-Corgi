import os

if 'COLAB_TPU_ADDR' in os.environ:
  os.environ['TPU_NAME'] = 'grpc://' + os.environ['COLAB_TPU_ADDR']

os.environ['NOISY'] = '1'

# --- set resolution and label size here:
label_size = int(os.environ['LABEL_SIZE'])
resolution = int(os.environ['RESOLUTION'])
fmap_base = (int(os.environ['FMAP_BASE']) if 'FMAP_BASE' in os.environ else 16) << 10
num_channels = int(os.environ['NUM_CHANNELS'])
model_dir = os.environ['MODEL_DIR']
channel = os.environ['CHANNEL'] if 'CHANNEL' in os.environ else 'chaos'
count = int(os.environ['COUNT']) if 'COUNT' in os.environ else 1
discord_token = os.environ['DISCORD_TOKEN']
grid_image_size = int(os.environ['GRID_SIZE']) if 'GRID_SIZE' in os.environ else 9
# ------------------------

import tqdm
from pprint import pprint as pp
from training.networks_stylegan2 import *
from training import misc

import dnnlib
from dnnlib import EasyDict

import tensorflow as tf
import tflex
import os
import numpy as np

dnnlib.tflib.init_tf()

sess = tf.get_default_session()
sess.list_devices()



cores = tflex.get_cores()
tflex.set_override_cores(cores)

#synthesis_func          = 'G_synthesis_stylegan2'
#kwargs = {'resolution': 512}
#synthesis = tflib.Network('G_synthesis', func_name=globals()[synthesis_func], **kwargs)

#sess.reset(os.environ['TPU_NAME']) # don't do this, this breaks the session

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

if 'G' not in globals():
  with tflex.device('/gpu:0'):
    G = tflib.Network('G', num_channels=num_channels, resolution=resolution, label_size=label_size, fmap_base=fmap_base, **G_args)
    G.print_layers()
    Gs, Gs_finalize = G.clone2('Gs')
    Gs_finalize()
    D = tflib.Network('D', num_channels=num_channels, resolution=resolution, label_size=label_size, fmap_base=fmap_base, **D_args)
    D.print_layers()

def rand_latent(n, seed=None):
  if seed is not None:
    if seed < 0:
      seed = 2*32 - seed
    np.random.seed(seed)
  result = np.random.randn(n, *G.input_shape[1:])
  if seed is not None:
    np.random.seed()
  return result

grid_size = (2, 2)
gw, gh = grid_size
gn = np.prod(grid_size)
grid_latents = rand_latent(gn, seed=-1)
grid_labels = np.zeros([gw * gh, label_size], dtype=label_dtype)

def tfinit():
  tflib.run(tf.global_variables_initializer())

tfinit()

saver = tf.train.Saver()

def load_checkpoint(path):
  ckpt = tf.train.latest_checkpoint(path)
  assert ckpt is not None
  print('Loading checkpoint ' + ckpt)
  saver.restore(sess, ckpt)
  return ckpt


tflex.state.noisy = False

# https://stackoverflow.com/a/18284900/9919772
try:
    from StringIO import StringIO as BytesIO ## for Python 2
except ImportError:
    from io import BytesIO ## for Python 3
#import IPython.display
import numpy as np

def get_grid_size(n):
  gw = 1
  gh = 1
  i = 0
  while gw*gh < n:
    if i % 2 == 0:
      gw += 1
    else:
      gh += 1
    i += 1
  return (gw, gh)

def gen_images(latents, outfile=None, display=False, labels=None, randomize_noise=False, is_validation=True, network=None, numpy=False):
  if network is None:
    network = Gs
  n = latents.shape[0]
  grid_size = get_grid_size(n)
  drange_net = [-1, 1]
  with tflex.device('/gpu:0'):
    result = network.run(latents, labels, is_validation=is_validation, randomize_noise=randomize_noise, minibatch_size=sched.minibatch_gpu)
    #if result.shape[1] > 3:
    #  final = result[:, 3, :, :]
    #else:
    #  final = None
    result = result[:, 0:3, :, :]
    img = misc.convert_to_pil_image(misc.create_image_grid(result, grid_size), drange_net)
    if outfile is not None:
      img.save(outfile)
    if display:
      f = BytesIO()
      img.save(f, 'png')
      IPython.display.display(IPython.display.Image(data=f.getvalue()))
  return result if numpy else img


def grab(name, postfix, i, n=1, latents=None, **kwargs):
  load_checkpoint('checkpoint/'+name)
  if latents is None:
    latents = rand_latent(n, seed=i)
  gw, gh = get_grid_size(latents.shape[0])
  outfile = 'checkpoint/%s_fakes_%dx%d_%04d_%s.png' % (name, gw, gh, i, postfix)
  if not os.path.isfile(outfile):
    return gen_images(latents, outfile=outfile, **kwargs)

def grab_grid(i, n=1, latents=None, outfile=None, **kwargs):
  if latents is None:
    latents = rand_latent(n, seed=i)
  gw, gh = get_grid_size(latents.shape[0])
  #outfile = 'checkpoint/%s_fakes_%dx%d_%04d_%s.png' % (name, gw, gh, i, postfix)
  return gen_images(latents, outfile=outfile, **kwargs)

import discord
import asyncio

def post_picture(channel_name, image, text=None, name='test', kind='png'):
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    client = discord.Client()
    token=discord_token

    async def send_picture(channel, image, kind='png', name='test', text=None):
        img = misc.convert_to_pil_image(image, [-1, 1])
        f = BytesIO()
        img.save(f, kind)
        f.seek(0)
        picture = discord.File(f)
        picture.filename = name + '.' + kind
        await channel.send(content=text, file=picture)

    @client.event
    async def on_ready():
        print('Logged on as {0}!'.format(client.user))
        try:
          channel = [x for x in list(client.get_all_channels()) if channel_name in x.name]
          assert len(channel) == 1
          channel = channel[0]
          print(channel)
          await send_picture(channel, image, kind=kind, name=name, text=text)
        finally:
          await client.logout()

    #@client.event
    #async def on_message(message):
    #    print('Message from {0.author}: {0.content}'.format(message))

    client.run(token)

all_labels = np.array([[1.0 if i == j else 0.0 for j in range(1000)] for i in range(1000)], dtype=np.float32)
n = grid_image_size
labels = [all_labels[i] for i in range(n)]
labels2 = [all_labels[0] for i in range(n)]

for ckpt in tf.train.checkpoints_iterator(model_dir, 1.0):
    print('posting ' + ckpt)
    saver.restore(sess, ckpt)
    seed = np.random.randint(10000)
    for i in range(seed,seed + count):
      print('------- %d -------' % i)
      result = grab_grid(i, n=n, labels=labels, numpy=True)
      post_picture(channel, misc.create_image_grid(result, get_grid_size(n)), "`" + ckpt + ' seed %d`' % i)


