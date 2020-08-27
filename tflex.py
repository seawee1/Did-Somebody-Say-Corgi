import tensorflow as tf
import numpy as np
from glob import glob
import os
import re
from tensorflow.python import pywrap_tensorflow
import tqdm
import h5py
import shutil
import tempfile
import traceback
import time
import threading

from tensorflow.python.framework import dtypes
from tensorflow.python.distribute.cluster_resolver import TPUClusterResolver as BaseTPUClusterResolver
from tensorflow.python.training import server_lib
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.contrib import tpu

class _DefaultState(threading.local):
  def __init__(self, **kws):
    super(_DefaultState, self).__init__()
    for k, v in kws.items():
      setattr(self, k, v)

  def save(self):
    return [(k, v) for k, v in self.__dict__.items()]

  def restore(self, state):
    for k, v in state:
      setattr(self, k, v)

local = _DefaultState()
lock = threading.RLock()

def with_defaults(thunk):
  with lock:
    state = local.save()
    session = tf.get_default_session() or get_default_session()
    graph = tf.get_default_graph() or get_default_graph()
  def f(*args, **kws):
    with lock:
      local.restore(state)
    lock.acquire()
    with session.as_default() if session else nullcontext():
      with graph.as_default() if graph else nullcontext():
        lock.release()
        result = thunk(*args, **kws)
        lock.acquire()
    lock.release()
    return result
  return f

def get_default(name, required=True):
  with lock:
    value = getattr(local, name) if hasattr(local, name) else None
  if required:
    assert value is not None
  return value

def set_default(name, value):
  with lock:
    setattr(local, name, value)

def ensure_default(name, value):
  with lock:
    current = get_default(name, required=False)
    if current is None:
      set_default(name, value)
    return value

def get_default_session(required=False):
  return get_default('session', required=required)

def get_default_graph(required=False):
  return get_default('graph', required=required)

class Future(object):
  def __init__(self, dependencies, thunk, *args, **kws):
    if isinstance(dependencies, Future):
      dependencies = [dependencies]
    self.dependencies = [defer(_) if callable(_) else _ for _ in dependencies]
    if thunk is None:
      thunk = lambda: None
    self.thunk = thunk
    self.args = args
    self.kws = kws
    self.result = None
    self.complete = False
    self.thread = None
    self.daemon = True
    self.error = None
  def run(self):
    try:
      self.result = self.thunk(*self.args, **self.kws)
    except Exception as e:
      traceback.print_exc()
      self.error = e
    self.complete = True
  def run_async(self):
    assert self.thread is None
    def thunk():
      [_.join() for _ in self.dependencies]
      self.run()
    self.thread = threading.Thread(target=with_defaults(thunk), daemon=self.daemon)
    self.thread.start()
  def join(self):
    if not self.complete:
      assert self.thread
      while not self.complete:
        time.sleep(1.0)
    return self.result

def defer(thunk, *args, **kws):
  dependencies = []
  if 'dependencies' in kws:
    dependencies = kws.pop('dependencies')
  future = Future(dependencies=dependencies, thunk=thunk, *args, **kws)
  future.run_async()
  return future

def parallelize(xs, thunk, *args, daemon=True):
  threads = []
  for x in xs:
    thread = threading.Thread(target=with_defaults(thunk), args=(x, *args), daemon=daemon)
    thread.start()
    threads.append(thread)
  return threads

def parallelize_verbose(label, xs, thunk, *args, daemon=True):
  xs = [x for x in xs]
  with tqdm.tqdm(total=len(xs)) as pbar:
    pbar.set_description(label)
    def run(*args, **kws):
      try:
        return thunk(*args, **kws)
      finally:
        pbar.update(1)
    return parallelize(xs, run, *args, daemon=daemon)

def parallelize_verbose(label, xs, thunk, *args, daemon=True, synchronous=False):
  xs = [x for x in xs]
  if synchronous:
    for i in tqdm.trange(len(xs), desc=label):
      x = xs[i]
      thunk(x, *args)
  else:
    with tqdm.tqdm(total=len(xs)) as pbar:
      pbar.set_description(label)
      threads = parallelize(xs, thunk, *args, daemon=daemon)
      while len(threads) > 0:
        for i in range(len(threads)):
          if not threads[i].is_alive():
            pbar.update(1)
            threads.remove(threads[i])
            break
        time.sleep(0.1)

# http://stackoverflow.com/questions/1624883/alternative-way-to-split-a-list-into-groups-of-n
import itertools
def group(n, iterable, fillvalue=None):
    "group(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def tuples(*args, **kws):
  return [x for x in group(*args, **kws)]

class Namespace(object):
  pass

if 'state' not in globals():
  state = Namespace()

if not hasattr(state, 'noisy'):
  state.noisy = 'NOISY' in os.environ

if not hasattr(state, 'debug'):
  state.debug = 'DEBUG' in os.environ

if not hasattr(state, 'noisy_backtrace'):
  state.noisy_backtrace = 'NOISY_BACKTRACE' in os.environ

if not hasattr(state, 'break_next_run'):
  state.break_next_run = False

def reroute(addr, host=None):
  if host is None or host is False:
    return addr
  if addr.startswith('grpc://'):
    return 'grpc://' + reroute(addr[len('grpc://'):], host=host)
  if not re.match('[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+[:]8470', addr):
    return addr
  if not addr.endswith(':8470'):
    return addr
  a, b, c, d = [int(x) for x in addr.split(':')[0].split('.')]
  if a == 10 and b in [48, 49]:
    assert (d == 2)
    port = b * 1000 + c
  elif a == 10 and b in range(2, 66) and c == 0:
    port = b * 1000 + d
  else:
    return addr
  return host + ':' + str(port)


class TPUClusterResolver(BaseTPUClusterResolver):
  def __init__(self, *args, host=None, **kws):
    super(TPUClusterResolver, self).__init__(*args, **kws)
    if host is None:
      if 'TPU_HOST' in os.environ:
        host = os.environ['TPU_HOST']
    self._host = host

  def master(self, *args, **kws):
    ip = super(TPUClusterResolver, self).master(*args, **kws)
    return reroute(ip, host=self._host)

  def cluster_spec(self):
    spec = super(TPUClusterResolver, self).cluster_spec()
    r = dict()
    for k, v in spec.as_dict().items():
      r[k] = [reroute(ip, host=self._host) for ip in v]
    return server_lib.ClusterSpec(r)

def init_tpu(name, host=None, timeout_in_ms=600 * 60 * 1000):
  tpu_init = [tpu.initialize_system()]
  cluster_resolver = TPUClusterResolver(name, host=host)
  config = tf.ConfigProto(operation_timeout_in_ms=timeout_in_ms,
                          graph_options=tf.GraphOptions(
                            rewrite_options=rewriter_config_pb2.RewriterConfig(
                              disable_meta_optimizer=True)),
                          isolate_session_state=True)
  cluster_spec = cluster_resolver.cluster_spec()
  if cluster_spec:
    config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
  init_sess = tf.Session(cluster_resolver.get_master(), config=config)
  init_sess.run(tpu_init)
  return init_sess, cluster_resolver

def get_session(session=None):
  if session is None:
    session = get_default_session()
  return session

def get_devices(session=None):
  session = get_session(session)
  if hasattr(session, '_cached_devices'):
    devices = session._cached_devices
  else:
    devices = session._cached_devices = session.list_devices()
  return devices

def has_gpu(session=None):
  session = get_session(session)
  if hasattr(session, '_has_gpu'):
    result = session._has_gpu
  else:
    devices = get_devices(session=session)
    result = session._has_gpu = len([x for x in devices if ':GPU:' in x.name]) > 0
  return result

def has_tpu(session=None):
  session = get_session(session)
  if hasattr(session, '_has_tpu'):
    result = session._has_tpu
  else:
    devices = get_devices(session=session)
    result = session._has_tpu = len([x for x in devices if ':TPU:' in x.name]) > 0
  return result

def get_cores_from_devices(devices):
  cores = [x for x in devices if ':TPU:' in x.name]
  if len(cores) <= 0:
    cores = [x for x in devices if ':GPU:' in x.name]
  if len(cores) <= 0:
    cores = [x for x in devices if ':CPU:' in x.name]
  return cores

def get_cores(session=None, devices=None):
  if devices is None:
    devices = get_devices(session=session)
  return get_cores_from_devices(devices)

def get_cpus(session=None, devices=None):
  if devices is None:
    devices = get_devices(session=session)
  cpus = [x for x in devices if ':CPU:' in x.name]
  return cpus

def get_tpu_resolver(tpu_name='auto'):
  # Get the TPU's location
  if tpu_name != 'auto':
    return TPUClusterResolver(tpu_name)
  elif 'COLAB_TPU_ADDR' in os.environ:
    return TPUClusterResolver()
  elif 'TPU_NAME' in os.environ:
    return TPUClusterResolver(os.environ['TPU_NAME'])

def pretty(x, ellipsize=120):
  r = str(x)
  if len(r) > ellipsize:
    return r[0:ellipsize - 3] + '...'
  return r

def print_backtrace():
  try:
    raise Exception("Printing traceback...")
  except:
    import traceback
    traceback.print_exc()

class Session(tf.Session):
  def __init__(self, target='auto', graph=None, config=None, init_tpu=False, id=None):
    if config is None:
      timeout_in_ms = int(os.environ['TIMEOUT_IN_MS']) if 'TIMEOUT_IN_MS' in os.environ else 10 * 60 * 1000
      config = tf.ConfigProto(operation_timeout_in_ms=timeout_in_ms,
                              graph_options=tf.GraphOptions(
                                rewrite_options=rewriter_config_pb2.RewriterConfig(
                                  disable_meta_optimizer=True)),
                              isolate_session_state=True)
    config.isolate_session_state = True
    resolver = get_tpu_resolver(target)
    if resolver is not None:
      target = resolver.get_master()
      cluster_spec = resolver.cluster_spec()
      if cluster_spec:
        config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    elif target == 'auto':
      target = None
    super().__init__(target, graph=graph, config=config)
    self.id = id
    self._tflex_resolver = resolver
    self._tflex_target = target
    self._tflex_config = config
    ensure_default('session', self)
    ensure_default('devices', self.list_devices())
    ensure_default('graph', self.graph)

  @property
  def _spec(self):
    return '#%d' % self.id if self.id is not None else ''

  def ensure(self):
    if self.init_tpu:
      print(self._spec, "Initializing TPU...")
      #sess.run(tpu.initialize_system())
      init_tpu(session=self, timeout_in_ms=20000)
      self.init_tpu = None

  def run(self, *args, **kws):
    if state.break_next_run:
      import pdb; pdb.set_trace()
    if state.debug:
      check_commands()
    if state.noisy:
      print(self._spec, 'Session.run', *[pretty(x) for x in args], *[pretty(k)+'='+pretty(v) for k, v in kws.items()])
      if state.noisy_backtrace:
        print_backtrace()
    start = time.time()
    result = super(Session, self).run(*args, **kws)
    elapsed = time.time() - start
    if state.noisy:
      print(self._spec, 'Session.run (finished in %.2fs)' % elapsed, pretty(result), *[pretty(x) for x in args], *[pretty(k)+'='+pretty(v) for k, v in kws.items()])
      if state.noisy_backtrace:
        print_backtrace()
    return result


def split_by_params(vs, n=20e6, f=None):
  if f is None:
    f = lambda x: np.prod(x.shape.as_list())
  i = 0
  xs = []
  for variable in vs:
    xs.append(variable)
    count = f(variable)
    i += count
    if i >= n:
      yield xs
      xs = []
      i = 0
  yield xs

def latest_checkpoint(checkpoint_dir, latest_filename=None):
  paths = [x for x in glob(os.path.join(checkpoint_dir, 'model-*.*')) if not x.endswith(".tmp")]
  ctrs = np.array([[int(y) for y in re.findall(r'model-([0-9]+)(?:-[0-9]+)?[.](?:npy|hdf5)', x)] for x in paths]).flatten()
  if len(ctrs) <= 0:
    ckpt = tf.train.latest_checkpoint(checkpoint_dir, latest_filename=latest_filename)
    return ckpt
  ctr = ctrs.max()
  return os.path.join(checkpoint_dir, 'model-{}').format(ctr)

def truncate_value(variable, value, reshape=True):
  if not reshape:
    return value
  shape = variable.shape.as_list()
  params = np.prod(shape)
  params2 = np.prod(value.shape)
  if params == params2:
    return value
  print('Truncating {} from shape {} to shape {}'.format(variable.name, value.shape, shape))
  value = np.array(value)
  value = value.reshape([-1])
  value = value[0:params]
  value = value.reshape(shape)
  return value

from tensorflow.core.protobuf import config_pb2

def initialize_tpu(session=None, timeout_in_ms=None):
  session = session or get_default_session()
  with session.as_default():
    op = tpu.initialize_system()
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(op, options=options)

def load(variable, value, session=None, timeout_in_ms=None):
  session = session or get_default_session()
  ops = variable.initializer
  vals = dict([(variable.initializer.inputs[1], value)])
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(ops, vals, options=options)

def eval(variable, session=None, timeout_in_ms=None):
  session = session or get_default_session()
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(variable, options=options)

def grab_values(variables, reader, reshape=False):
  for variable in variables:
    name = variable_name(variable).split(':')[0]
    value = reader.get_tensor(name)
    value = truncate_value(variable, value, reshape=reshape)
    yield variable, value

def assign_values(variables, values, session=None, timeout_in_ms=60000):
  session = session or get_default_session()
  variables = [x for x in variables]
  values = [x for x in values]
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value.value() if isinstance(value, tf.Variable) else value) for x, value in zip(variables, values)]) # TODO: bfloat16 support
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  session.run(ops, vals, options=options)

def load_snapshot(ckpt, session=None, var_list=None, reshape=False):
  session = session or get_default_session()
  reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
  vs = var_list or tf.trainable_variables()
  for variables in tqdm.tqdm(list(split_by_params(vs))):
    values = [value for variable, value in grab_values(variables, reader, reshape=reshape)]
    assign_values(variables, values, session=session)

def get_variable(name, var_list=None):
  name, num = name.split(':') if ':' in name else (name, '0')
  num = int(num)
  name = os.path.join(tf.get_variable_scope().name, name)
  vs = var_list or tf.trainable_variables()
  for x in vs:
      if x.name.startswith(name + ':%d' % num):
          return x

def load_weights(ckpt, session=None, var_list=None, reshape=False):
  session = session or get_default_session()
  vs = var_list or tf.trainable_variables()
  files = list(sorted(glob(ckpt + '-*.npy')))
  for out in tqdm.tqdm(files):
    for name, value in np.load(out, allow_pickle=True):
      variable = get_variable(name)
      if variable is None:
        print('Warning: variable %s not loaded' % name)
      else:
        value = truncate_value(variable, value, reshape=reshape)
        variable.load(value, session)

def load_variables(ckpt, session=None, var_list=None, reshape=False):
  session = session or get_default_session()
  vs = var_list or tf.trainable_variables()
  with h5py.File(ckpt, "r") as f:
    for variables in tqdm.tqdm(list(split_by_params(vs))):
      values = [truncate_value(x, f[variable_name(x)], reshape=reshape)  for x in variables]
      assign_values(variables, values, session=session)

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

state.cache_ops = {}

def cast_variables(variables, graph=None, cache_ops=None):
  if graph is None:
    graph = get_default_graph()
  if cache_ops is None:
    cache_ops = state.cache_ops
  if graph not in cache_ops:
    cache_ops[graph] = {}
  cache = cache_ops[graph]
  ops = []
  for variable in variables:
    if variable in cache:
      op = cache[variable]
    elif variable.dtype == dtypes.bfloat16_ref or variable.dtype == tf.bfloat16:
      op = tf.cast(variable, tf.float32)
    else:
      op = variable
    cache[variable] = op
    ops.append(op)
  return ops

import re

def variable_name(variable):
  if re.match(r'core[0-9]+/', variable.name):
    return variable.name.split('/', 1)[-1]
  return variable.name

def save_variables(ckpt, session=None, var_list=None):
    session = session or get_default_session()
    vs = var_list or tf.trainable_variables()
    maketree(os.path.dirname(ckpt))
    fname = ckpt+'.tmp'
    with h5py.File(fname, "w") as f:
      for variables in tqdm.tqdm(list(split_by_params(vs))):
        ops = cast_variables(variables)
        values = session.run(ops)
        for value, variable in zip(values, variables):
          name = variable_name(variable)
          shape = variable.shape.as_list()
          dtype = variable.dtype
          dset = f.create_dataset(name, shape, dtype=np.float32)
          dset[:] = value
    print('Writing snapshot %s' % ckpt)
    os.rename(ckpt+'.tmp', ckpt)

def fetch_variables(session=None, var_list=None):
    session = session or get_default_session()
    vs = var_list or tf.trainable_variables()
    for variables in tqdm.tqdm(list(split_by_params(vs))):
      values = session.run(variables)
      yield variables, values

def partition_variables(session=None, var_list=None):
    session = session or get_default_session()
    vs = var_list or tf.trainable_variables()
    for variables in tqdm.tqdm(list(split_by_params(vs))):
      yield variables

class Saver(object):
  def __init__(
    self,
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None):
    self.var_list = var_list
    self.reshape = reshape
    self.sharded = sharded
    self.max_to_keep = max_to_keep
    self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self.name = name
    self.restore_sequentially = restore_sequentially
    self.saver_def = saver_def
    self.builder = builder
    self.defer_build = defer_build
    self.allow_empty = allow_empty
    self.write_version = write_version
    self.pad_step_number = pad_step_number
    self.save_relative_paths = save_relative_paths
    self.filename = filename
    self.checkpoints = []

  def restore(self, sess, save_path):
    if save_path.endswith('.ckpt'):
      load_snapshot(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif save_path.endswith('.hdf5'):
      load_variables(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif os.path.exists(save_path + '.npy') or os.path.exists(save_path + '-0.npy'):
      load_weights(save_path, session=sess, var_list=self.var_list, reshape=self.reshape)
    elif os.path.exists(save_path + '.hdf5'):
      load_variables(save_path + '.hdf5', session=sess, var_list=self.var_list, reshape=self.reshape)
    else:
      raise Exception("Can't load checkpoint %s" % save_path)

  def save(self,
        sess,
        save_path,
        global_step=None,
        latest_filename=None,
        meta_graph_suffix="meta",
        write_meta_graph=True,
        write_state=True,
        strip_default_attrs=False,
        save_debug_info=False):
    if global_step is not None:
      name = '%s-%d.hdf5' % (save_path, global_step)
    else:
      name = '%s.hdf5' % save_path
    save_variables(name, session=sess, var_list=self.var_list)
    self.checkpoints.append(name)
    if self.max_to_keep > 0:
      while len(self.checkpoints) > self.max_to_keep:
        fname = self.checkpoints[0]
        if fname != name:
          print('Truncating %s' % fname)
          try:
            with open(fname, "wb") as f:
              pass
          except:
            print('Failed to truncate %s' % fname)
        self.checkpoints = self.checkpoints[1:]

  def fetch(self, sess, var_list=None):
    if var_list == None:
      var_list = self.var_list
    for variables, values in fetch_variables(session=sess, var_list=var_list):
      yield variables, values

  def variables(self, sess, var_list=None):
    if var_list == None:
      var_list = self.var_list
    for variables in partition_variables(session=sess, var_list=var_list):
      yield variables

  def assign(self, sess, variables, values):
    return assign_values(variables, values, session=sess)

class Commands(object):
  def __init__(self, path='commands'):
    self.path = path
    self.commands = []
    self.args = []
    self.keys = {}
    self.frozen = False

  def has(self, name, **keys):
    if 'action' in keys:
      action = keys.pop('action')
      for name1, action1 in self.commands:
        if name == name1 and action1 == action:
          return True
    else:
      for name1, action1 in self.commands:
        if name == name1:
          return True
    return False

  def add(self, name, action=None):
    if not self.has(name=name, action=action):
      self.commands.append((name, action))
      full = self.full_path(name)
      maketree(full)

  def full_path(self, name):
    return os.path.join(self.path, name)

  def check(self, *args, **keys):
    if not self.frozen:
      heartbeat()
    ops = []
    seen = set()
    for name, action in self.commands:
      full = self.full_path(name)
      if not os.path.isdir(full):
        if name not in seen:
          seen.add(name)
          ops.append(name)
    for op in ops:
      self.run(op, *args, **keys)
    return ops

  def run(self, op):
    ran = False
    for name, action in self.commands:
      if name == op:
        print('Running command', name, action)
        if not ran:
          full = self.full_path(op)
          maketree(full)
          ran = True
        if action:
          action()
    if not ran:
      raise Exception('Commands.execute failed: no such command: {}'.format(op))
  
  def run_with_args(self, op, *args, **keys):
    with CommandArgs(*args, **keys):
      return self.run(op)

commander = None

def commands(**keys):
  global commander
  if commander is None:
    commander = Commands()
  cmds = keys.pop('commands') if 'commands' in keys else None
  if cmds is not None:
    for cmd in cmds:
      action = None
      if isinstance(cmd, str):
        name = cmd
      elif len(cmd) >= 2:
        name, action = cmd
      elif len(cmd) >= 1:
        name = cmd[0]
      else:
        continue
      commander.add(name=name, action=action)
  return commander

class CommandArgs(object):
  def __init__(self, *args, **keys):
    self.args = list(args)
    self.keys = keys.copy()
    self.cmdr = commands()

  def __enter__(self):
    self.args_prev = self.cmdr.args
    self.keys_prev = self.cmdr.keys
    self.cmdr.args = self.args
    self.cmdr.keys = self.keys

  def __exit__(self, *excinfo):
    self.cmdr.args = self.args_prev
    self.cmdr.keys = self.keys_prev

def check_commands():
  try:
    cmdr = commands()
    return cmdr.check()
  except:
    traceback.print_exc()

def check_commands_with_args(*args, **keys):
  try:
    cmdr = commands()
    with CommandArgs(*args, **keys):
      return cmdr.check()
  except:
    traceback.print_exc()

def add_command(name, action=None, **keys):
  cmdr = commands()
  return cmdr.add(name=name, action=action)

def register_command(*args, **keys):
  fn = args[0]
  if isinstance(fn, str):
    add_command(fn)
  else:
    name = fn.__qualname__
    name = name.replace('.<locals>.', '_command_')
    if name.endswith('_command_save'):
      name = 'save'
    name = name.replace('___', '/')
    action = fn
    print(name, action)
    add_command(name, action)
  return fn

def has_command(name):
  cmdr = commands()
  return cmdr.has(name)

def run_command(command_name):
  cmdr = commands()
  return cmdr.run(command_name)

def run_command_with_args(command_name, *args, **keys):
  cmdr = commands()
  return cmdr.run_with_args(command_name, *args, **keys)

def command_arg(x, unset=None):
  cmdr = commands()
  if isinstance(x, int):
    try:
      return cmdr.args[x]
    except:
      return unset
  else:
    if x in cmdr.keys:
      return cmdr.keys[x]
    return unset

def command_args():
  cmdr = commands()
  return cmdr.args, cmdr.keys

@register_command
def attach_debugger():
  import pdb
  pdb.set_trace()

from pprint import pprint

@register_command
def print_status():
  args, props = command_args()
  for k, v in enumerate(args):
    pprint(v)
  for k, v in props.items():
    pprint({k: v})


#
# return current UTC timestamp.
#
def utc():
    from datetime import datetime
    d = datetime.utcnow()
    import calendar
    return calendar.timegm(d.utctimetuple())

def heartbeat():
  pongfile=os.environ['PONG'] if 'PONG' in os.environ else 'pong.txt'
  with open(pongfile, "a+") as f:
    nonce = os.urandom(8).hex()
    now=utc()
    out="pid{}_time{}_nonce{}\n".format(os.getpid(), now, nonce)
    #print("PONG! Writing {} to {}".format(out, pongfile))
    f.write(out)
    f.flush()

import time

@register_command
def freeze_forever():
  cmdr = commands()
  if cmdr.frozen:
    print("Already frozen.")
    return
  prev = cmdr.frozen
  cmdr.frozen = True
  print('Simulating a freeze; going into an infinite loop:')
  prev=time.time()
  try:
    while not should_quit():
      elapsed=time.time() - prev
      print('Frozen for {}s'.format(elapsed))
      time.sleep(1)
      check_commands()
  finally:
    cmdr.frozen = prev

_quit = False

import sys

@register_command
def quit():
  global _quit
  if _quit:
    print("Failed to quit; running sys.exit(1)")
    sys.exit(1)
  else:
    print("Quitting...")
    _quit = True

def should_quit():
  return _quit

@register_command
def save_and_quit():
  global _quit
  if has_command('save'):
    print("Saving...")
    run_command('save')
  quit()

@register_command
def throw_exception():
  raise Exception("This exception should be caught and logged by the tflex command system")


import tensorflow as tf
from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def set_override_device(value, session=None):
  session = get_session(session)
  session._override_device = value
  return value

def has_override_device(session=None):
  session = get_session(session)
  return hasattr(session, '_override_device')

def get_override_device(session=None):
  session = get_session(session)
  if hasattr(session, '_override_device'):
    return session._override_device

def set_override_cores(value, session=None):
  session = get_session(session)
  session._override_cores = value
  return value

def has_override_cores(session=None):
  session = get_session(session)
  return hasattr(session, '_override_cores')

def get_override_cores(session=None):
  session = get_session(session)
  if hasattr(session, '_override_cores'):
    return session._override_cores

def device_for_tpu_core(task=0, core=0, job_name="tpu_worker"):
  return "/job:%s/task:%d/device:TPU_REPLICATED_CORE:%d" % (job_name, task, core)

def device(name=''):
  if has_override_device():
    return nullcontext()
  if has_override_cores():
    if name is None:
      return tf.device(name)
    if name.startswith('/gpu:'):
      i = int(name.split(':', 1)[-1])
      return tf.device(get_cores()[i].name)
    if name.startswith('/tpu:'):
      i = int(name.split(':', 1)[-1])
      return tf.device(device_for_tpu_core(core=i))
    if name.startswith('/cpu:'):
      i = int(name.split(':', 1)[-1])
      return tf.device(get_cpus()[i].name)
    return nullcontext()
  if name is None:
    return tf.device(None)
  if 'gpu' in name:
    if has_gpu():
      return tf.device(name)
  if 'cpu' in name:
    return tf.device(name)
  return nullcontext()

def tuples(l, n=2):
  r = []
  for i in range(0, len(l), n):
    r.append(l[i:i+n])
  return r

import hashlib

def sha256hex(x):
  if isinstance(x, str):
    x = x.encode('utf8')
  return hashlib.sha224(x).hexdigest()

def sha256label(x):
  return [int(x, 16) / 255 * 2 - 1 for x in tuples(sha256hex(x))]

