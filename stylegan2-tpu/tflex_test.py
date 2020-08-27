import sys
import tflex
import time

def make_future(label, delay=0.0, deps=[], thunk=None, args=[]):
  def toplevel():
    if delay is not None and delay > 0.0:
      time.sleep(delay)
    nonlocal thunk, args
    if callable(thunk):
      thunk(*args)
    nonlocal label
    if callable(label):
      label = label()
    print('Future {}'.format(label))
  return tflex.defer(thunk=toplevel, dependencies=deps)

def warmup():
  make_future('warmup', 0.0).join()

def test1():
  a = make_future('A', 0.1)
  b = make_future('B', 0.0, deps=[a])
  c = make_future('C', 0.0, deps=[a, b])
  make_future('done', 0.0, deps=[a, b, c]).join()

def test2():
  a = make_future('A', 0.1)
  b = make_future('B', 0.0)
  c = make_future('C', 0.2)
  make_future('done', 0.0, deps=[a, b, c]).join()

def test3():
  tflex.local.foo = 'foo'
  make_future(lambda: tflex.local.foo).join()

def test4():
  tflex.local.foo = 'foo'
  def bar():
    tflex.local.foo = 'bar'
  make_future(lambda: tflex.local.foo, thunk=bar).join()
  make_future(lambda: tflex.local.foo).join()

def test5():
  tflex.local.foo = 'foo'
  def baz():
    tflex.local.foo = 'baz'
  def bar():
    make_future(lambda: tflex.local.foo, thunk=baz).join()
    tflex.local.foo = 'bar'
  make_future(lambda: tflex.local.foo, thunk=bar).join()
  make_future(lambda: tflex.local.foo).join()

def run_test(name, thunk, *args, **kws):
  print('Running {}'.format(name))
  sys.stdout.flush()
  start_time = time.time()
  result = thunk(*args, **kws)
  end_time = time.time()
  print('Finished {} in {:.2f}'.format(name, end_time - start_time))
  print('--------------------')

def run_tests():
  run_test('warmup', warmup)
  run_test('test1', test1)
  run_test('test2', test2)
  run_test('test3', test3)
  run_test('test4', test4)
  run_test('test5', test5)

if __name__ == '__main__':
  run_tests()

