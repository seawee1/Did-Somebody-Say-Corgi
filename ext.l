
(def image-to-discord-file (img name: name kind: kind)
  (set kind (or kind "jpg"))
  (set name (or name "test"))
  (let f (BytesIO)
    (f.write (image-to-bytes img))
    (f.seek 0)
    (let picture (discord.File f)
      (set picture.filename (+ name "." kind))
      picture)))

(def listtab (l)
  (with r (obj)
    (step (k v) l
      (if (number? k)
          (set (at r k) v)
        (set (get r k) v)))))

(def tablist (h)
  (with r ()
    (each (k v) h
      (add r (list k v)))))

(def assoc (key al)
  (let-when i (first (fn (x) (hd? x key)) al)
    (at al i)))

(def alref (al key)
  (let-when l (assoc key al)
    (at l 1)))

(def union (f xs ys)
  (join xs (rem (fn (y) (some (fn (_) (f _ y)) xs))
                ys)))

(define-global testify (x test)
  (if (function? x) x
      test
      (fn (y) (test y x))
    (fn (y) (= x y))))

(define-global find (x t)
  (let f (testify x)
    (each x t
      (let y (f x)
        (if y (return y))))))

(def saveable-vars (network)
  (map (fn ((k v)) (list (cat "G_synthesis/" k) v))
       (list x for x in (synthesis.trainables.items))))

;(go (chan*.send "hi" file: (image-to-discord-file img)))

(import tflex)

(define-symbol chan* tflex.message.channel)

(from training import misc)

(def rand-latent (seed shape)
  (let (shape (or shape '(1 512))
        seed (np.random.RandomState seed)
        latent (apply seed.randn shape))
    latent))

(from pprint import pprint)

(from io import StringIO)

(def bytes-to-discord-file (bs name)
  (let f (BytesIO)
    (f.write bs)
    (f.seek 0)
    (let f1 (discord.File f)
      (when (is? name)
        (set f1.filename name))
      f1)))

(def image-to-discord-file (img name: name kind: kind)
  (set kind (or kind "jpg"))
  (set name (or name "test"))
  (bytes-to-discord-file (image-to-bytes img) (+ name "." kind)))

(def pps (x)
  (let f (StringIO)
    (pprint x stream: f)
    (ellipsize (f.getvalue) 1190)))

(def netrun (network latent label |**kws|)
  (let latent (if (or (nil? latent) (isinstance latent int))
                  (rand-latent latent)
                  latent)
    (network.run latent label |**kws|)))

(defvar target-size* nil)

(import asyncio)

(def go (coro)
  (asyncio.run_coroutine_threadsafe coro (asyncio.get_event_loop)))

(def prfile (f text: text name: name chan: chan)
  (set chan (or chan chan*))
  (go (chan.send (or text "") file: f)))

(def numpy-to-image-grid (result drange-net: drange-net)
  (let (drange-net (or drange-net '(-1 1))
        n result.shape.0
        grid-size (get-grid-size n)
        x (misc.create-image-grid result grid-size)
        img (misc.convert-to-pil-image x drange-net))
    img))

(def primg (img text: text name: name)
  (let (img (if (numpy? img) (numpy-to-image-grid img) img)
        img (if target-size* (resize-image img target-size*) img)
        f (image-to-discord-file img))
    ;(set (get f 'filename) name)
    (go (chan*.send (or text "") file: f))))

(import os)
;(set (get os.environ 'TPU_NAME) "tpu-v3-32-euw4a-4")

(import dnnlib)
(import tflex)
(set me (dnnlib.EasyDict tflex.self))
(from training.networks_stylegan2 import *)
(from training import misc)

(me.init)

(defvar G (tflib.Network "G" num_channels: 3 resolution: 512 label_size: 0 func_name: "training.networks_stylegan2.G_main"))
(defvar Gs (G.clone "Gs"))

(defvar model-dir* nil)
(defvar model-step* 0)

(def checkpoint (path i network: network)
  (global model-dir*)
  (let (path (or path model-dir*)
        network (or network Gs))
    (set model-dir* path)
    (let path (if (is? i)
                  (os.path.join path (idx "model-{}.ckpt" i))
                  (or (tf.train.latest-checkpoint path) path))
      (me.load-checkpoint path var_list: network.trainables)
      path)))


(def grab (latent ckpt network: network label: label truncation_psi: psi)
  (let network (or network Gs)
    (when ckpt
      (me.load-checkpoint ckpt var_list: network.trainables))
    (primg (hd (netrun network latent label truncation_psi: (either psi 0.7))))))

(def latent-grid (seed n)
  (rand-latent (or n 9) seed: (or seed 0)))

(def o (seed n)
  (gen-images (rand-latent (or n 9) (or seed 0))))

(mac see args
  (let (args (keep (fn (x) (not (= x "seed"))) args)
       (path seed n offset) args
        a (or offset 0)
        b `(+ ,a ,(or n 9))
        n 100)
    `(do (checkpoint ,(escape path))
             (primg (gen-images (np.array (array (cut (rand-latent ,n seed: ,(or seed 0)) ,a ,b)))) text: ,(escape (cat path " seed " (str seed)))))))

(mac seevid (path _ seed x y dur fps)
  `(do (checkpoint ,(escape path))
       (interpv ,(or dur 2.0) ,(or fps 'nil) ,seed ,x ,y)))

(defvar latents*)

(def lerp (a b t)
  (+ a (* t (- b a))))

(def lerp-latent (a b t)
  (np.array (list (lerp a b t))))

(def latent-range (a b)
  (fn (t)
    (lerp-latent a b t)))

(def interpolator (seed x0 x1)
  (let (seed (or seed 0)
        latents (rand-latent 9 seed: seed)
        a (at latents (or x0 0))
        b (at latents (or x1 1)))
    (latent-range a b)))

(def i args
  (interp (apply interpolator args)))

(def time-percentage (duration f)
  (fn (t)
    (f (/ t duration))))

(def then-image (latent-maker)
  (fn (t)
    (let latent (latent-maker t)
      (image-to-numpy (gen-images latent)))))

(def to-numpy (x)
  (if (image? x) x (image-to-numpy x)))

(def pimg (x)
  (primg (to-numpy x)))

(def interp (f n)
  (let n (or n 10.0)
    (step t (np.arange 0.0 (+ n 1.0) 1.0)
      (pimg (gen-images (f (/ t n)))))))

(def i (seed a b n)
  (interp (interpolator seed a b) n))

(def prfile (f text: text chan: chan name: name)
  (set chan (or chan chan*))
  (when name
    (set f.filename name))
  (go (chan.send (or text "") file: f)))

(mac after (x rest: body)
  `(let ((ok v) (guard ,x))
     ,@body
     (if ok v (throw v))))

(import os)
(import tempfile)

|sys.path += ["/home/train/.local/lib/python3.6/site-packages"]|
(import moviepy)
(import moviepy.editor)

(unless (%in "gwern#1782" me.admins)
  (add me.admins "gwern#1782"))

(def videobytes (make-frame duration fps)
  (set duration (or duration 0.5))
  (set fps (or fps 10))
  (set vid (moviepy.editor.VideoClip make-frame duration: duration))
  (let ((fd path) (tempfile.mkstemp)
        path "test"
        path (+ path ".mp4"))
    (after (do (vid.write-videofile path fps: fps codec: mp4-codec bitrate: mp4-bitrate)
               ((idx (open path "rb") read)))
      ;(os.remove path)
      )))

(def interpv (duration fps rest: args)
  (set duration (either duration 0.5))
  (set fps (or fps 10))
  (set vid (videobytes (then-image (time-percentage duration (apply interpolator args))) duration fps))
  (set res (bytes-to-discord-file vid "interp.mp4"))
  (go (chan*.send file: res)))

(mac check (path)
  `(let s (me.shell '(gsutil-checkpoints ,path))
     (go (chan*.send file: (bytes-to-discord-file (s.encode "utf8") "checkpoints.txt")))))


|

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

def gen_images(latents, outfile=None, display=False, labels=None, randomize_noise=False, is_validation=True, network=None):
  if network is None:
    network = Gs
  n = latents.shape[0]
  grid_size = get_grid_size(n)
  drange_net = [-1, 1]
  with tflex.device('/gpu:0'):
    result = network.run(latents, labels, is_validation=is_validation, randomize_noise=randomize_noise, minibatch_size=sched.minibatch_gpu)
    img = misc.convert_to_pil_image(misc.create_image_grid(result, grid_size), drange_net)
    if outfile is not None:
      img.save(outfile)
    if display:
      f = BytesIO()
      img.save(f, 'png')
      IPython.display.display(IPython.display.Image(data=f.getvalue()))
  return result

import discord
import tensorflow as tf
grid_size = [1,1]
image_shrink = 1
image_zoom = 1
duration_sec = 5.0
smoothing_sec = 1.0
mp4_fps = 20
mp4_codec = 'libx264'
mp4_bitrate = '5M'
random_seed = 82
minibatch_size = 8

num_frames = int(np.rint(duration_sec * mp4_fps))
random_state = np.random.RandomState(random_seed)

# Generate latent vectors
shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
all_latents = random_state.randn(*shape).astype(np.float32)
import scipy
all_latents = scipy.ndimage.gaussian_filter(all_latents,
                [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
all_latents /= np.sqrt(np.mean(np.square(all_latents)))


def rand_latent(n, seed=None):
  if seed is not None:
    if seed < 0:
      seed = 2*32 - seed
    np.random.seed(seed)
  result = np.random.randn(n, *G.input_shape[1:])
  if seed is not None:
    np.random.seed()
  return result

sched = dnnlib.EasyDict()
sched.minibatch_gpu = 1


| 

nil

