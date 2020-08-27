import os
import bz2
import PIL.Image
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.utils import get_file
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.backend as K
import traceback

def load_images(images_list, image_size=256):
    loaded_images = list()
    for img_path in images_list:
      img = PIL.Image.open(img_path).convert('RGB').resize((image_size,image_size),PIL.Image.LANCZOS)
      img = np.array(img)
      img = np.expand_dims(img, 0)
      loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    return loaded_images

def tf_custom_l1_loss(img1,img2):
  return tf.math.reduce_mean(tf.math.abs(img2-img1), axis=None)

def tf_custom_logcosh_loss(img1,img2):
  return tf.math.reduce_mean(tf.keras.losses.logcosh(img1,img2))

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class PerceptualModel:
    def __init__(self, args, batch_size=1, perc_model=None, sess=None, optimizer=None):
        self.sess = tf.get_default_session() if sess is None else sess
        self.optimizer = optimizer
        K.set_session(self.sess)
        self.epsilon = 0.00000001
        self.lr = args.lr
        self.decay_rate = args.decay_rate
        self.decay_steps = args.decay_steps
        self.img_size = args.image_size
        self.layer = args.use_vgg_layer
        self.vgg_loss = args.use_vgg_loss
        self.face_mask = args.face_mask
        self.use_grabcut = args.use_grabcut
        self.scale_mask = args.scale_mask
        self.mask_dir = args.mask_dir
        if (self.layer <= 0 or self.vgg_loss <= self.epsilon):
            self.vgg_loss = None
        self.pixel_loss = args.use_pixel_loss
        if (self.pixel_loss <= self.epsilon):
            self.pixel_loss = None
        self.mssim_loss = args.use_mssim_loss
        if (self.mssim_loss <= self.epsilon):
            self.mssim_loss = None
        self.lpips_loss = args.use_lpips_loss
        if (self.lpips_loss <= self.epsilon):
            self.lpips_loss = None
        self.l1_penalty = args.use_l1_penalty
        if (self.l1_penalty <= self.epsilon):
            self.l1_penalty = None
        self.batch_size = batch_size
        if perc_model is not None and self.lpips_loss is not None:
            self.perc_model = perc_model
        else:
            self.perc_model = None
        self.ref_img = None
        self.ref_weight = None
        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

        if self.face_mask:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
            landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                                    LANDMARKS_MODEL_URL, cache_subdir='temp'))
            self.predictor = dlib.shape_predictor(landmarks_model_path)

    def compare_images(self,img1,img2):
        if self.perc_model is not None:
            return self.perc_model.get_output_for(tf.transpose(img1, perm=[0,3,2,1]), tf.transpose(img2, perm=[0,3,2,1]))
        return 0

    def add_placeholder(self, var_name):
        var_val = getattr(self, var_name)
        setattr(self, var_name + "_placeholder", tf.placeholder(var_val.dtype, shape=var_val.get_shape()))
        setattr(self, var_name + "_op", var_val.assign(getattr(self, var_name + "_placeholder")))

    def assign_placeholder(self, var_name, var_val):
        self.sess.run(getattr(self, var_name + "_op"), {getattr(self, var_name + "_placeholder"): var_val})

    def build_perceptual_model(self, generator):
        self.generator = generator
        # Learning rate
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self._incremented_global_step = tf.assign_add(self._global_step, 1)
        self._reset_global_step = tf.assign(self._global_step, 0)
        self.learning_rate = tf.train.exponential_decay(self.lr, self._incremented_global_step,
                self.decay_steps, self.decay_rate, staircase=True)
        self.sess.run([self._reset_global_step])

        self.generated_image_tensor = self.generator.generated_image
        self.generated_image = tf.image.resize_nearest_neighbor(self.generated_image_tensor,
                                                                  (self.img_size, self.img_size), align_corners=True)

        self.ref_img = tf.get_variable('ref_img', shape=self.generated_image.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
        self.ref_weight = tf.get_variable('ref_weight', shape=self.generated_image.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
        self.add_placeholder("ref_img")
        self.add_placeholder("ref_weight")

        if (self.vgg_loss is not None):
            self.vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
            self.perceptual_model = Model(self.vgg16.input, self.vgg16.layers[self.layer].output)
            self.generated_img_features = self.perceptual_model(preprocess_input(self.ref_weight * self.generated_image))
            self.ref_img_features = tf.get_variable('ref_img_features', shape=self.generated_img_features.shape,
                                                dtype='float32', initializer=tf.initializers.zeros())
            self.features_weight = tf.get_variable('features_weight', shape=self.generated_img_features.shape,
                                               dtype='float32', initializer=tf.initializers.zeros())
            self.sess.run([self.features_weight.initializer, self.features_weight.initializer])
            self.add_placeholder("ref_img_features")
            self.add_placeholder("features_weight")

        self.loss = 0
        # L1 loss on VGG16 features
        if (self.vgg_loss is not None):
            self.loss += self.vgg_loss * tf_custom_l1_loss(self.features_weight * self.ref_img_features, self.features_weight * self.generated_img_features)
        # + logcosh loss on image pixels
        if (self.pixel_loss is not None):
            self.loss += self.pixel_loss * tf_custom_logcosh_loss(self.ref_weight * self.ref_img, self.ref_weight * self.generated_image)
        # + MS-SIM loss on image pixels
        if (self.mssim_loss is not None):
            self.loss += self.mssim_loss * tf.math.reduce_mean(1-tf.image.ssim_multiscale(self.ref_weight * self.ref_img, self.ref_weight * self.generated_image, 1))
        # + extra perceptual loss on image pixels
        if self.perc_model is not None and self.lpips_loss is not None:
            self.loss += self.lpips_loss * tf.math.reduce_mean(self.compare_images(self.ref_weight * self.ref_img, self.ref_weight * self.generated_image))
        # + L1 penalty on dlatent weights
        if self.l1_penalty is not None:
            self.loss += self.l1_penalty * 512 * tf.math.reduce_mean(tf.math.abs(self.generator.dlatent_variable[:,0]-self.generator.get_dlatent_avg()))

        vars_to_optimize = self.generator.dlatent_variable
        self.vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        if self.optimizer is not None:
            self.min_op = self.optimizer.minimize(self.loss, var_list=[self.vars_to_optimize])
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.min_op = self.optimizer.minimize(self.loss, var_list=[self.vars_to_optimize])
            self.sess.run(tf.variables_initializer(self.optimizer.variables()))
            self.sess.run(self._reset_global_step)

    def generate_face_mask(self, im):
        from imutils import face_utils
        import cv2
        rects = self.detector(im, 1)
        # loop over the face detections
        for (j, rect) in enumerate(rects):
            """
            Determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            """
            shape = self.predictor(im, rect)
            shape = face_utils.shape_to_np(shape)

            # we extract the face
            vertices = cv2.convexHull(shape)
            mask = np.zeros(im.shape[:2],np.uint8)
            cv2.fillConvexPoly(mask, vertices, 1)
            if self.use_grabcut:
                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                rect = (0,0,im.shape[1],im.shape[2])
                (x,y),radius = cv2.minEnclosingCircle(vertices)
                center = (int(x),int(y))
                radius = int(radius*self.scale_mask)
                mask = cv2.circle(mask,center,radius,cv2.GC_PR_FGD,-1)
                cv2.fillConvexPoly(mask, vertices, cv2.GC_FGD)
                cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
                mask = np.where((mask==2)|(mask==0),0,1)
            return mask

    def load_images(self, images_list):
        assert (len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        return loaded_image

    def set_reference_images(self, images_list):
        return self.set_reference_image_data(self.load_images(images_list), images_list)

    def set_reference_image_data(self, loaded_image, images_list=None):
        if len(loaded_image.shape) < 4:
            loaded_image = loaded_image.reshape([1] + list(loaded_image.shape))
        self.loaded_image = loaded_image
        self.images_list = images_list
        self.image_features = None
        if self.perceptual_model is not None:
            self.image_features = self.perceptual_model.predict_on_batch(preprocess_input(self.loaded_image))
            self.weight_mask = np.ones(self.features_weight.shape)

        if self.face_mask and self.images_list is not None:
            self.image_mask = np.zeros(self.ref_weight.shape)
            for (i, im) in enumerate(self.loaded_image):
                try:
                    _, img_name = os.path.split(self.images_list[i])
                    mask_img = os.path.join(self.mask_dir, f'{img_name}')
                    if (os.path.isfile(mask_img)):
                        print("Loading mask " + mask_img)
                        imask = PIL.Image.open(mask_img).convert('L')
                        mask = np.array(imask)/255
                        mask = np.expand_dims(mask,axis=-1)
                    else:
                        mask = self.generate_face_mask(im)
                        imask = (255*mask).astype('uint8')
                        imask = PIL.Image.fromarray(imask, 'L')
                        print("Saving mask " + mask_img)
                        imask.save(mask_img, 'PNG')
                        mask = np.expand_dims(mask,axis=-1)
                    mask = np.ones(im.shape,np.float32) * mask
                except Exception as e:
                    print("Exception in mask handling for " + mask_img)
                    traceback.print_exc()
                    mask = np.ones(im.shape[:2],np.uint8)
                    mask = np.ones(im.shape,np.float32) * np.expand_dims(mask,axis=-1)
                self.image_mask[i] = mask
        else:
            self.image_mask = np.ones(self.ref_weight.shape)

        image_count = self.loaded_image.shape[0]
        if image_count != self.batch_size:
            if self.image_features is not None:
                features_space = list(self.features_weight.shape[1:])
                existing_features_shape = [image_count] + features_space
                empty_features_shape = [self.batch_size - image_count] + features_space
                existing_examples = np.ones(shape=existing_features_shape)
                empty_examples = np.zeros(shape=empty_features_shape)
                self.weight_mask = np.vstack([existing_examples, empty_examples])
                self.image_features = np.vstack([self.image_features, np.zeros(empty_features_shape)])

            images_space = list(self.ref_weight.shape[1:])
            existing_images_space = [image_count] + images_space
            empty_images_space = [self.batch_size - image_count] + images_space
            existing_images = np.ones(shape=existing_images_space)
            empty_images = np.zeros(shape=empty_images_space)
            self.image_mask = self.image_mask * np.vstack([existing_images, empty_images])
            self.loaded_image = np.vstack([self.loaded_image, np.zeros(empty_images_space)])

        if self.image_features is not None:
            self.assign_placeholder("features_weight", self.weight_mask)
            self.assign_placeholder("ref_img_features", self.image_features)
        self.assign_placeholder("ref_weight", self.image_mask)
        self.assign_placeholder("ref_img", self.loaded_image)

    def optimize(self, iterations=200):
        self.fetch_ops = [self.min_op, self.loss, self.learning_rate]
        for _ in range(iterations):
            _, loss, lr = self.sess.run(self.fetch_ops)
            yield {"loss":loss, "lr": lr}
