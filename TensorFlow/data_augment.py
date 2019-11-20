import numpy as np
import scipy as sc
import tensorflow as tf

"""
 Defines own random image transformation methods. Difference to tf.image.random... is that here the probability of
 the application of these transformations can be altered.

 With many transformations in row, the problem might arise that too many transformations are applied at once and the network
 rarely 'sees' the original image.
 In order to overcome this problem, a two step decision process is implemented:
    1. it will be decided for each image, whether data augmentation is to be applied (probability is adjustable)
    2. if yes, then data augmentation techniques are applied subject to 'select_aug_fcts()'. 
"""


# ------------------------------------
#
# ------------------------------------
def random_brightness(image, p=0.2, max_delta=0.05):
    '''
    adjusts brightness of the image with probability p
    '''
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)
    image = tf.cond(cond,
                    lambda: tf.image.random_brightness(image, max_delta=max_delta),
                    # randomly picks delta from [-max_delta, max_delta]
                    lambda: tf.identity(image))
    return image


def random_flipping(image, p=0.2):
    '''
    flips image from left to right with probability p
    '''
    rand = tf.random_uniform([], 0, 1.0)
    # rand = tf.Print(rand,[rand])
    cond = tf.less(rand, p)
    image = tf.cond(cond,
                    lambda: tf.image.flip_left_right(image),
                    lambda: tf.identity(image))
    return image


def random_hue(image, p=0.2, max_delta=0.04):
    '''
    adjusts hue of the image with probability p
    delta is picked out of [-max_delta, max_delta]
    '''
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)
    image = tf.cond(cond,
                    lambda: tf.image.random_hue(image, max_delta=max_delta),
                    lambda: tf.identity(image))
    return image


def random_contrast(image, lower=0.8, upper=1.2, p=0.2):
    '''
    randomly adjusts the contrast of the image
    '''
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)
    image = tf.cond(cond,
                    lambda: tf.image.random_contrast(image, lower=lower, upper=upper),
                    lambda: tf.identity(image))
    return image


# ------------------------------------
# randomly adjusts the saturation of the image
# ------------------------------------
# def random_saturation(image, lower=0.8, upper=1.2, p=0.2):
# 	rand = tf.random_uniform([], 0 , 1.0)
# 	cond = tf.less(rand,p)
# 	image = tf.cond(cond,
# 		lambda: tf.image.random_saturation(image, lower=lower, upper=upper),
# 		lambda: tf.identity(image))
# 	return image

# !! Raus, da komisches Verhalten bei pixel-mean-subtraction: viele konstante Nullen im Bild.. 
# nicht nachvollziehbar, geschieht bei /255 nicht...


def cutout(image, cutout_dim=[16, 16], p=0.2):
    '''
    performs cutout/random delete on the image

    '''

    # cutout benötigt numpy für die maske und die zufallszahlen (in tf sind indizierungen wie unten scheinbar schwer
    # machbar), daher wird die komplette Funktion in 'build_mask()' ausgelagert und mittels tf.py_func in 'cutout()'
    # aufgerufen.
    ori_image = image
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)

    def build_mask(image, p):
        # rand = np.random.random_sample()
        # if rand > p:
        #    return image
        ori_spatial_dims = image.shape
        image = np.transpose(image, (2, 0, 1))  # image transformation from HWC to CHW
        height = np.size(image, 1)
        width = np.size(image, 2)
        mask = np.ones((height, width), np.uint8)
        # pick cutout rectangle dimension: width and height randomly sampled between 12 and 16
        # np.random.seed(1)
        # w = np.random.randint(12, 17)
        # h = np.random.randint(12, 17)
        w = cutout_dim[0]
        h = cutout_dim[1]
        # define random center of cutout area

        x = np.random.randint(ori_spatial_dims[0])  # TODO
        y = np.random.randint(ori_spatial_dims[1])
        # print(w,h,x,y)
        # calculate resulting coordinates of cutout edges, within feasible interval [0, height]
        y1 = np.clip(y - h // 2, 0, height)
        y2 = np.clip(y + h // 2, 0, height)
        x1 = np.clip(x - w // 2, 0, width)
        x2 = np.clip(x + w // 2, 0, width)
        # set cutout area in mask to 0
        mask[y1:y2, x1:x2] = 0
        # masking operation
        image = image * mask
        image = np.transpose(image, (1, 2, 0))  # transform image back to HWC format
        return image

    image = tf.cond(cond,
                    lambda: tf.py_func(build_mask, [image, p], tf.float32),
                    lambda: tf.identity(image))
    image = tf.reshape(image, ori_image.get_shape())
    return image


def pad_crop(image, p=0.2):
    '''
    first increases the size of the pictures by 2px and then crops them back to 32x32,
    thereby the image is shifted slightly and partially occluded.

    :param image:
    :param p:
    :return:
    '''
    orig_image = image
    orig_shape = orig_image.get_shape()
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)
    image = tf.pad(image, [[2, 2], [2, 2], [0, 0]], "SYMMETRIC",
                   constant_values=0)  # maybe more than 1px? maybe const_val = mean?
    image_out = tf.cond(cond,
                        lambda: tf.image.random_crop(image, orig_shape),
                        lambda: tf.identity(orig_image))
    image_out = tf.reshape(image_out, orig_shape)
    return image_out


def crop_and_resize(image, crop_size=28, p=0.2):
    '''
    performs random cropping and then resizes the image back to its original size.
    '''
    orig_image = image
    ori_spatial_dims = orig_image.get_shape()[0:2]
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)

    image = tf.image.random_crop(image, [crop_size, crop_size, 3])

    image_out = tf.cond(cond,
                        # casting nur nötig solange mean-subtraction deaktiviert ist -> dann ist image ja wieder float.
                        # lambda: tf.cast(tf.image.resize_images(image, [32,32], align_corners=True), dtype=tf.uint8),
                        lambda: tf.image.resize_images(image, ori_spatial_dims, align_corners=True),
                        lambda: tf.identity(orig_image))
    image = tf.reshape(image, orig_image.get_shape())
    return image_out


def rotate_image(image, max_angle=12, p=0.2):
    '''
    rotates the image randomly between -max_angle and max_angle degrees.
    then crops the image and resizes it again to image spatial dim px
    '''
    orig_image = image
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)

    # scipy fct used to rotate the image, due to the needed random scalar number produced by numpy
    def rand_rotate(image):
        angle = np.random.randint(-max_angle, max_angle)  # TODO set numpy seed manually
        return sc.ndimage.rotate(image, angle=angle, reshape=False)  # tf.contrib.image.rotate hat probleme gemacht..

    image = tf.py_func(rand_rotate, [image], Tout=tf.float32)
    image = tf.reshape(image, orig_image.get_shape())  # nötig damit tensorflow nicht rumheult..

    image = tf.image.central_crop(image, 0.85)

    ori_spatial_dims = orig_image.get_shape()[0:2]
    image_out = tf.cond(cond,
                        lambda: tf.image.resize_images(image, ori_spatial_dims, align_corners=True),
                        lambda: tf.identity(orig_image))
    return image_out


def gaussian_noise(image, p=0.2):
    '''
    only useful with floats... hence difficult to evaluate..
    good value for sd can only be assessed for /255 normalization
    (because for pixel_mean subtraction the image cannot be visualized)

    '''
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)

    noise = tf.random_normal(image.get_shape(), mean=0, stddev=0.03)  # mit mean-subtraction casting wieder entfernen!
    image_noise = image + noise
    image_out = tf.cond(cond,
                        lambda: tf.identity(image_noise),
                        lambda: tf.identity(image))
    return image_out


def transform_image(image, forward_transforms, output_shape=None):
    t = tf.contrib.image.matrices_to_flat_transforms(tf.linalg.inv(forward_transforms))
    # forward transform needs to be a float matrix!
    image = tf.contrib.image.transform(image, t, interpolation="BILINEAR", output_shape=output_shape)
    return image


def shear(image, shear_value=0.2, p=0.2):
    orig_image = image
    rand = tf.random_uniform([], 0, 1.0)
    cond = tf.less(rand, p)
    image = transform_image(image, [[1.0, shear_value, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], output_shape=[32, 37])
    image = tf.image.central_crop(image, 0.85)
    image = tf.cond(cond,
                    lambda: tf.image.resize_images(image, [32, 32], align_corners=True),
                    lambda: tf.identity(orig_image))
    return image


def cifar10_augment(image, label):
    image = f4(image, [16, 16])
    return image, label


def cifar100_augment(image, label):
    image = f4(image, [8, 8])
    return image, label


def imagenet_augment(image, label):
    image = f3(image, [56, 56])
    return image, label


def f4(image, cutout_dim):
    image = pad_crop(image, p=0.5)
    image = rotate_image(image, p=0.5)
    image = random_flipping(image, p=0.5)
    image = random_hue(image, p=0.5)
    image = random_brightness(image, p=0.5)
    image = random_contrast(image, p=0.5)
    # image = gaussian_noise(image, p=.0.2) -> nur für /255 normalisierung sicher verwendbar.
    image = cutout(image, cutout_dim, p=0.5)
    # save= save_image(image)
    # with tf.control_dependencies([save]):
    ##    image = tf.Print(image, ["f1"])
    return image


def f1(image):
    image = rotate_image(image, p=0.5)
    image = random_flipping(image, p=0.5)
    image = random_hue(image, p=0.1)
    image = random_brightness(image, p=0.1)
    image = random_contrast(image, p=0.1)
    # image = gaussian_noise(image, p=.0.2) -> nur für /255 normalisierung sicher verwendbar.
    image = cutout(image, p=0.6)
    # save= save_image(image)
    # with tf.control_dependencies([save]):
    ##    image = tf.Print(image, ["f1"])
    return image


def f2(image):
    image = pad_crop(image, p=0.5)
    image = random_flipping(image, p=0.5)
    image = random_hue(image, p=0.1)
    image = random_brightness(image, p=0.1)
    image = random_contrast(image, p=0.1)
    # image = gaussian_noise(image, p=.0.2) -> nur für /255 normalisierung sicher verwendbar.
    image = cutout(image, p=0.5)
    # save= save_image(image)
    # with tf.control_dependencies([save]):
    #    image = tf.Print(image, ["f2"])
    return image


def f3(image, cutout_dim):
    # image = crop_and_resize(image,crop_size, p=0.5) # looses dim
    image = random_flipping(image, p=0.5)
    image = random_hue(image, p=0.5)
    image = random_brightness(image, p=0.5)
    image = random_contrast(image, p=0.5)
    image = rotate_image(image, p=0.5)  # looses dim
    #  image = gaussian_noise(image, p=.0.2) -> nur für /255 normalisierung sicher verwendbar.
    image = cutout(image, cutout_dim, p=0.5)
    # save_op= save_image(image)
    # with tf.control_dependencies([save_op]):
    #    image = tf.Print(image, ["f3"])
    return image


i = 1


def save_image(image):
    image = tf.cast(image, tf.uint8)
    output_image = tf.image.encode_jpeg(image)

    # Create a constant as filename
    # num = tf.Variable(0,trainable=False,dtype=tf.int32) # käse wird ha immer wieder nue erstellt
    # one= tf.constant(1, dtype=tf.int32) TODO create one variable outside add
    # num_add_op= num.assign(num+one)
    # num = tf.Variable(0,use_resource=True)
    num = tf.dtypes.as_string(tf.random.uniform([], maxval=100000, dtype=tf.int32))
    prefix = tf.constant('./output_images/image')
    postfix = tf.constant('.jpg')
    file_name = prefix + num + postfix
    file_name = tf.Print(file_name, [file_name])

    save_op = tf.write_file(file_name, output_image)
    # file = tf.Print(file, ["file written"])
    # with tf.control_dependencies([file,tf.print("psaved file")]):
    # tf.identity(image)
    return save_op
    # print(session.run(file))
