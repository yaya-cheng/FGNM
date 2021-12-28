"""Implementation of sample attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from utils import *
from attack_method import *
from tqdm import tqdm
from tensorpack import TowerContext
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2, densenet, fdnets, vgg
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_path', '/models', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string('input_csv', 'dataset/dev_dataset.csv', 'Input directory with images.')
tf.flags.DEFINE_string('input_dir', 'dataset/images', 'Input directory with images.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('num_classes', 1001, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 70, 'How many images process at one time.')
tf.flags.DEFINE_integer('k', 200001, 'top k gradient')
tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')
tf.flags.DEFINE_string('method', 'm', 'attack method. m: MIFGSM, i: IFGSM, t: TIFGSM, d: DIFGSM, s: SIFGSM')
tf.flags.DEFINE_string('mode', 'sign', 'attack mode: raw, sign, affine')
tf.flags.DEFINE_float('decimal', '0.9', 'decimal')

FLAGS = tf.flags.FLAGS

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2_101': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt'),
    'vgg_16': os.path.join(FLAGS.checkpoint_path,'vgg_16.ckpt'),
    'resnet_v2_152': os.path.join(FLAGS.checkpoint_path,'resnet_v2_152.ckpt'),
    'adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2_50': os.path.join(FLAGS.checkpoint_path,'resnet_v2_50.ckpt'),
    'densenet': os.path.join(FLAGS.checkpoint_path, 'tf-densenet161.ckpt'),
    'X101-DA': os.path.join(FLAGS.checkpoint_path, 'X101-DenoiseAll_rename.npz'),
    'R152-B': os.path.join(FLAGS.checkpoint_path, 'R152_rename.npz'),
    'R152-D': os.path.join(FLAGS.checkpoint_path, 'R152-Denoise_rename.npz'),
}

T_kern = gkern(15, 3)

def l2Sum(x):
    x = tf.reshape(x, [x.shape[0], -1])
    return tf.math.reduce_sum(x ** 2, -1)

def l1Sum(x):
    x = tf.reshape(x, [x.shape[0], -1])
    return tf.math.reduce_sum(tf.abs(x), -1)

def test(x, maxValue: float, K):
    zero_ = tf.zeros_like(x) + 1e20
    x_ = tf.where(tf.abs(x)<1e-20, x=zero_, y=x)
    countDown = 1.0 / tf.abs(x_)
    scale_ = tf.sort(tf.reshape(countDown, [x.shape[0], -1]), axis=-1, direction='DESCENDING')
    scale_ = scale_[:, K]
    scale_ = tf.expand_dims(scale_, -1)
    scale_ = tf.expand_dims(scale_, -1)
    scale_ = tf.expand_dims(scale_, -1)
    return scale_*x

def affine(x, maxValue: float):
    boost = l2Sum(x)
    zero_ = tf.zeros_like(boost) + 1e-6
    boost = tf.where(boost<1e-20, x=zero_, y=boost)
    scale = tf.sqrt(l2Sum(tf.sign(x)) / boost)
    with tf.control_dependencies([tf.print(tf.reduce_max(scale), tf.reduce_min(scale))]):
        x = tf.identity(x)
    scale = tf.expand_dims(scale, -1)
    scale = tf.expand_dims(scale, -1)
    scale = tf.expand_dims(scale, -1)
    if maxValue is not None:
        scale *= maxValue
    return scale * x

def Boost(x, mode, K):
    if mode=='raw':
        return x
    elif mode=='sign':
        return tf.sign(x)
    elif mode=='affine':
        return affine(x, None)
    elif mode=='test':
        return test(x, None, K)

def IFGSM(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    zero = tf.zeros_like(x)
    one = tf.ones_like(x)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']
    auxlogits = auxlogits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    x_ori = x
    x = x + alpha * Boost(noise, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    cosine, _lambda = get_cos_and_lambda(noise, x_ori, x, x_min, x_max, alpha, value_for_K)
    return x, y, i, x_max, x_min, noise, amplification, value_for_K

def TD(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    x_d=input_diversity(FLAGS, x)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']   
    auxlogits = auxlogits_v3                                             
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')
    x = x + alpha * Boost(noise, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise, amplification, value_for_K

def SD(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter

    x_nes = x
    x_nes_d=input_diversity(FLAGS, x_nes)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                    x_nes_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']
    auxlogits = auxlogits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_2 = 1/2 * x_nes
    x_nes2_d=input_diversity(FLAGS, x_nes_2)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes2_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_3 = 1/4 * x_nes
    x_nes3_d=input_diversity(FLAGS, x_nes_3)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes3_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_4 = 1/8 * x_nes
    x_nes4_d=input_diversity(FLAGS, x_nes_4)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes4_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_5 = 1/16 * x_nes
    x_nes5_d=input_diversity(FLAGS, x_nes_5)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes5_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy/5.0, x)[0]
    x = x + alpha * Boost(noise, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise, amplification, value_for_K

def SDT(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter

    x_nes = x
    x_nes_d=input_diversity(FLAGS, x_nes)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                    x_nes_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']
    auxlogits = auxlogits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_2 = 1/2 * x_nes
    x_nes2_d=input_diversity(FLAGS, x_nes_2)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes2_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_3 = 1/4 * x_nes
    x_nes3_d=input_diversity(FLAGS, x_nes_3)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes3_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_4 = 1/8 * x_nes
    x_nes4_d=input_diversity(FLAGS, x_nes_4)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes4_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_5 = 1/16 * x_nes
    x_nes5_d=input_diversity(FLAGS, x_nes_5)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes5_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy/5.0, x)[0]
    noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')
    x = x + alpha * Boost(noise, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise, amplification, value_for_K

def STDM(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter

    x_nes = x
    x_nes_d=input_diversity(FLAGS, x_nes)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                    x_nes_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']
    auxlogits = auxlogits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_2 = 1/2 * x_nes
    x_nes2_d=input_diversity(FLAGS, x_nes_2)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes2_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_3 = 1/4 * x_nes
    x_nes3_d=input_diversity(FLAGS, x_nes_3)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes3_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_4 = 1/8 * x_nes
    x_nes4_d=input_diversity(FLAGS, x_nes_4)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes4_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_5 = 1/16 * x_nes
    x_nes5_d=input_diversity(FLAGS, x_nes_5)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes5_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy/5.0, x)[0]
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = FLAGS.momentum * grad + noise
    noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')
    x = x + alpha * Boost(noise, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise, amplification, value_for_K

def MIFGSM(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']   
    auxlogits = auxlogits_v3                                             
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    noise_m = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise_m = FLAGS.momentum * grad + noise_m
    x = x + alpha * Boost(noise_m, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise_m, amplification, value_for_K

def get_cos_and_lambda(gradient, x_ori, x, x_min, x_max, alpha, value_for_K):
    clip_noise = x - x_ori
    cosine = tf.reduce_sum(gradient * clip_noise, [1, 2, 3]) / (tf.sqrt(tf.reduce_sum(clip_noise ** 2, [1, 2, 3])) * tf.sqrt(tf.reduce_sum(gradient ** 2, [1, 2, 3])))
    _lambda = l2Sum(clip_noise)
    return cosine, _lambda

def DIFGSM(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    x_d=input_diversity(FLAGS, x)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_d, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    x = x + alpha * Boost(noise, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise, amplification, value_for_K

def TIFGSM(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']   
    auxlogits = auxlogits_v3                                             
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')
    x = x + alpha * Boost(noise, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise, amplification, value_for_K

def SIFGSM(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter

    x_nes = x
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']
    auxlogits = auxlogits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_2 = 1/2 * x_nes
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes_2, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_3 = 1/4 * x_nes
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes_3, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_4 = 1/8 * x_nes
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes_4, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    x_nes_5 = 1/16 * x_nes
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x_nes_5, num_classes = FLAGS.num_classes, is_training = False, reuse=tf.AUTO_REUSE)
    logits = logits_v3
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    auxlogits_v3 = end_points_v3['AuxLogits']  
    auxlogits = auxlogits_v3                                              
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=1.0)
    noise = tf.gradients(cross_entropy/5.0, x)[0]
    x = x + alpha * Boost(noise, mode=FLAGS.mode, K=value_for_K)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise, amplification, value_for_K

def stop(x, y, i, x_max, x_min, grad, amplification, value_for_K):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)

def main(_):
    # Because we normalized the input through "input * 2.0 - 1.0" to [-1,1],
    # the corresponding perturbation also needs to be multiplied by 2
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = FLAGS.num_classes

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as g:
        x_input = tf.placeholder(tf.float32, shape = batch_shape)
        adv_img = tf.placeholder(tf.float32, shape = batch_shape)
        y = tf.placeholder(tf.int32, shape = batch_shape[0])
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        K = tf.placeholder(tf.int32, shape = ())

        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits_adv_vgg16, end_poiincvnts_adv_vgg16 = vgg.vgg_16(
                    tf.image.resize(adv_img, [224, 224]), num_classes = num_classes-1, is_training = False, scope = 'vgg_16')
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_incv3, end_poiincvnts_adv_incv3 = inception_v3.inception_v3(
                    adv_img, num_classes = num_classes, is_training = False, scope = 'InceptionV3', reuse=tf.AUTO_REUSE)
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_adv_incv4, end_poiincvnts_adv_incv4 = inception_v4.inception_v4(
                    adv_img, num_classes = num_classes, is_training = False, scope = 'InceptionV4', reuse=tf.AUTO_REUSE)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_adv_res152, end_poiincvnts_adv_res152 = resnet_v2.resnet_v2_152(
                    adv_img, num_classes = num_classes, is_training = False, scope = 'resnet_v2_152', reuse=tf.AUTO_REUSE)
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_adv_incres, end_poiincvnts_adv_incres = inception_resnet_v2.inception_resnet_v2(
                    adv_img, num_classes = num_classes, is_training = False, scope = 'InceptionResnetV2', reuse=tf.AUTO_REUSE)
        with slim.arg_scope(densenet.densenet_arg_scope()):
            logits_adv_dense, end_poiincvnts_adv_dense = densenet.densenet161(
                    tf.image.resize(adv_img, [224, 224]), num_classes = num_classes-1, is_training = False, reuse=tf.AUTO_REUSE)
        
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                adv_img, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')
        pre_ens3_adv_v3 = tf.argmax(logits_ens3_adv_v3, 1)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                adv_img, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
        pre_ens4_adv_v3 = tf.argmax(logits_ens4_adv_v3, 1)
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                adv_img, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
        pre_ensadv_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)

        logits_adv_dense = tf.squeeze(logits_adv_dense)
        # logits_adv_dense = logits_adv_dense[tf.newaxis,:]
        pre_adv_incv3 = tf.argmax(logits_adv_incv3, 1) #第二维
        pre_adv_incv4 = tf.argmax(logits_adv_incv4, 1)
        pre_adv_res152 = tf.argmax(logits_adv_res152, 1)
        pre_adv_incres = tf.argmax(logits_adv_incres, 1)
        pre_adv_dense = tf.argmax(logits_adv_dense, 1)
        pre_adv_vgg = tf.argmax(logits_adv_vgg16, 1)

        i = tf.constant(0, name="iteration")
        grad = tf.zeros(shape=batch_shape)
        amplification = tf.zeros(shape = batch_shape)
        if FLAGS.method == 'if':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, IFGSM, [x_input, y, i, x_max, x_min, grad, amplification, K])
        if FLAGS.method == 'm':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, MIFGSM, [x_input, y, i, x_max, x_min, grad, amplification, K])
        if FLAGS.method == 'd':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, DIFGSM, [x_input, y, i, x_max, x_min, grad, amplification, K])
        if FLAGS.method == 't':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, TIFGSM, [x_input, y, i, x_max, x_min, grad, amplification, K])
        if FLAGS.method == 's':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, SIFGSM, [x_input, y, i, x_max, x_min, grad, amplification, K])
        if FLAGS.method == 'sd':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, SD, [x_input, y, i, x_max, x_min, grad, amplification, K])
        if FLAGS.method == 'stdm':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, STDM, [x_input, y, i, x_max, x_min, grad, amplification, K])
        if FLAGS.method == 'td':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, TD, [x_input, y, i, x_max, x_min, grad, amplification, K])
        if FLAGS.method == 'sdt':
            x_adv, _, _, _, _, _, _, K_ = tf.while_loop(stop, SDT, [x_input, y, i, x_max, x_min, grad, amplification, K])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s12 = tf.train.Saver(slim.get_model_variables(scope='densenet161'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s5.restore(sess, model_checkpoint_map['inception_v4'])
            s6.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s8.restore(sess, model_checkpoint_map['resnet_v2_152'])
            s12.restore(sess, model_checkpoint_map['densenet'])
            s3.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            s4.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            s9.restore(sess, model_checkpoint_map['vgg_16'])

            import pandas as pd
            allResult = list()
            for kk in range(1, 299*299*3-1, 500):
                kk = 120001
                dev = pd.read_csv(FLAGS.input_csv)
                eps = 2.0 * FLAGS.max_epsilon / 255.0
                num_classes = FLAGS.num_classes
                sum_ensadv_res_v2 = 0
                sum_adv_incv3 = 0
                sum_adv_incv4 = 0
                sum_adv_res152 = 0
                sum_adv_incres = 0
                sum_adv_dense = 0
                sum_adv_ensen3Incv3 = 0
                sum_ens3_adv_v3, sum_ens4_adv_v3, sum_ensadv_res_v2, sum_ensadv_vgg = 0, 0, 0, 0
                meanALL = 0.0
                for idx in range(0, 1000 // FLAGS.batch_size):
                    images, filenames, True_label = load_images(FLAGS.input_dir, dev, idx * FLAGS.batch_size, batch_shape)
                    my_adv_images = sess.run(x_adv, feed_dict={x_input: images, y: True_label, K: kk})
                    my_adv_images = my_adv_images.astype(np.float32)
                    advImg = my_adv_images
                    ratio = np.abs(advImg - images)
                    mean = np.mean(ratio)
                    meanALL += mean

                    pre_adv_incv3_ = sess.run(pre_adv_incv3,
                                            feed_dict = {adv_img: my_adv_images})
                    pre_adv_incv4_ = sess.run(pre_adv_incv4,
                                            feed_dict = {adv_img: my_adv_images})
                    pre_adv_res152_ = sess.run(pre_adv_res152,
                                            feed_dict = {adv_img: my_adv_images})
                    pre_adv_incres_ = sess.run(pre_adv_incres,
                                            feed_dict = {adv_img: my_adv_images})
                    pre_adv_dense_ = sess.run(pre_adv_dense,
                                            feed_dict = {adv_img: my_adv_images})
                    pre_ens3_adv_v3_, pre_ens4_adv_v3_, pre_ensadv_res_v2_ = sess.run([pre_ens3_adv_v3, pre_ens4_adv_v3, pre_ensadv_res_v2,], feed_dict = {adv_img: my_adv_images})

                    sum_adv_incv3 += (pre_adv_incv3_ != True_label).astype(np.float).sum()
                    sum_adv_incv4 += (pre_adv_incv4_ != True_label).astype(np.float).sum()
                    sum_adv_res152 += (pre_adv_res152_ != True_label).astype(np.float).sum()
                    sum_adv_incres += (pre_adv_incres_ != True_label).astype(np.float).sum()
                    sum_adv_dense += ((pre_adv_dense_ + 1) != True_label).astype(np.float).sum()
                    sum_ens3_adv_v3 += (pre_ens3_adv_v3_ != True_label).sum()
                    sum_ens4_adv_v3 += (pre_ens4_adv_v3_ != True_label).sum()
                    sum_ensadv_res_v2 += (pre_ensadv_res_v2_ != True_label).sum()

                # save result with different k
                #allResult.append([kk, (meanALL/(1000.0/FLAGS.batch_size))*255.0/2.0, sum_adv_incv3 / 1000.0, sum_adv_incv4 / 1000.0, sum_adv_res152 / 1000.0, sum_adv_incres / 1000.0, sum_adv_dense / 1000.0, sum_ens3_adv_v3/1000.0, sum_ens4_adv_v3/1000.0, sum_ensadv_res_v2/1000.0])
                print('K: ', kk)
                # mean of noise
                print('mean noise: ', (meanALL/(1000.0/FLAGS.batch_size))*255.0/2.0)
                print('inc_v3 = {}'.format(sum_adv_incv3 / 1000.0))
                print('inc_v4 = {}'.format(sum_adv_incv4 / 1000.0))
                print('res152 = {}'.format(sum_adv_res152 / 1000.0))
                print('inc_res = {}'.format(sum_adv_incres / 1000.0))
                print('dense = {}'.format(sum_adv_dense / 1000.0))
                print('sum_ens3_adv_v3 = {}'.format(sum_ens3_adv_v3/1000.0))
                print('sum_ens4_adv_v3 = {}'.format(sum_ens4_adv_v3/1000.0))
                print('sum_ensadv_Incres_v2 = {}'.format(sum_ensadv_res_v2/1000.0))
                exit()
            # allResult = np.array(allResult)
            # np.savetxt(f"tensorflow_allResults_{FLAGS.method}_{FLAGS.mode}.csv", allResult, fmt="%.5f", delimiter=',')

if __name__ == '__main__':
    tf.app.run()
