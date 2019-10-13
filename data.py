from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import numpy as np
import tensorflow as tf

from distutils.version import LooseVersion

VERSION_GTE_0_12_0 = LooseVersion(tf.__version__) >= LooseVersion('0.12.0')

# Name change in TF v 0.12.0
if VERSION_GTE_0_12_0: #对比TF的版本，如果版本高于0.12.0，则返回同纬度的标准图
    standardize_image = tf.image.per_image_standardization #返回图像同纬度的标准图
else:
    standardize_image = tf.image.per_image_whitening #对图像进行标准化，转化成亮度均值为0，方差为1.

def data_files(data_dir, subset):
    """Returns a python list of all (sharded) data subset files.
    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    if subset not in ['train', 'validation']: #如果子集不在训练集或者确认集，无效子集
        print('Invalid subset!')
        exit(-1)

    tf_record_pattern = os.path.join(data_dir, '%s-*' % subset) #把目录和文件名合成一个路径
    data_files = tf.gfile.Glob(tf_record_pattern) #查找匹配pattern的文件并以列表的形式返回，
    # filename可以是一个具体的文件名，也可以是包含通配符的正则表达式。
    print(data_files)
    if not data_files:
      print('No files found for data dir %s at %s' % (subset, data_dir))

      exit(-1)
    return data_files

def decode_jpeg(image_buffer, scope=None): #图像转换
  """Decode a JPEG string into one 3-D float image Tensor. #将一个图片转换成一个3D浮点数图向量
  Args:
    image_buffer: scalar string Tensor. #标量字符串张量
    scope: Optional scope for op_scope. #可选范围
  Returns:
    3-D float Tensor with values ranging from [0, 1). #返回3维浮点张量，值在0-1之间
  """
  with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time. #结果图片可由 decode_jpeg 动态设置
    image = tf.image.decode_jpeg(image_buffer, channels=3)  # 将图像使用JPEG的格式解码从而得到图像对应的三维矩阵。3通道

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
     #将一个uint类型的tensor转换为float类型时，该方法会自动对数据进行归一化处理，将数据缩放到0-1范围内
    return image

def distort_image(image, height, width): #失真图像

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  distorted_image = tf.random_crop(image, [height, width, 3]) #随机裁剪

  #distorted_image = tf.image.resize_images(image, [height, width])

  # Randomly flip the image horizontally. 水平移动图像
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.

  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63) #随机调整图片亮度

  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8) #随机调整图片对比度

  return distorted_image


def _is_tensor(x): #判断x是否是向量
    return isinstance(x, (tf.Tensor, tf.Variable))

def eval_image(image, height, width): #调整图片大小
    return tf.image.resize_images(image, [height, width])

def data_normalization(image): #正态化图片

    image = standardize_image(image)

    return image

def image_preprocessing(image_buffer, image_size, train, thread_id=0): #图像预处理
    """Decode and preprocess one image for evaluation or training.
    Args:
    image_buffer: JPEG encoded string Tensor
    train: boolean #布尔数 若为真，使图像失真，若为假，调整图片大小
    thread_id: integer indicating preprocessing thread #整数表示与预处理线程
    Returns:
    3-D float Tensor containing an appropriately scaled image #返回3维张量，包含了合适的处理过的图片
    Raises:
    ValueError: if user does not provide bounding box
    """

    image = decode_jpeg(image_buffer)
    
    if train:
        image = distort_image(image, image_size, image_size)
    else:
        image = eval_image(image, image_size, image_size)
        
    image = data_normalization(image)
    return image


def parse_example_proto(example_serialized): #解析样例原型
  # Dense features in Example proto. #密集特征
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''), #编码
      'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''), #文件名

      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1), #类标签
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''), #类文本
      'image/height': tf.FixedLenFeature([1], dtype=tf.int64,
                                         default_value=-1), #高
      'image/width': tf.FixedLenFeature([1], dtype=tf.int64,
                                         default_value=-1), #宽

  }

  features = tf.parse_single_example(example_serialized, feature_map) #解析一个Example原型（连载样例，特征图）
  label = tf.cast(features['image/class/label'], dtype=tf.int32)  #标签数据类型转换
  return features['image/encoded'], label, features['image/filename']

def batch_inputs(data_dir, batch_size, image_size, train, num_preprocess_threads=4,
                 num_readers=1, input_queue_memory_factor=16): #分支输入
  with tf.name_scope('batch_processing'):

    if train:
        files = data_files(data_dir, 'train')
        filename_queue = tf.train.string_input_producer(files,
                                                        shuffle=True,
                                                        capacity=16)
    else:
        files = data_files(data_dir, 'validation')
        filename_queue = tf.train.string_input_producer(files,
                                                        shuffle=False,
                                                        capacity=1) #输出字符串到一个输入管道队列
                                                                    #shuffle：布尔值。如果为true，则在每个epoch内随机打乱顺序。
    if num_preprocess_threads % 4:
              raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads) #需要是4的倍数。。

    if num_readers < 1:
      raise ValueError('Please make num_readers at least 1')

    # Approximate number of examples per shard.
    examples_per_shard = 1024
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    min_queue_examples = examples_per_shard * input_queue_memory_factor
    if train:
      examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string]) #创建一个队列，按随机顺序进行dequeue(出列）定义最大容量
    else:
      examples_queue = tf.FIFOQueue(
          capacity=examples_per_shard + 3 * batch_size,
          dtypes=[tf.string]) #先入先出队

    # Create multiple readers to populate the queue of examples.。以下为数据读取线程管理
    if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
        reader = tf.TFRecordReader() #创建reader
        _, value = reader.read(filename_queue) #返回文件名和文件
        enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
    else:
      reader = tf.TFRecordReader()
      _, example_serialized = reader.read(filename_queue)

    images_labels_fnames = []
    for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata. 解析一个连续样例摘取图片和诠释数据
      image_buffer, label_index, fname = parse_example_proto(example_serialized) #图片缓冲，标签索引，文件名
          
      image = image_preprocessing(image_buffer, image_size, train, thread_id)
      images_labels_fnames.append([image, label_index, fname]) #增补

    images, label_index_batch, fnames = tf.train.batch_join(
        images_labels_fnames,
        batch_size=batch_size,
        capacity=2 * num_preprocess_threads * batch_size)

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, image_size, image_size, 3])

    # Display the training images in the visualizer. 输出图片，最大批处理20
    tf.summary.image('images', images, 20) 

    return images, tf.reshape(label_index_batch, [batch_size]), fnames

def inputs(data_dir, batch_size=128, image_size=227, train=False, num_preprocess_threads=4):
    with tf.device('/cpu:0'): #指定CPU运行
        images, labels, filenames = batch_inputs(
            data_dir, batch_size, image_size, train,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=1)
    return images, labels, filenames

def distorted_inputs(data_dir, batch_size=128, image_size=227, num_preprocess_threads=4):

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation. 预留GPU给正向推理和反向传播
  with tf.device('/cpu:0'):
    images, labels, filenames = batch_inputs(
        data_dir, batch_size, image_size, train=True,
        num_preprocess_threads=num_preprocess_threads,
        num_readers=1)
  return images, labels, filenames
