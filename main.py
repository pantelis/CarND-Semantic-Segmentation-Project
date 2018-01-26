import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Runtime parameters:

# The number of skips controls the complexity of the network.
# with 0 skips we obtain the so called fcn32s network,
# with 1 skip the fcn16s network and
# with 2 skips the fcn8s network.
# please note that here we train all at once the three networks rather than in stages.
NUM_SKIPS = 2
EPOCHS = 50
BATCH_SIZE = 10

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # This model is modified compared to the standard VGG16
    # While in the original VGG16 there are three FC layers after the 5th pooling layer, the model here
    # has the third FC layer decapitated and the two FC layers converted to convolutional layers.
    # by replacing the fully connected layers 6 and 7 with convolutional layers. We call this network
    # the modified VGG16 network (mVGG16).

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # FCN Encoder
    # It is constructed by appending a 1x1 convolutional layer to the mVGG16 network
    # (see above for mVGG16 definition). The number of classes is 2 for the case of road or no-road but
    # the structure is generic to accommodate multiple classes.

    encoder_output = tf.layers.conv2d(inputs=vgg_layer7_out, filters=num_classes, kernel_size=[1, 1],
                                      padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # FCN-8s Decoder
    # The FCN-8s decoder is a three stream decoder
    #
    # The first stream takes the encoder output and upsamples it via interpolation to the original image size
    # in a single step. This means that the shape of the tensor after this convolutional transpose layer will be
    # 4-dimensional: (batch_size, original_height, original_width, num_classes).

    fcn_32s = tf.layers.conv2d_transpose(inputs=encoder_output, filters=num_classes, kernel_size=[4, 4], strides=2,
                                        padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    if NUM_SKIPS == 0:
        return fcn_32s
    # While the fcn_32s is able to be used for semantic segmentation, we can improve by adding two other streams
    # as follows:

    # Stream 2:
    # The second stream fuses the output of the 4th layer and the final layer and such fusion improves the segmentation
    # as it adds more local features (as the 4th layer is a higher resolution layer having finer strides in prediction).
    # Effectively with this technique the model is able to make local predictions that respect global structure.
    # This technique is called "adding skips" as shallower layers skip few layers and feed straight into upper layers.

    # As fusion is numerically an element-wise addition, in the implementation we need to take care of the different
    # scaling (resolutions) that between the two layers that need to be fused. Similar to the 1st stream where we took
    # a low resolution layer and we interpolated its elements via a transpose convolution operation, we score
    # the output of layer 4 (pool 4) via a 1x1 convolution layer and then interpolate aligning the scales (dimensions).
    # No cropping was implemented although croping is present in the reference Caffe implementation
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s-atonce/net.py
    # Before we apply the 1x1 convolutional layer we scale the layer 4 (pool 4) output that empirically it has been
    # shown to avoid convergence issues - we apply the same scaling as in the original implementation.

    # scaling as described in stream 2 comments above
    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01)

    # apply 1x1 conv in scaled output
    scored_scaled_layer4 = tf.layers.conv2d(inputs=vgg_layer4_out, filters=num_classes, kernel_size=[1, 1],
                     padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # fusion
    fused_4 = tf.add(fcn_32s, scored_scaled_layer4)

    # interpolation
    fcn_16s = tf.layers.conv2d_transpose(inputs=fused_4, filters=num_classes, kernel_size=[4, 4], strides=2,
                                         padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    if NUM_SKIPS == 1:
        return fcn_16s

    # Stream 3:
    # It is similar to the Stream 2 and in this case we skip connect from the output of layer 3 (pool 3).

    # scaling as described above - the scalar was extracted from the Caffe implementation.
    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001)

    # apply 1x1 conv in scaled output
    scored_scaled_layer3 = tf.layers.conv2d(inputs=vgg_layer3_out, filters=num_classes, kernel_size=[1, 1],
                                            padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # fusion
    fused_3 = tf.add(fcn_16s, scored_scaled_layer3)

    # interpolation
    fcn_8s = tf.layers.conv2d_transpose(inputs=fused_3, filters=num_classes, kernel_size=[16, 16], strides=8,
                                         padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return fcn_8s

tests.test_layers(layers)


def optimize(nn_last_layer, labels, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # In the layers function above we have seen that the shape of the tensor after this convolutional transpose layer
    # will be 4D: (batch_size, original_height, original_width, num_classes). This is irrespective of number of streams
    # we define in the layers function (fcn32s, fcn16s or fcn8s).
    # We first reshape this tensor and the associated labels below

    # the reshaped logits is a 2D tensor, where each row represents a pixel and each column a class.
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    # labels are similarly reshaped
    labels = tf.reshape(labels, (-1, num_classes))

    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Implement function
    # function implementation is standard and follows:
    # https://github.com/pantelis/traffic-sign-classifier/blob/master/LeNet-Traffic-Sign-Classifier.py

    beginTime = time.time()
    # saver = tf.train.Saver(sess)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i + 1))
        for image, label in get_batches_fn(batch_size):

            # dropout parameter 50% is from the original paper
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5,
                                          learning_rate: 0.001})

            print("Loss: = {:.3f}".format(loss))
        print()

    # # Save the model after training.
    # saver.save(sess, './trained_model'+'_skip_'+str(NUM_SKIPS))
    # print("Model saved")

    endTime = time.time()
    print('Training time: {:5.2f}s'.format(endTime - beginTime))
    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        print(tf.trainable_variables())

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # load the modified VGG model from disk
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # define the fcn32s, fcn16s and fcn8s network
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # deine the SGD optimizer
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train the network
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
