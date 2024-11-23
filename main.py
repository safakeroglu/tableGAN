"""
Paper: http://www.vldb.org/pvldb/vol11/p1071-park.pdf
Modified for TensorFlow 2.x compatibility
"""
import os
import datetime
import tensorflow as tf
import sys
from absl import flags
from absl import app

from model import TableGan
from utils import pp, generate_data, show_all_variables

# Define flags using absl.flags instead of tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("epoch", 300, "Epoch to train [was: 150]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam")
flags.DEFINE_integer("train_size", sys.maxsize, "The size of train images [np.inf]")
flags.DEFINE_integer("y_dim", 2, "Number of unique labels")
flags.DEFINE_integer("batch_size", 1000, "The size of batch [was: 750]")
flags.DEFINE_integer("input_height", 16, "The size of image to use")
flags.DEFINE_integer("input_width", None, "The size of image to use. If None, same as input_height")
flags.DEFINE_integer("output_height", 16, "The size of output images")
flags.DEFINE_integer("output_width", None, "The size of output images. If None, same as output_height")
flags.DEFINE_string("dataset", "celebA", "The name of dataset")
flags.DEFINE_string("checkpoint_par_dir", "checkpoint", "Parent directory for checkpoints")
flags.DEFINE_string("checkpoint_dir", "", "Directory for checkpoints")
flags.DEFINE_string("sample_dir", "samples", "Directory for samples")
flags.DEFINE_boolean("train", False, "True for training, False for testing")
flags.DEFINE_boolean("crop", False, "True for training, False for testing")
flags.DEFINE_boolean("generate_data", False, "True for visualizing")
flags.DEFINE_float("alpha", 0.5, "Weight of original GAN loss [0-1.0]")
flags.DEFINE_float("beta", 2.0, "Weight of information loss [was: 1.0]")
flags.DEFINE_float("delta_m", 0.1, "Delta mean parameter [was: 0.0]")
flags.DEFINE_float("delta_v", 0.1, "Delta variance parameter [was: 0.0]")
flags.DEFINE_string("test_id", "5555", "Experiment settings ID")
flags.DEFINE_integer("label_col", -1, "Label column index (from 0)")
flags.DEFINE_integer("attrib_num", 0, "Number of columns in dataset")
flags.DEFINE_integer("feature_size", 512, "Size of last FC layer [was: 256]")
flags.DEFINE_boolean("shadow_gan", False, "True for loading fake data")
flags.DEFINE_integer("shgan_input_type", 0, "Input type for shadow_gan (1=Fake, 2=Test, 3=Train)")
flags.DEFINE_integer("discriminator_steps", 2, "Number of discriminator updates per generator update")
flags.DEFINE_float("gradient_penalty_weight", 10.0, "Weight for gradient penalty")
flags.DEFINE_boolean("use_batch_norm", True, "Use batch normalization")
flags.DEFINE_float("noise_std", 0.2, "Standard deviation of noise input")
flags.DEFINE_integer("generator_layers", 3, "Number of generator layers")
flags.DEFINE_integer("discriminator_layers", 3, "Number of discriminator layers")
flags.DEFINE_boolean("use_label_smoothing", True, "Use label smoothing")
flags.DEFINE_float("label_smoothing", 0.1, "Label smoothing factor")

def main(argv):
    start_time = datetime.datetime.now()

    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Handle input/output dimensions
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    # Create directories
    os.makedirs(FLAGS.checkpoint_par_dir, exist_ok=True)
    os.makedirs(FLAGS.sample_dir, exist_ok=True)

    # Test cases configuration
    test_cases = [
        {'id': 'OI_11_00', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.0, 'delta_m': 0.0},
        {'id': 'OI_11_11', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.1, 'delta_m': 0.1},
        {'id': 'OI_11_22', 'alpha': 1.0, 'beta': 1.0, 'delta_v': 0.2, 'delta_m': 0.2},
        {'id': 'OI_101_00', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.0, 'delta_m': 0.0},
        {'id': 'OI_101_11', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.1, 'delta_m': 0.1},
        {'id': 'OI_101_22', 'alpha': 1.0, 'beta': 0.1, 'delta_v': 0.2, 'delta_m': 0.2},
        {'id': 'OI_1001_00', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.0, 'delta_m': 0.0},
        {'id': 'OI_1001_11', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.1, 'delta_m': 0.1},
        {'id': 'OI_1001_22', 'alpha': 1.0, 'beta': 0.01, 'delta_v': 0.2, 'delta_m': 0.2}
    ]

    # Set parameters based on test case
    found = False
    for case in test_cases:
        if case['id'] == FLAGS.test_id:
            found = True
            FLAGS.alpha = case['alpha']
            FLAGS.beta = case['beta']
            FLAGS.delta_m = case['delta_m']
            FLAGS.delta_v = case['delta_v']
            print(case)
            break

    if not found:
        print("Using OI_11_00")
        FLAGS.test_id = "OI_11_00"
        FLAGS.alpha = 1.0
        FLAGS.beta = 1.0
        FLAGS.delta_m = 0.0
        FLAGS.delta_v = 0.0

    # Set dimensions
    FLAGS.input_height = 7
    FLAGS.input_width = 7
    FLAGS.output_height = 7
    FLAGS.output_width = 7

    # Set checkpoint directory
    if FLAGS.shadow_gan:
        checkpoint_folder = os.path.join(FLAGS.checkpoint_par_dir, FLAGS.dataset, f'atk_{FLAGS.test_id}')
    else:
        checkpoint_folder = os.path.join(FLAGS.checkpoint_par_dir, FLAGS.dataset, FLAGS.test_id)

    os.makedirs(checkpoint_folder, exist_ok=True)
    FLAGS.checkpoint_dir = checkpoint_folder

    # Print configuration
    pp.pprint(flags.FLAGS.flag_values_dict())
    print(f"y_dim: {FLAGS.y_dim}")
    print(f"Checkpoint: {FLAGS.checkpoint_dir}")

    # Initialize model
    tablegan = TableGan(
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        y_dim=FLAGS.y_dim,
        dataset_name=FLAGS.dataset,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        alpha=FLAGS.alpha,
        beta=FLAGS.beta,
        delta_mean=FLAGS.delta_m,
        delta_var=FLAGS.delta_v,
        label_col=FLAGS.label_col,
        attrib_num=FLAGS.attrib_num,
        is_shadow_gan=FLAGS.shadow_gan,
        test_id=FLAGS.test_id
    )

    show_all_variables()

    if FLAGS.train:
        # Training mode
        tablegan.train(FLAGS)
    else:
        # Testing mode
        if not tablegan.load(FLAGS.checkpoint_dir):
            raise Exception("[!] Train a model first, then run test mode")

        # Generate data
        option = 5 if FLAGS.shadow_gan else 1
        generate_data(tablegan, FLAGS, option)

    # Print elapsed time
    end_time = datetime.datetime.now()
    print('Time Elapsed:', end_time - start_time)

if __name__ == '__main__':
    app.run(main)
