from absl import flags
from absl import app
import sys
import os
import datetime
import tensorflow as tf
from model import TableGan
from utils import pp, generate_data, show_all_variables
import pickle
import numpy as np

# Define flags
FLAGS = flags.FLAGS

# Update default values for Adult dataset
flags.DEFINE_string("dataset", "Adult", "Dataset name")
flags.DEFINE_integer("epoch", 500, "Number of epochs [increased for better convergence]")
flags.DEFINE_integer("batch_size", 2000, "Batch size [increased for better stability]")
flags.DEFINE_boolean("train", True, "Train the model")
flags.DEFINE_integer("y_dim", 2, "Number of unique classes in native-country column")
flags.DEFINE_integer("attrib_num", 13, "Number of attributes in Adult dataset (excluding label)")
flags.DEFINE_integer("label_col", 13, "Label column index (0-based)")
flags.DEFINE_string("test_id", "OI_11_00", "Test case ID")

# Additional required flags
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate for adam [decreased]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam")
flags.DEFINE_integer("train_size", sys.maxsize, "The size of train images")
flags.DEFINE_integer("input_height", 7, "The size of image to use")
flags.DEFINE_integer("input_width", 7, "The size of image to use")
flags.DEFINE_integer("output_height", 7, "The size of output images")
flags.DEFINE_integer("output_width", 7, "The size of output images")
flags.DEFINE_string("checkpoint_par_dir", "checkpoint", "Parent directory for checkpoints")
flags.DEFINE_string("checkpoint_dir", "", "Directory for checkpoints")
flags.DEFINE_string("sample_dir", "samples", "Directory for samples")
flags.DEFINE_boolean("crop", False, "True for training, False for testing")
flags.DEFINE_boolean("generate_data", False, "True for visualizing")
flags.DEFINE_float("alpha", 2.0, "Weight of original GAN loss [increased]")
flags.DEFINE_float("beta", 5.0, "Weight of information loss [increased significantly]")
flags.DEFINE_float("delta_m", 0.2, "Delta mean parameter [increased]")
flags.DEFINE_float("delta_v", 0.2, "Delta variance parameter [increased]")
flags.DEFINE_integer("feature_size", 1024, "Size of last FC layer [increased]")
flags.DEFINE_boolean("shadow_gan", False, "True for loading fake data")
flags.DEFINE_integer("shgan_input_type", 0, "Input type for shadow_gan")

def prepare_data():
    """Ensure data is preprocessed before training"""
    if not os.path.exists('data/Adult/train_Adult_cleaned.pickle'):
        print("Preprocessing dataset...")
        from preprocess_adult import preprocess_adult_dataset
        preprocess_adult_dataset('data/Adult/train_Adult_cleaned.csv', 'data/Adult')

def main(argv):
    start_time = datetime.datetime.now()

    # Prepare data
    prepare_data()

    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Create directories
    os.makedirs(FLAGS.checkpoint_par_dir, exist_ok=True)
    os.makedirs(FLAGS.sample_dir, exist_ok=True)

    # Set checkpoint directory
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
        tablegan.train(FLAGS)
    else:
        if not tablegan.load(FLAGS.checkpoint_dir):
            raise Exception("[!] Train a model first, then run test mode")
        option = 5 if FLAGS.shadow_gan else 1
        generate_data(tablegan, FLAGS, option)

    end_time = datetime.datetime.now()
    print('Time Elapsed:', end_time - start_time)

if __name__ == '__main__':
    app.run(main)