"""
main.py
"""
import tensorflow as tf

from tasks.addition.env.generate_data import generate_addition
from tasks.addition.eval import evaluate_addition
from tasks.addition.train import train_addition

from tasks.card_pattern_matching.trace_generator import generate as generate_card_pattern_matching
from tasks.card_pattern_matching.train import train_card_pattern_matching
from tasks.card_pattern_matching.eval import evaluate_card_pattern_matching


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("task", "card_pattern_matching", "Which NPI Task to run - [addition, card_pattern_matching, merge_sort].")

tf.app.flags.DEFINE_boolean("generate", True, "Boolean whether to generate training/test data.")
tf.app.flags.DEFINE_integer("num_training", 1000, "Number of training examples to generate.")
tf.app.flags.DEFINE_integer("num_test", 100, "Number of test examples to generate.")

tf.app.flags.DEFINE_boolean("do_train", False, "Boolean whether to continue training model.")
tf.app.flags.DEFINE_boolean("do_eval", False, "Boolean whether to perform model evaluation.")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs to perform.")


def addition():
    # Generate Data (if necessary)
    if FLAGS.generate:
        generate_addition('train', FLAGS.num_training)
        generate_addition('test', FLAGS.num_test)

    # Train Model (if necessary)
    if FLAGS.do_train:
        train_addition(FLAGS.num_epochs)

    # Evaluate Model
    if FLAGS.do_eval:
        evaluate_addition()


def card_pattern_matching():
    # Generate Data (if necessary)
    if FLAGS.generate:
        generate_card_pattern_matching('train', num=FLAGS.num_training)
        generate_card_pattern_matching('test', num=FLAGS.num_test, only_random=True)

    # Train Model (if necessary)
    if FLAGS.do_train:
        train_card_pattern_matching(FLAGS.num_epochs)

    # Evaluate Model
    if FLAGS.do_eval:
        evaluate_card_pattern_matching()


def main(_):

    if FLAGS.task == "addition":
        addition()
    elif FLAGS.task == "card_pattern_matching":
        card_pattern_matching()


if __name__ == "__main__":
    tf.app.run()