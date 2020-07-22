from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--repeat',
                    dest='repeat',
                    type=int,
                    default=20,
                    help='augmented repetition')
parser.add_argument('--batch',
                    dest='batch',
                    type=int,
                    default=3,
                    help='batch size')
parser.add_argument('--epochs',
                    dest='epochs',
                    type=int,
                    default=40,
                    help='number of epochs')
args =  parser.parse_args()

import tensorflow as tf
import numpy as np

from model import build_model
from dataset import get_dataset, get_rotation_augmentor, get_translation_augmentor

AUTOTUNE = tf.data.experimental.AUTOTUNE


dataset, num_classes = get_dataset()

model = build_model(num_classes)
model.summary()

rng = np.random.RandomState()

training_dataset = dataset.concatenate(
    dataset.repeat(args.repeat)
    .map(get_rotation_augmentor(rng), num_parallel_calls=AUTOTUNE)
    .map(get_translation_augmentor(rng), num_parallel_calls=AUTOTUNE))

model.fit(training_dataset.batch(args.batch), epochs=args.epochs, shuffle=True)

model.save_weights('./saved_weights/weights')


