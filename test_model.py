import tensorflow as tf
import numpy as np

from dataset import get_dataset, get_rotation_augmentor, get_translation_augmentor
from model import build_model

AUTOTUNE = tf.data.experimental.AUTOTUNE

dataset, num_classes = get_dataset()

model = build_model(num_classes)
model.load_weights('./saved_weights/weights')

rng = np.random.RandomState()

test_dataset = dataset.concatenate(
    dataset.repeat(500)
    .map(get_rotation_augmentor(rng), num_parallel_calls=AUTOTUNE)
    .map(get_translation_augmentor(rng), num_parallel_calls=AUTOTUNE))

model.evaluate(test_dataset.batch(8))
