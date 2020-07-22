import tensorflow as tf
import numpy as np

from dataset import get_dataset, get_rotation_augmentor, get_translation_augmentor
from model import build_model

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset, num_classes = get_dataset()

model = build_model(num_classes)
model.load_weights('./saved_weights/weights')

rng = np.random.RandomState()

augmented_dataset = train_dataset.concatenate(
    train_dataset
    .repeat(20)
    .map(get_rotation_augmentor(rng), num_parallel_calls=AUTOTUNE)
    .map(get_translation_augmentor(rng), num_parallel_calls=AUTOTUNE))

model.evaluate(augmented_dataset.batch(8), verbose=2)
