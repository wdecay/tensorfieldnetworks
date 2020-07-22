import tensorflow as tf
from dataset import get_dataset
from model import build_model

train_dataset, num_classes = get_dataset()

model = build_model(num_classes)
model.summary()

model.fit(train_dataset.repeat(30).batch(5), epochs=40, shuffle=True)

model.save_weights('./saved_weights/weights')
