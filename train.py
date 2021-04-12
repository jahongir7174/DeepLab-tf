import os
import sys

import numpy as np
import tensorflow as tf

from nets import nn
from utils import config, util
from utils.dataset import input_fn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(config.seed)
tf.random.set_seed(config.seed)
strategy = tf.distribute.MirroredStrategy()

_, palette = util.get_label_info(os.path.join(config.data_dir, 'class_dict.csv'))
file_names = [file_name[:-4] for file_name in os.listdir(os.path.join(config.data_dir, config.image_dir))]

dataset = input_fn(file_names, palette)
dataset = strategy.experimental_distribute_dataset(dataset)
weights = util.get_class_weights(file_names)

with strategy.scope():
    optimizer = tf.keras.optimizers.RMSprop(0.0001, decay=0.995)
    model = nn.build_model((config.height, config.width, 3), len(palette))
    model(tf.zeros((1, config.height, config.width, 3)))

with strategy.scope():
    loss_fn = nn.segmentation_loss(weights)


    def compute_loss(y_true, y_pred):
        return tf.reduce_sum(loss_fn(y_true, y_pred)) * 1. / config.batch_size

with strategy.scope():
    def train_step(image, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(image)
            loss = compute_loss(y_true, y_pred)
        train_variable = model.trainable_variables
        gradient = tape.gradient(loss, train_variable)
        optimizer.apply_gradients(zip(gradient, train_variable))

        return loss

with strategy.scope():
    @tf.function
    def distribute_train_step(image, y_true):
        loss = strategy.run(train_step, args=(image, y_true))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)


def main():
    steps = len(file_names) // config.batch_size
    if not os.path.exists('weights'):
        os.makedirs('weights')
    pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
    for step, inputs in enumerate(dataset):
        if step % steps == 0:
            print(f'Epoch {step // steps + 1}/{config.epochs}')
            pb = tf.keras.utils.Progbar(steps, stateful_metrics=['loss'])
        step += 1
        image, y_true = inputs
        loss = distribute_train_step(image, y_true)
        pb.add(1, [('loss', loss)])
        if step % steps == 0:
            model.save_weights(os.path.join("weights", f"model.h5"))
        if step // steps == config.epochs:
            sys.exit("--- Stop Training ---")


if __name__ == '__main__':
    main()
