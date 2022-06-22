# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")
import sys
import time
import ddsp
from ddsp.training import (data, decoders, encoders, models, preprocessing, 
                           train_util, trainers)

import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

sample_rate = 16000
print("import done")

# Get a single example from NSynth.
# Takes a few seconds to load from GCS.
"""
data_provider = data.NSynthTfds(split='test')
dataset = data_provider.get_batch(batch_size=1, shuffle=False).take(1).repeat()
batch = next(iter(dataset))
audio = batch['audio']
n_samples = audio.shape[1]
"""
audio = tf.math.sin(tf.linspace(0, 16000, 16000*3))[tf.newaxis, :]

#todo normalize??
strategy = train_util.get_strategy()


with open(sys.argv[1]) as f:
    lines = f.readlines()

with open(sys.argv[2]) as f:
    lines.extend(f.readlines())

with gin.unlock_config():
  gin.parse_config(lines)

with strategy.scope():
    model = models.get_model()

    trainer = trainers.get_trainer_class()(model, strategy)

data_provider = ddsp.training.data.TFRecordProvider(file_pattern="/home/marko/PycharmProjects/OwnDDSP/data/cello/tf_record*")

# Get a distributed dataset iterator.
dataset = data_provider.get_batch(16, shuffle=True, repeats=-1)
dataset = trainer.distribute_dataset(dataset)
dataset_iter = iter(dataset)

# Build model, easiest to just run forward pass.
trainer.build(next(dataset_iter))
dataset_iter = iter(dataset)

for i in range(300):
  losses = trainer.train_step(dataset_iter)
  res_str = 'step: {}\t'.format(i)
  for k, v in losses.items():
    res_str += '{}: {:.2f}\t'.format(k, v)
  print(res_str)
