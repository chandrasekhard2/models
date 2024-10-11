# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train and evaluate the Ranking model."""

from typing import Dict

from absl import app
from absl import flags
from absl import logging
from mlperf_logging import mllog
mllog_constants = mllog.constants
import math
import numpy as np
from time import perf_counter
import mllog_utils
import tensorflow as tf, tf_keras
from official.common import distribute_utils
from official.core import base_trainer
from official.core import train_lib
from official.core import train_utils
from official.recommendation.ranking import common
from official.recommendation.ranking.task import RankingTask
from official.utils.misc import keras_utils
import threading

import sys
import os
sys.path.append(os.path.expanduser('~/fastauc/fastauc/'))
import fast_auc

FLAGS = flags.FLAGS


AUC_THRESHOLD = 0.8275

cpp_auc = fast_auc.CppAuc()

sparse_tensors_info = {
    '0': {'dtype': tf.int64, 'dense_shape': [None, 3]},
    '1': {'dtype': tf.int64, 'dense_shape': [None, 2]},
    '9': {'dtype': tf.int64, 'dense_shape': [None, 7]},
    '10': {'dtype': tf.int64, 'dense_shape': [None, 3]},
    '11': {'dtype': tf.int64, 'dense_shape': [None, 8]},
    '19': {'dtype': tf.int64, 'dense_shape': [None, 12]},
    '20': {'dtype': tf.int64, 'dense_shape': [None, 100]},
    '21': {'dtype': tf.int64, 'dense_shape': [None, 27]},
    '22': {'dtype': tf.int64, 'dense_shape': [None, 10]},
}

# For Dense Tensors
dense_tensors_info = {
    '2': {'dtype': tf.int64, 'shape': [None, 1]},
    '3': {'dtype': tf.int64, 'shape': [None, 2]},
    '4': {'dtype': tf.int64, 'shape': [None, 6]},
    '5': {'dtype': tf.int64, 'shape': [None, 1]},
    '6': {'dtype': tf.int64, 'shape': [None, 1]},
    '7': {'dtype': tf.int64, 'shape': [None, 1]},
    '8': {'dtype': tf.int64, 'shape': [None, 1]},
    '12': {'dtype': tf.int64, 'shape': [None, 1]},
    '13': {'dtype': tf.int64, 'shape': [None, 6]},
    '14': {'dtype': tf.int64, 'shape': [None, 9]},
    '15': {'dtype': tf.int64, 'shape': [None, 5]},
    '16': {'dtype': tf.int64, 'shape': [None, 1]},
    '17': {'dtype': tf.int64, 'shape': [None, 1]},
    '18': {'dtype': tf.int64, 'shape': [None, 1]},
    '23': {'dtype': tf.int64, 'shape': [None, 3]},
    '24': {'dtype': tf.int64, 'shape': [None, 1]},
    '25': {'dtype': tf.int64, 'shape': [None, 1]},
}

def generate_dummy_data(batch_size):
    data = {}

    # Generate random binary values for 'clicked'
    data['clicked'] = tf.random.uniform(
        (batch_size,),
        minval=0,
        maxval=2,  # maxval is exclusive, so 2 will give values 0 or 1
        dtype=tf.int64
    )

    # Adjust 'dense_features' to have values similar to real data
    data['dense_features'] = tf.random.uniform(
        (batch_size, 13),
        minval=0.0,
        maxval=15.0,  # Adjust maxval to match the range in real data
        dtype=tf.float32
    )

    sparse_features = {}

    # Define the shapes and types for each feature
    

    # Create SparseTensors with random indices and values
    for key, info in sparse_tensors_info.items():
        dense_shape = info['dense_shape']
        num_samples = batch_size
        max_columns = dense_shape[1]
        indices = []
        values = []
        for i in range(num_samples):
            # Random number of entries per sample (up to max_columns)
            num_entries = np.random.randint(1, max_columns + 1)
            # Random column indices without duplicates
            col_indices = np.random.choice(max_columns, size=num_entries, replace=False)
            for col_index in col_indices:
                indices.append([i, col_index])
                # Generate random values similar to real data
                value = np.random.randint(1, 8)  # Adjust the range as needed
                values.append(value)
        indices = np.array(indices, dtype=np.int64)
        values = np.array(values, dtype=info['dtype'].as_numpy_dtype)
        sparse_features[key] = tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=dense_shape
        )

    # Create Dense Tensors with random integer values
    for key, info in dense_tensors_info.items():
        # Adjust the range of values to match real data
        sparse_features[key] = tf.random.uniform(
            info['shape'],
            minval=0,
            maxval=20000,  # Adjust maxval as per the real data's range
            dtype=info['dtype']
        )

    data['sparse_features'] = sparse_features
    return data

def data_generator(batch_size):
    while True:
        yield generate_dummy_data(batch_size)

def get_output_signature():
    # Output signature for 'clicked' and 'dense_features'
    output_signature = {
        'clicked': tf.TensorSpec(shape=(None,), dtype=tf.int64),
        'dense_features': tf.TensorSpec(shape=(None, 13), dtype=tf.float32),
        'sparse_features': {}
    }

    # Output signature for sparse features
    for key, info in sparse_tensors_info.items():
        output_signature['sparse_features'][key] = tf.SparseTensorSpec(
            shape=[None, info['dense_shape'][1]],
            dtype=info['dtype']
        )

    # Output signature for dense tensors in sparse_features
    for key, info in dense_tensors_info.items():
        output_signature['sparse_features'][key] = tf.TensorSpec(
            shape=[None] + info['shape'][1:],
            dtype=info['dtype']
        )

    return output_signature

class RankingTrainer(base_trainer.Trainer):
  """A trainer for Ranking Model.

  The RankingModel has two optimizers for embedding and non embedding weights.
  Overriding `train_loop_end` method to log learning rates for each optimizer.
  """
  mllogger = mllog.get_mllogger()
  auc_threshold = AUC_THRESHOLD
  
  def initialize(self):
    super().initialize()
    self.mllogger.end(mllog_constants.INIT_STOP)
    self.mllogger.start(mllog_constants.RUN_START)
    self._step_per_epoch = math.ceil(mllog_utils._NUM_TRAIN_EXAMPLES / self.config.task.train_data.global_batch_size) 
    self._start_time = perf_counter()
    self._total_time = -1.0
    self._throughput = -1.0
    

  def train_loop_begin(self):
    self.join()
    if self.epoch_num == 0.:
      self.mllogger.start(
        key=mllog_constants.EPOCH_START,
        metadata={mllog_constants.EPOCH_NUM: self.epoch_num},
      )
    else:
      self.mllogger.start(
           key=mllog_constants.RUN_START,
           metadata={mllog_constants.EPOCH_NUM: self.epoch_num},
        )

  @property
  def epoch_num(self) -> float:
    return (self.global_step / self.config.trainer.train_steps).numpy()
  
  def _compute_stats(self):
    self._total_time = perf_counter() - self._start_time
    # _compute_stats after finishing the global_step'th step
    self._throughput = (self.global_step.numpy() + 1) * self.config.task.train_data.global_batch_size / self._total_time

  def train_loop_end(self) -> Dict[str, float]:
    """See base class."""

    self.join()
    self._compute_stats()
    print(f"Finish {self.global_step.numpy() + 1} iterations with "
          f"batchsize: {self.config.task.train_data.global_batch_size} in {self._total_time:.2f}s."
          )
    self.mllogger.event(
      key="tracked_throughtput",
      value={"throughput": self._throughput},
      metadata={mllog_constants.EPOCH_NUM: self.epoch_num},
    )
    if self.global_step.numpy() % self._step_per_epoch == 0:
      self.mllogger.end(
        key=mllog_constants.EPOCH_STOP,
        metadata={
          mllog_constants.EPOCH_NUM: self.epoch_num,
        },
      )

  def eval_begin(self):
    self.join()
    self.mllogger.start(
        key=mllog_constants.EVAL_START,
        metadata={mllog_constants.EPOCH_NUM: self.epoch_num},
    )


  def run_stop(self):
    self.join()
    self.mllogger.end(
        key=mllog_constants.RUN_STOP,
        metadata={
          'status': 'success',
          mllog_constants.EPOCH_NUM: self.epoch_num,
          },
        )
    self.mllogger.end(
        key=mllog_constants.EPOCH_STOP,
        metadata={
          mllog_constants.EPOCH_NUM: self.epoch_num,
        },
      )
  
  def calculate_roc_async(self, outputs):
    """Calculate ROC metrics in a separate thread."""
    outputs_np = outputs.numpy()
    labels_np = outputs_np[:, 0::2, :]
    predictions_np = outputs_np[:, 1::2, :]
    
    labels_np = labels_np.reshape(-1)
    mask = labels_np != -1
    labels_np = labels_np[mask]
    predictions_np = predictions_np.reshape(-1)
    predictions_np = predictions_np[mask]
    if self._stop_training:
        return
    auc = cpp_auc.roc_auc_score(labels_np.astype(np.bool), predictions_np.astype(np.float32))
    self.mllogger.end(
      key=mllog_constants.EVAL_ACCURACY,
      value=auc,
      metadata={mllog_constants.EPOCH_NUM: self.epoch_num},
    )
    self.mllogger.end(
      key=mllog_constants.EVAL_STOP,
      metadata={mllog_constants.EPOCH_NUM: self.epoch_num},
    )
    if not self._stop_training and auc >= self.auc_threshold:
      self._stop_training = True
    return auc

  def eval_end(self, aggregated_logs=None):
    self.join()
    logs = {}
    roc_thread = None
    if not self._stop_training:
        roc_thread = threading.Thread(target=self.calculate_roc_async, args=(aggregated_logs,))
        roc_thread.start()
    return roc_thread
    

def main(_) -> None:
  """Train and evaluate the Ranking model."""
  mllogger = mllog.get_mllogger()
  mllogger.event(mllog_constants.CACHE_CLEAR)
  mllogger.start(mllog_constants.INIT_START)
  params = train_utils.parse_configuration(FLAGS)

  tf.profiler.experimental.server.start(6008)
  mllog_utils.init_print(params, FLAGS.seed)
  mode = FLAGS.mode
  tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  if FLAGS.seed is not None:
    logging.info('Setting tf seed.')
    tf.random.set_seed(FLAGS.seed)

  task = RankingTask(
      params=params.task,
      trainer_config=params.trainer,
      logging_dir=model_dir,
      steps_per_execution=params.trainer.steps_per_loop,
      name='RankingTask')

  enable_tensorboard = params.trainer.callbacks.enable_tensorboard

  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  with strategy.scope():
    model = task.build_model()

  def get_dataset_fn(params):
    return lambda input_context: task.build_inputs(params, input_context)

  train_dataset = None
  if 'train' in mode:
    train_dataset = strategy.distribute_datasets_from_function(
        get_dataset_fn(params.task.train_data),
        options=tf.distribute.InputOptions(experimental_fetch_to_device=False))

  validation_dataset = None
  if 'eval' in mode:
    validation_dataset = strategy.distribute_datasets_from_function(
        get_dataset_fn(params.task.validation_data),
        options=tf.distribute.InputOptions(experimental_fetch_to_device=False))

  #dummy input fn

  #step_fn = tf.function(model.train_step)
  '''
  print('started dummy train step')
  for i in model.optimizer.optimizers:
      i.learning_rate = 0.0
  with strategy.scope():
      @tf.function
      def train_loop(inputs):
          strategy.run(model.train_step, args=(inputs,))
          strategy.run(model.test_step, args=(inputs,))

      inputs = generate_dummy_data(2112)
      #print(inputs)
      train_loop(inputs)
  print('finished_dummy_train_step')  
  for i in model.optimizer.optimizers:
      i.learning_rate = 0.0034
  '''
    
  if params.trainer.use_orbit:
    with strategy.scope():
      checkpoint_exporter = train_utils.maybe_create_best_ckpt_exporter(
          params, model_dir)
      trainer = RankingTrainer(
          config=params,
          task=task,
          model=model,
          optimizer=model.optimizer,
          train='train' in mode,
          evaluate='eval' in mode,
          train_dataset=train_dataset,
          validation_dataset=validation_dataset,
          checkpoint_exporter=checkpoint_exporter)
      trainer.initialize()

    
    for i in trainer.model.optimizer.optimizers:
      i.learning_rate = 0.0
    print('dummy train step started')
    with strategy.scope():
      #@tf.function
      def train_loop(dataset_iterator):
          trainer.train_step(dataset_iterator)
          print('dunny train done')
          trainer.eval_step(dataset_iterator)

      #inputs = generate_dummy_data(2112)
      dataset = tf.data.Dataset.from_generator(
                    data_generator,
                    output_signature=get_output_signature()
                )
      #print(inputs)
      dataset = dataset.repeat()
      dataset_iterator =  tf.nest.map_structure(iter, dataset)
      train_loop(dataset_iterator)

    print('dummy train finished')
    for i in trainer.model.optimizer.optimizers:
      i.learning_rate = 0.0034

    train_lib.run_experiment(
        distribution_strategy=strategy,
        task=task,
        mode=mode,
        params=params,
        model_dir=model_dir,
        trainer=trainer)

  else:  # Compile/fit
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if latest_checkpoint:
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=params.trainer.max_to_keep,
        step_counter=model.optimizer.iterations,
        checkpoint_interval=params.trainer.checkpoint_interval)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

    time_callback = keras_utils.TimeHistory(
        params.task.train_data.global_batch_size,
        params.trainer.time_history.log_steps,
        logdir=model_dir if enable_tensorboard else None)
    callbacks = [checkpoint_callback, time_callback]

    if enable_tensorboard:
      tensorboard_callback = tf.keras.callbacks.TensorBoard(
          log_dir=model_dir,
          update_freq=min(1000, params.trainer.validation_interval),
          profile_batch=FLAGS.profile_steps)
      callbacks.append(tensorboard_callback)

    num_epochs = (params.trainer.train_steps //
                  params.trainer.validation_interval)
    current_step = model.optimizer.iterations.numpy()
    initial_epoch = current_step // params.trainer.validation_interval

    eval_steps = params.trainer.validation_steps if 'eval' in mode else None

    if mode in ['train', 'train_and_eval']:
      logging.info('Training started')
      history = model.fit(
          train_dataset,
          initial_epoch=initial_epoch,
          epochs=num_epochs,
          steps_per_epoch=params.trainer.validation_interval,
          validation_data=validation_dataset,
          validation_steps=eval_steps,
          callbacks=callbacks,
      )
      model.summary()
      logging.info('Train history: %s', history.history)
    elif mode == 'eval':
      logging.info('Evaluation started')
      validation_output = model.evaluate(validation_dataset, steps=eval_steps)
      logging.info('Evaluation output: %s', validation_output)
    else:
      raise NotImplementedError('The mode is not implemented: %s' % mode)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  common.define_flags()
  app.run(main)

