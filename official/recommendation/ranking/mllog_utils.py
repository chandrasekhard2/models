"""mllog_utils"""
# cloud
from mlperf_logging import mllog
mllog_constants = mllog.constants

_NUM_TRAIN_EXAMPLES = 4195197692
_NUM_EVAL_EXAMPLES = 89137319


def init_print(params, seed):
  mllogger = mllog.get_mllogger()
  mllogger.event(mllog_constants.CACHE_CLEAR)
  mllogger.start(mllog_constants.INIT_START)
  mllogger.event(mllog_constants.SUBMISSION_BENCHMARK, mllog_constants.DLRM_DCNv2)
  mllogger.event(mllog_constants.SUBMISSION_ORG, 'Google')
  mllogger.event(mllog_constants.SUBMISSION_PLATFORM, 'tpu-v5p')
  mllogger.event(mllog_constants.SUBMISSION_STATUS, mllog_constants.CLOUD)
  mllogger.event(mllog_constants.SUBMISSION_DIVISION, mllog_constants.CLOSED)
  mllogger.event(mllog_constants.GLOBAL_BATCH_SIZE, params.task.train_data.global_batch_size)

  mllogger.event(mllog_constants.TRAIN_SAMPLES, _NUM_TRAIN_EXAMPLES)
  mllogger.event(mllog_constants.EVAL_SAMPLES, _NUM_EVAL_EXAMPLES)

  assert params.trainer.optimizer_config.embedding_optimizer == params.trainer.optimizer_config.dense_optimizer, "optimizer mismatch"
  assert params.trainer.optimizer_config.embedding_optimizer == "Adagrad", "optimizer type error"
  mllogger.event(mllog_constants.OPT_NAME, mllog_constants.ADAGRAD)

  mllogger.event(mllog_constants.OPT_BASE_LR, params.trainer.optimizer_config.lr_config.learning_rate)

  mllogger.event(mllog_constants.OPT_ADAGRAD_INITIAL_ACCUMULATOR_VALUE, params.trainer.optimizer_config.initial_accumulator_value)
  mllogger.event(mllog_constants.OPT_ADAGRAD_EPSILON, params.trainer.optimizer_config.epsilon)

  # learning rate is a constant number, warmup and decay should be deactivated
  mllogger.event(mllog_constants.OPT_ADAGRAD_LR_DECAY, 0)
  mllogger.event(mllog_constants.OPT_WEIGHT_DECAY, 0)

  mllogger.event(mllog_constants.OPT_LR_WARMUP_STEPS, params.trainer.optimizer_config.lr_config.warmup_steps)
  mllogger.event(mllog_constants.OPT_LR_DECAY_START_STEP, params.trainer.optimizer_config.lr_config.decay_start_steps)
  mllogger.event(mllog_constants.OPT_LR_DECAY_STEPS, params.trainer.optimizer_config.lr_config.decay_steps)

  mllogger.event(mllog_constants.GRADIENT_ACCUMULATION_STEPS, 1)
  if seed:
    mllogger.event(mllog_constants.SEED, seed)
