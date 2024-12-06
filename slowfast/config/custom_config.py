#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.

    # -----------------------------------------------------------------------------
    # Config definition
    # -----------------------------------------------------------------------------
    # _C = CfgNode()

    # ---------------------------------------------------------------------------- #
    # Basic options.
    # ---------------------------------------------------------------------------- #
    # Number of GPUs to use (applies to both training and testing).
    _C.NUM_GPUS = 1

    # Output basedir.
    _C.OUTPUT_DIR = "."

    # Note that non-determinism may still be present due to non-deterministic
    # operator implementations in GPU operator libraries.
    _C.RNG_SEED = 1

    # # ---------------------------------------------------------------------------- #
    # # Training options.
    # # ---------------------------------------------------------------------------- #
    # # _C.TRAIN = CfgNode()

    # # If True Train the model, else skip training.
    # _C.TRAIN.ENABLE = True

    # # Kill training if loss explodes over this ratio from the previous 5 measurements.
    # # Only enforced if > 0.0
    # _C.TRAIN.KILL_LOSS_EXPLOSION_FACTOR = 0.0

    # # Dataset.
    # _C.TRAIN.DATASET = "Ucfcrime"

    # # Total mini-batch size.
    # _C.TRAIN.BATCH_SIZE = 4

    # # Evaluate model on test data every eval period epochs.
    # _C.TRAIN.EVAL_PERIOD = 10

    # # Save model checkpoints every checkpoints period epochs.
    # _C.TRAIN.CHECKPOINT_PERIOD = 10

    # # Resume training from the latest checkpoints in the output directory.
    # _C.TRAIN.AUTO_RESUME = True

    # # Path to the checkpoints to load the initial weight.
    # _C.TRAIN.CHECKPOINT_FILE_PATH = ""

    # # Path to list of extracted videos features
    # _C.TRAIN_FEATURES_LIST = ''

    # # Checkpoint types include `caffe2` or `pytorch`.
    # _C.TRAIN.CHECKPOINT_TYPE = "pytorch"

    # # # If True, perform inflation when loading checkpoints.
    # # _C.TRAIN.CHECKPOINT_INFLATE = False

    # # # If True, reset epochs when loading checkpoints.
    # # _C.TRAIN.CHECKPOINT_EPOCH_RESET = False

    # # ---------------------------------------------------------------------------- #
    # # Testing options.
    # # ---------------------------------------------------------------------------- #
    # # _C.TEST = CfgNode()

    # # If True, test the model, else skip the testing.
    # _C.TEST.ENABLE = True

    # # Dataset for testing.
    # _C.TEST.DATASET = "Ucfcrime"

    # # Total mini-batch size
    # _C.TEST.BATCH_SIZE = 1

    # # Path to the checkpoints to load the initial weight.
    # _C.TEST.CHECKPOINT_FILE_PATH = ""

    # # Checkpoint types include `caffe2` or `pytorch`.
    # _C.TEST.CHECKPOINT_TYPE = "pytorch"

    # # # Path to saving prediction results file.
    # # _C.TEST.SAVE_RESULTS_PATH = ""

    # # # Path to list of extracted videos features
    # # _C.TEST.TEST_FEATURES_LIST = ''

    # # ---------------------------------------------------------------------------- #
    # Data options.
    # ---------------------------------------------------------------------------- #
    # _C.DATA = CfgNode()

    # Type of feature extractor input (i3d or c3d).
    _C.DATA.FEATURE_EXTRACTOR = ""

    # Size of the features.
    _C.DATA.FEATURE_SIZE = 2048

    # # The video sampling rate of the input clip.
    # _C.DATA.SAMPLING_RATE = 8

    # # Input videos may have different fps, convert it to the target video fps before
    # # frame sampling. (For feature extractor)
    # _C.DATA.TARGET_FPS = 30

    # The path to the feature extractor directory.
    _C.DATA.FEATURE_EXTRACTOR_CKPT = ""

    # # ---------------------------------------------------------------------------- #
    # # Optimizer options.
    # # ---------------------------------------------------------------------------- #
    # # _C.SOLVER = CfgNode()

    # # Base learning rate.
    # _C.SOLVER.LR = 0.001

    # # Step size for 'exp' and 'cos' policies (in epochs).
    # _C.SOLVER.STEP_SIZE = 1

    # # Steps for 'steps_' policies (in epochs).
    # _C.SOLVER.STEPS = []

    # # Learning rates for 'steps_' policies.
    # _C.SOLVER.LRS = []

    # # Maximal number of epochs.
    # _C.SOLVER.MAX_EPOCH = 100

    # # FOR REFERENCE (FROM Slowfast GITHUB)
    # # Momentum.
    # _C.SOLVER.MOMENTUM = 0.9

    # # Momentum dampening.
    # _C.SOLVER.DAMPENING = 0.0

    # # Nesterov momentum.
    # _C.SOLVER.NESTEROV = True

    # # L2 regularization.
    # _C.SOLVER.WEIGHT_DECAY = 1e-4

    # # Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
    # _C.SOLVER.WARMUP_FACTOR = 0.1

    # # Gradually warm up the SOLVER.BASE_LR over this number of epochs.
    # _C.SOLVER.WARMUP_EPOCHS = 0.0

    # # The start learning rate of the warm up.
    # _C.SOLVER.WARMUP_START_LR = 0.01

    # # Optimization method.
    # _C.SOLVER.OPTIMIZING_METHOD = "sgd"

    # # Base learning rate is linearly scaled with NUM_SHARDS.
    # _C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

    # # If True, start from the peak cosine learning rate after warm up.
    # _C.SOLVER.COSINE_AFTER_WARMUP = False

    # # If True, perform no weight decay on parameter with one dimension (bias term, etc).
    # _C.SOLVER.ZERO_WD_1D_PARAM = False

    # # Clip gradient at this value before optimizer update
    # _C.SOLVER.CLIP_GRAD_VAL = None

    # # Clip gradient at this norm before optimizer update
    # _C.SOLVER.CLIP_GRAD_L2NORM = None

    # # LARS optimizer
    # _C.SOLVER.LARS_ON = False

    # # The layer-wise decay of learning rate. Set to 1. to disable.
    # _C.SOLVER.LAYER_DECAY = 1.0

    # # Adam's beta
    # _C.SOLVER.BETAS = (0.9, 0.999)

    # ---------------------------------------------------------------------------- #
    # Model options.
    # ---------------------------------------------------------------------------- #
    # _C.MODEL = CfgNode()

    # Model name
    _C.MODEL.MODEL_NAME = "RTFM"

    # The number of classes to predict for the model (Normal OR Abnormal).
    _C.MODEL.NUM_CLASSES = 2

    # Loss function.
    _C.MODEL.LOSS_FUNC = "SigmoidMAELoss"

    # ---------------------------------------------------------------------------- #
    # Common train/test dataloader options.
    # ---------------------------------------------------------------------------- #
    # _C.DATA_LOADER = CfgNode()

    # Number of data loader workers per training process.
    _C.DATA_LOADER.NUM_WORKERS = 8

    # Load data to pinned host memory.
    _C.DATA_LOADER.PIN_MEMORY = True

    # Enable multi thread decoding.
    _C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

    # ---------------------------------------------------------------------------- #
    # Demo options.
    # ---------------------------------------------------------------------------- #
    # _C.DEMO = CfgNode()

    # Number of processes to run video visualizer.
    _C.DEMO.NUM_VIS_INSTANCES = 2

    # Whether to run in with multi-threaded video reader.
    _C.DEMO.THREAD_ENABLE = False

    # # Number of overlapping frames between 2 consecutive clips.
    # # Increase this number for more frequent action predictions.
    # # The number of overlapping frames cannot be larger than
    # # half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
    # _C.DEMO.BUFFER_SIZE = 0

    # Enable Demo mode.
    _C.DEMO.ENABLE = False

    # Demo weight.
    _C.DEMO.WEIGHT = ''

    pass
