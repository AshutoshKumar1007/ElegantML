import os
from  pathlib import Path
BATCH_SIZE = 128
EPOCHS = 150
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
MODEL_PATH = 'model_checkpoint0.plk'
CURR_DIR = Path.cwd()