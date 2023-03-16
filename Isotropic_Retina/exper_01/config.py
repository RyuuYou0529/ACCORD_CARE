from os import path
import sys

GLOBAL_CONFIG_PATH = 'e:\\Project\\ACCORD_CARE'
sys.path.append(GLOBAL_CONFIG_PATH)

import global_config as GC

# Data path
DATASET_BASE_PATH = path.join(GC.CARE_DATA_BASE_PATH, 'Isotropic_Retina')

TRAIN_DATASET_PATH = path.join(DATASET_BASE_PATH, 'train_data/data_label.npz')

# Checkpoint Path
CHECKPOINT_PATH = './checkpoint'
FX_MODEL_PATH = path.join(CHECKPOINT_PATH, 'FX/FX.hdf5')
DX_MODEL_PATH = path.join(CHECKPOINT_PATH, 'DX/DX.hdf5')

# Log Path
FX_LOG_PATH = path.join(CHECKPOINT_PATH, 'FX/logs')
DX_LOG_PATH = path.join(CHECKPOINT_PATH, 'DX/logs')