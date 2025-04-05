DATASET_NAME = 'DeepFashion'
EMBEDDING_TYPE = 'roberta'
CONFIG_NAME = ''
GPU_ID = '0'
CUDA = True
WORKERS = 6

NET_G = ''
NET_D = ''
STAGE1_G = ''  # Если обучили Stage1

# Папки датасета
DATA_DIR = ''
VAL_DIR = ''

# Количество изображений, которое будет сохраняться при визуализации
VIS_COUNT = 64

# Параметры для GAN
Z_DIM = 100
IMSIZE = 64  # При Stage=1 используем 64x64, при Stage=2 — 256x256
STAGE = 1

TRAIN_FLAG = True
TRAIN_BATCH_SIZE = 256
TRAIN_MAX_EPOCH = 100
TRAIN_SNAPSHOT_INTERVAL = 5
TRAIN_PRETRAINED_MODEL = ''
TRAIN_PRETRAINED_EPOCH = 600
TRAIN_LR_DECAY_EPOCH = 20
TRAIN_DISCRIMINATOR_LR = 0.0002
TRAIN_GENERATOR_LR = 0.0002

TRAIN_COEFF_KL = 2.0

GAN_CONDITION_DIM = 128
GAN_DF_DIM = 96
GAN_GF_DIM = 192
GAN_R_NUM = 2

# Размерность текстового эмбеддинга
TEXT_DIMENSION = 1024

EMBEDDING_FILENAME = "/char-CNN-RNN-embeddings.pickle"
