import os

BASE_PATH = os.path.join(os.path.abspath(os.getcwd()), "use")
PRETRAIN_MODEL_PATH = os.path.join(BASE_PATH, "cd_res", 'pretrain')
os.makedirs(PRETRAIN_MODEL_PATH, exist_ok=True)
DATA_PATH = os.path.join(BASE_PATH, "changedetection/SceneChangeDet/BCD")
os.makedirs(DATA_PATH, exist_ok=True)
TRAIN_DATA_PATH = os.path.join(DATA_PATH)
TRAIN_LABEL_PATH = os.path.join(DATA_PATH)
TRAIN_TXT_PATH = os.path.join(TRAIN_DATA_PATH, 'train.txt')
VAL_DATA_PATH = os.path.join(DATA_PATH)
VAL_LABEL_PATH = os.path.join(DATA_PATH)
VAL_TXT_PATH = os.path.join(VAL_DATA_PATH,'val.txt')
SAVE_PATH = os.path.join(BASE_PATH, 'cdout/bone/resnet50/BCD2')
SAVE_CKPT_PATH = os.path.join(SAVE_PATH, 'ckpt')
os.makedirs(SAVE_CKPT_PATH, exist_ok=True)
SAVE_PRED_PATH = os.path.join(SAVE_PATH, 'prediction')
os.makedirs(SAVE_PRED_PATH, exist_ok=True)
TRAINED_BEST_PERFORMANCE_CKPT = os.path.join(SAVE_CKPT_PATH, 'model_best.pth')
INIT_LEARNING_RATE = 1e-4
DECAY = 5e-5
MOMENTUM = 0.90
MAX_ITER = 40000
BATCH_SIZE = 1
TRANSFROM_SCALES= (256,256)
T0_MEAN_VALUE = (98.62,113.27,123.59)
T1_MEAN_VALUE = (117.38 ,123.09 , 123.20)
