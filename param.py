# param.py

SEED_DEFAULT = 42
KEYWORDS =  list(set(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'] + ['unknown', 'silence']))

# Command Speech Dataset Paths
DATASET_PATH = '/home/luciano/Downloads/speech_commands_v0.02/'
TESTSET_PATH = '/home/luciano/Downloads/speech_commands_test_set_v0.02/'

# Spectrogram parameters
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
POWER = 2.0
F_MIN = 50.0
F_MAX = 7500.0
HOP_LENGTH=round(SAMPLE_RATE*0.01)
WIN_LENGTH=round(SAMPLE_RATE*0.025)
DURATION_SEC = 1.0

# Model training 
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

# Number of model to be trained
NUM_MODELS = 25

# Specific noise for continuous learning
NOISE_TYPES=["car_horn_background","dog_bark_foreground","street_music_foreground"]
SNR_VALUES = range(-3,25,3)