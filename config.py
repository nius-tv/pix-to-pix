import os

FPS = float(os.environ['FPS'])
IMG_FMT = os.environ['IMG_FMT']
MODELS_DIR_PATH = '/models/pix-to-pix'
OUTPUT_DIR_PATH = '/data/inferred'
SCALED_VIDEO_RESOLUTION = (512, 1024) # width, height
STORY_DIR_PATH = '/data' # calls test_A
STORY_FILE_PATH = '{}/story.yaml'.format(STORY_DIR_PATH)
