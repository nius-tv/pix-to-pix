import os

IMG_FMT = os.environ.get('IMG_FMT')
MODELS_DIR_PATH = '/models/pix-to-pix'
OUTPUT_DIR_PATH = '/data/inferred'
STORY_DIR_PATH = '/data' # calls test_A
STORY_FILE_PATH = '{}/story.yaml'.format(STORY_DIR_PATH)
