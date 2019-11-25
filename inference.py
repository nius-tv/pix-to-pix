import cv2
import numpy as np
import yaml

from config import *
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.test_options import TestOptions
from PIL import Image


def get_model(input_dir, output_dir, checkpoints_dir, model_name):
    opt = TestOptions().parse(save=False)
    opt.name = model_name
    opt.label_nc = 0
    opt.dataroot = input_dir
    opt.no_instance = True
    opt.results_dir = output_dir
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.checkpoints_dir = checkpoints_dir

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    return dataset, model


def load_story():
    with open(STORY_FILE_PATH) as f:
        data = f.read()
    return yaml.load(data, Loader=yaml.FullLoader)


def tensor2image(image):
	numpy_image = image.cpu().float().numpy()
	numpy_image = (np.transpose(numpy_image, (1, 2, 0)) + 1) / 2.0 * 255.0
	numpy_image = np.clip(numpy_image, 0, 255)
	numpy_image = numpy_image.astype(np.uint8)
	numpy_image = Image.fromarray(numpy_image)
	numpy_image = np.array(numpy_image)
	numpy_image = numpy_image[:, :, ::-1].copy()

	return numpy_image


if __name__ == '__main__':
    model_name = load_story()['pixToPixModel']
    dataset, model = get_model(STORY_DIR_PATH, OUTPUT_DIR_PATH, MODELS_DIR_PATH, model_name)

    num = len(dataset)
    for i, data in enumerate(dataset):
        output_file_path = '{}/{:010d}.{}'.format(OUTPUT_DIR_PATH, i + 1, IMG_FMT)
        print(i + 1, '/', num, output_file_path)
        # Infer image from input image
        generated = model.inference(data['label'], data['inst'])
        img = tensor2image(generated.data[0])
        # Save image
        assert cv2.imwrite(output_file_path, img)
