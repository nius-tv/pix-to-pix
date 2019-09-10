import numpy as np

from lib.settings import IMG_FMT, TMP_DIR_FMT
from lib.storage import Storage

from PIL import Image
from data.data_loader import CreateDataLoader
from models.models import create_model
from options.test_options import TestOptions


class PixToPixInfer(object):

    def __init__(self):
        self.storage = Storage()

    def _get_model(self, dataroot, results_dir):
        opt = TestOptions().parse(save=False)
        opt.name = 'nius'
        opt.label_nc = 0
        opt.dataroot = dataroot
        opt.no_instance = True
        opt.results_dir = results_dir
        opt.nThreads = 1   # test code only supports nThreads = 1
        opt.batchSize = 1  # test code only supports batchSize = 1
        opt.serial_batches = True  # no shuffle
        opt.no_flip = True  # no flip
        opt.checkpoints_dir = '/service/pixtopix/checkpoints'

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        model = create_model(opt)

        return dataset, model

    def _tensor2im(self, img):
        image_numpy = img.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        image_numpy = image_numpy.astype(np.uint8)
        image_numpy = Image.fromarray(image_numpy)
        image_numpy = np.array(image_numpy)
        image_numpy = image_numpy[:, :, ::-1].copy()
        return image_numpy

    def process(self, storage_input_path, storage_output_path):
        dataset, model = self._get_model(dataroot=TMP_DIR_FMT.format(storage_input_path),
                                         results_dir=TMP_DIR_FMT.format(storage_output_path))
        num = len(dataset)

        for i, data in enumerate(dataset):
            output_file_path = '{}/{:010d}.{}'.format(storage_output_path, i + 1, IMG_FMT)
            print(i + 1, '/', num, output_file_path)
            # Infer image from input image
            generated = model.inference(data['label'], data['inst'])
            img = self._tensor2im(generated.data[0])
            # Save image to storage
            self.storage.save_img(img, output_file_path)
