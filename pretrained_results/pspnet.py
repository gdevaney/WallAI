from matplotlib.pyplot import imshow

from models.models import SegmentationModule, build_encoder, build_decoder
from src.eval import segment_image
from utils.constants import DEVICE

import os

# path to image on which to run wall segmentation
path_image_folder = '/Users/yoon/Documents/gatech/spring24/comp_vision/ADEChallengeData2016/images/training'
image_name = 'ADE_train_00000017.jpg'

path_image_folder = '/Users/yoon/Documents/gatech/spring24/comp_vision/ADEChallengeData2016/images'
image_name = '/Users/yoon/Downloads/sample11.jpeg'

path_image = os.path.join(path_image_folder, image_name)

# Model weights (encoder and decoder)
weights_encoder = '/Users/yoon/Documents/gatech/spring24/WallSegmentation/model_weights/Transfer learning - entire decoder/transfer_encoder.pth'
weights_decoder = '/Users/yoon/Documents/gatech/spring24/WallSegmentation/model_weights/Transfer learning - entire decoder/transfer_decoder.pth'
