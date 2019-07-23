import argparse
import keras
from keras.preprocessing.image import img_to_array
import logging
import numpy as np

import keras_contrib

from style_transfer import layers

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('stylize_image')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stylize an image using a trained model.'
    )

    parser.add_argument(
        '--model-checkpoint', type=str, required=True,
        help='Checkpoint from a trained Style Transfer Network.'
    )

    parser.add_argument(
        '-d', '--device', default='normal_cam'
    ) # normal_cam / jetson_nano_raspi_cam


    args = parser.parse_args()

    logger.info('Loading model from %s' % args.model_checkpoint)
    custom_objects = {
        'InstanceNormalization':
            keras_contrib.layers.InstanceNormalization,
        'DeprocessStylizedImage': layers.DeprocessStylizedImage
    }
    transfer_net = keras.models.load_model(
        args.model_checkpoint,
        custom_objects=custom_objects
    )

    image_size = transfer_net.input_shape[1:3]

    inputs = [transfer_net.input, keras.backend.learning_phase()]
    outputs = [transfer_net.output]

    transfer_style = keras.backend.function(inputs, outputs)

    if args.device == 'normal_cam':
        cam = cv2.VideoCapture(0)
    elif args.device == 'jetson_nano_raspi_cam':
        GST_STR = 'nvarguscamerasrc \
            ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)21/1 \
            ! nvvidconv ! video/x-raw, width=(int)1280, height=(int)960, format=(string)BGRx \
            ! videoconvert \
            ! appsink'
        cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
    while True:
        ret, img = cam.read()
        if ret != True:
            break

        img = cv2.resize(img, (image_size[0], image_size[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_to_array(img)
        img = np.array(img)[:, :, :3]
        img = np.expand_dims(img, axis=0)

        out_img = transfer_style([img, 1])[0]

        out_img = cv2.cvtColor(out_img[0], cv2.COLOR_RGB2BGR)
        out_img = cv2.resize(out_img, (640, 480))
        cv2.imshow('stylize movie', np.uint8(out_img))

        key = cv2.waitKey(10)
        if key == 27: # ESC
            break
    cam.release()
    cv2.destroyAllWindows()
