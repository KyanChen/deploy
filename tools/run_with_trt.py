import time

import onnxruntime
import numpy as np
import cv2


def form_input(img):
    img = cv2.resize(img, (512, 512))[:, :, ::-1]
    img = (img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    img = np.transpose(img, axes=(2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_palette(num_classes=3):
    # state = np.random.get_state()
    # # random color
    # np.random.seed(42)
    # palette = np.random.randint(0, 256, size=(num_classes, 3))
    # np.random.set_state(state)
    palette = [[0, 0, 0], [0, 0, 255], [0, 255, 0]]
    return [tuple(c) for c in palette]

o_model_path = "E:/project/Template_detection/pytorch_net/test_model.onnx"
img_file = 'test_cloud_img.jpg'

onet_session = onnxruntime.InferenceSession(o_model_path)

img = cv2.imread(img_file)

time_t = 0
for i in range(20):
    t_s = time.time()
    inputs = {onet_session.get_inputs()[0].name: form_input(img)}
    outs = onet_session.run(None, inputs)

    time_t += time.time() - t_s
print(time_t / 20)
seg = outs[0]

palette = get_palette()
color_seg = np.zeros((512, 512, 3), dtype=np.uint8)
for label, color in enumerate(palette):
    color_seg[seg == label, :] = color
# convert to BGR
color_seg = color_seg[..., ::-1]

img = img * 0.2 + color_seg * 0.8
img = img.astype(np.uint8)
cv2.imwrite('output_segmentation_onnx.png', img)

