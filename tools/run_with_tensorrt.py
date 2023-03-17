from mmdeploy_python import PoseDetector
import cv2
# 第一个参数是模型转换命令中 --work-dir选项的值
detector = PoseDetector(model_path='work-dirs/mmpose/topdown/hrnet/trt', device_name='cuda', device_id=0)

# 需要读取自己路径下的图片
img = cv2.imread('demo/resources/human-pose.jpg')
result = detector(img)
_, point_num, _ = result.shape
points = result[:, :, :2].reshape(point_num, 2)
for [x, y] in points.astype(int):
    cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

cv2.imwrite('output_pose.png', img)
