import onnxruntime
import numpy as np
import cv2


def form_input(img):
    img = cv2.resize(img, (512, 512))[:, :, ::-1]
    img = (img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    img = np.transpose(img, axes=(2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


o_model_path = "E:/project/Template_detection/pytorch_net/test_model.onnx"
img_file = ''


# 模型图片输入的预处理，可以抄pth或者pt的图片处理
def get_img_tensor(img_path, use_cuda, get_size=False):
    img = Image.open(img_path)
    original_w, original_h = img.size

    img_size = (224, 224)  # crop image to (224, 224)
    img.thumbnail(img_size, Image.ANTIALIAS)
    img = img.convert('RGB')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        # transforms.RandomResizedCrop(img_size[0]), #随机裁剪
        # transforms.RandomHorizontalFlip(), #平移
        transforms.ToTensor(),
        normalize,
    ])
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    if use_cuda:
        img_tensor = img_tensor.cuda()
    if get_size:
        return img_tensor, original_w, original_w
    else:
        return img_tensor


onet_session = onnxruntime.InferenceSession(o_model_path)

img = cv2.imread(img_file)

for i in range(10):
    inputs = {onet_session.get_inputs()[0].name: to_numpy(img)}
    outs = onet_session.run(None, inputs)
preds = outs[0]
# print(preds)
# print(preds.shape)
tops_type = [3, 5, 10]  # 输出topk的值
# np.argsort(a)  返回的是元素值从小到大排序后的索引值的数组   [::-1] 将元素倒序排列
indexes = np.argsort(preds[0])[::-1]
print('np.argsort(preds[0]):', np.argsort(preds[0]))
print('np.argsort(preds[0][::-1]):', np.argsort(preds[0])[::-1])
# print(indexes)
# print(indexes[:10])
for topk in tops_type:
    idxes = indexes[:topk]
    print('[ Top%d Attribute Prediction ]' % topk)
    for idx in idxes:
        print(class_names[idx])