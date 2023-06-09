import os
import time
import torch
import torchvision
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def form_input(img):
    img = cv2.resize(img, (512, 512))[:, :, ::-1]
    img = (img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    img = np.transpose(img, axes=(2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def build_engine(onnx_file_path, engine_file_path, max_batch_size=1, fp16_mode=False, save_engine=False):
    """
    Args:
      max_batch_size: 预先指定大小好分配显存
      fp16_mode:      是否采用FP16
      save_engine:    是否保存引擎
    return:
      ICudaEngine
    """
    if os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化

    # 如果是动态输入，需要显式指定EXPLICIT_BATCH
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # builder创建计算图 INetworkDefinition
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(EXPLICIT_BATCH) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, TRT_LOGGER) as parser:  # 使用onnx的解析器绑定计算图

        builder.max_workspace_size = 1 << 60  # ICudaEngine执行时GPU最大需要的空间
        builder.max_batch_size = max_batch_size  # 执行时最大可以使用的batchsize
        builder.fp16_mode = fp16_mode
        config.max_workspace_size = 1 << 30  # 1G

        # 动态输入profile优化
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 512, 512), (8, 3, 512, 512), (8, 3, 512, 512))
        config.add_optimization_profile(profile)

        # 解析onnx文件，填充计算图
        if not os.path.exists(onnx_file_path):
            quit("ONNX file {} not found!".format(onnx_file_path))
        print('loading onnx file from path {} ...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print("Begining onnx file parsing")
            if not parser.parse(model.read()):  # 解析onnx文件
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))  # 打印解析错误日志
                return None

        last_layer = network.get_layer(network.num_layers - 1)
        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))
        print("Completed parsing of onnx file")

        # 使用builder创建CudaEngine
        print("Building an engine from file{}' this may take a while...".format(onnx_file_path))
        # engine=builder.build_cuda_engine(network)    # 非动态输入使用
        engine = builder.build_engine(network, config)  # 动态输入使用
        print("Completed creating Engine")
        if save_engine:
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
        return engine


def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size  # 非动态输入
        # size = trt.volume(engine.get_binding_shape(binding))                       # 动态输入
        size = abs(size)  # 上面得到的size(0)可能为负数，会导致OOM
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)  # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # cuda分配空间
        bindings.append(int(device_mem))  # binding在计算图中的缓冲地址
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    # 如果创建network时显式指定了batchsize，使用execute_async_v2, 否则使用execute_async
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

def get_palette():
    palette = [[0, 0, 0], [0, 0, 255], [0, 255, 0]]
    return [tuple(c) for c in palette]


if __name__ == '__main__':
    onnx_file_path = "results/onnxdeployfull/end2end.onnx"
    fp16_mode = False
    max_batch_size = 1
    trt_engine_path = "results/onnxdeployfull/end2end_fp16.engine"
    img_file = 'test_cloud_img.jpg'
    img = cv2.imread(img_file)

    # 1.创建cudaEngine
    engine = build_engine(onnx_file_path, trt_engine_path, max_batch_size, fp16_mode)

    # 2.将引擎应用到不同的GPU上配置执行环境
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # 3.推理
    output_shape = (max_batch_size, 1, 512, 512)
    dummy_input = np.ones([1, 3, 512, 512], dtype=np.float32)
    dummy_input = form_input(img)
    inputs[0].host = dummy_input.reshape(-1)

    # # 如果是动态输入，需以下设置
    # context.set_binding_shape(0, dummy_input.shape)

    time_t = 0
    max_try = 100
    for _ in range(max_try):
        t_s = time.time()
        trt_outputs = inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        time_t += time.time() - t_s
    # 由于tensorrt输出为一维向量，需要reshape到指定尺寸
    feat = postprocess_the_outputs(trt_outputs[0], output_shape)

    seg = feat[0][0]
    palette = get_palette()
    color_seg = np.zeros((512, 512, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    img = cv2.resize(img, (512, 512))
    img = img * 0.2 + color_seg * 0.8
    img = img.astype(np.uint8)
    cv2.imwrite('output_segmentation_trt.png', img)

    # # 4.速度对比
    # model = torchvision.models.resnet50(pretrained=True).cuda()
    # model = model.eval()
    # dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32).cuda()
    # t3 = time.time()
    # feat_2 = model(dummy_input)
    # t4 = time.time()
    # feat_2 = feat_2.cpu().data.numpy()
    # mse = np.mean((feat - feat_2) ** 2)

    print("TensorRT engine time cost: {}".format(time_t/max_try))
    # print("PyTorch model time cost: {}".format(t4 - t3))
    # print('MSE Error = {}'.format(mse))