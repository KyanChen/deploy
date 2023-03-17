import time
import pycuda.autoinit  #负责数据初始化，内存管理，销毁等
import pycuda.driver as cuda  #GPU CPU之间的数据传输
import numpy as np
import cv2
import tensorrt as trt


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


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append ti the appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input to the GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs
    return [out.host for out in outputs]


def main():
    trt_file = "results/deploy/end2end.trt.engine"
    img_file = 'test_cloud_img.jpg'
    img = cv2.imread(img_file)
    input = form_input(img).ravel()

    with open(trt_file, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    inputs[0].host = input
    outs_shape = [(1, 512, 512)]

    trt_outs = do_inference(context, bindings, inputs, outputs, stream)

    trt_out = [output.reshape(shape) for output, shape in zip(trt_outs, outs_shape)]

    import pdb
    pdb.set_trace()

    # time_t = 0
    # for i in range(20):
    #     t_s = time.time()
    #     inputs = {onet_session.get_inputs()[0].name: form_input(img)}
    #     outs = onet_session.run(None, inputs)
    #
    #     time_t += time.time() - t_s
    # print(time_t / 20)
    # seg = outs[0]
    #
    # palette = get_palette()
    # color_seg = np.zeros((512, 512, 3), dtype=np.uint8)
    # for label, color in enumerate(palette):
    #     color_seg[seg == label, :] = color
    # # convert to BGR
    # color_seg = color_seg[..., ::-1]
    #
    # img = img * 0.2 + color_seg * 0.8
    # img = img.astype(np.uint8)
    # cv2.imwrite('output_segmentation_onnx.png', img)

if __name__ == '__main__':
    main()