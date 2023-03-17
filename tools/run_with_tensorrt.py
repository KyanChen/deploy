import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# 1. 确定batch size大小，与导出的trt模型保持一致
BATCH_SIZE = 1

# 2. 选择是否采用FP16精度，与导出的trt模型保持一致
USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32

# 3. 创建Runtime，加载TRT引擎
f = open("results/deploy/.trt", "rb")  # 读取trt模型
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))  # 创建一个Runtime(传入记录器Logger)
engine = runtime.deserialize_cuda_engine(f.read())  # 从文件中加载trt引擎
context = engine.create_execution_context()  # 创建context

# 4. 分配input和output内存
input_batch = np.random.randn(BATCH_SIZE, 224, 224, 3).astype(target_dtype)
output = np.empty([BATCH_SIZE, 1000], dtype=target_dtype)

d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()


# 5. 创建predict函数
def predict(batch):  # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)  # 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output


# 6. 调用predict函数进行推理，并记录推理时间
def preprocess_input(input):  # input_batch无法直接传给模型，还需要做一定的预处理
    # 此处可以添加一些其它的预处理操作（如标准化、归一化等）
    result = torch.from_numpy(input).transpose(0, 2).transpose(1, 2)  # 利用torch中的transpose,使(224,224,3)——>(3,224,224)
    return np.array(result, dtype=target_dtype)


preprocessed_inputs = np.array(
    [preprocess_input(input) for input in input_batch])  # (BATCH_SIZE,224,224,3)——>(BATCH_SIZE,3,224,224)

print("Warming up...")
pred = predict(preprocessed_inputs)
print("Done warming up!")

t0 = time.time()
pred = predict(preprocessed_inputs)
t = time.time() - t0
print("Prediction cost {:.4f}s".format(t))
