# import torch
import onnxruntime as ort
import numpy as np

onnx_path = r"D:\code\onnx_study\conv.onnx"
sess = ort.InferenceSession(
    path_or_bytes=onnx_path, 
    providers=["CUDAExecutionProvider"]
    # providers=["CPUExecutionProvider"]
)

x = np.full((2, 3, 224, 224), fill_value=2, dtype=np.float32)


y = sess.run(
    output_names=["output", ], 
    input_feed={
        "image": x, 
    }, 
)[0]
print(y.shape)