import torch
import onnxruntime as ort
import numpy as np
import cv2

# onnx_path = r"D:\code\onnx_study\conv.onnx"
onnx_path = r"D:/code/inference_web/server/models/yolov8n.onnx"
sess = ort.InferenceSession(
    path_or_bytes=onnx_path, 
    providers=["CUDAExecutionProvider"]
    # providers=["CPUExecutionProvider"]
)

# x = np.full((1, 3, 640, 640), fill_value=2, dtype=np.float32)
x = cv2.imread(r"D:\code\mmdetection\demo\demo.jpg")
x = cv2.resize(x, (640, 640))
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = (x.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

y = sess.run(
    output_names=["output0", ], 
    input_feed={
        "images": x, 
    }, 
)[0]
print(y.shape)