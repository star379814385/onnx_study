from onnxsim import simplify
import onnx
from pathlib import Path

model_path = r"D:\code\onnx_study\just_reshape.onnx"
# model = onnx.load_model(model_path)
# input_shapes = (2, 3, 224, 112)
input_shapes = {
    "input": [2, 3, 4, 5],   
}

sim_model, flag = simplify(model=model_path, overwrite_input_shapes=input_shapes, test_input_shapes=False)
f = sim_model.SerializeToString()
if flag:
    print("simplify success.")
    f = sim_model.SerializeToString()
    save_path = str(Path(model_path).parent / f"sim_{Path(model_path).name}")
    file = open(save_path, "wb")
    file.write(f)
