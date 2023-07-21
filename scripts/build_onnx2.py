import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

x = helper.make_tensor_value_info("image", TensorProto.FLOAT, (2, 3, 224, 224), )


w = helper.make_tensor_value_info("Conv_0.weight", TensorProto.FLOAT, (3, 3, 3, 3))

b = helper.make_tensor_value_info("Conv_0.bias", TensorProto.FLOAT, (3, ))



w_value = helper.make_tensor("Conv_0.weight", TensorProto.FLOAT, (3, 3, 3, 3), vals=np.zeros((3, 3, 3, 3)))

b_value = helper.make_tensor("Conv_0.bias", TensorProto.FLOAT, (3, ), vals=np.ones((3, )))

y = helper.make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 112, 112), )



conv = helper.make_node(
    "Conv", 
    inputs=["image", "Conv_0.weight", "Conv_0.bias"], 
    outputs=["output", ], 
    name="Conv_0", 
    # param
    dilations=(1, 1), 
    group=1, 
    kernel_shape=(3, 3), 
    pads=(1, 1, 1, 1), 
    strides=(2, 2),
    
)


graph_def = helper.make_graph(
    [conv, ],
    'conv_model',
    [x, w, b],
    [y, ],
    initializer=[w_value, b_value]
)

model_def = helper.make_model(graph_def, producer_name='onnx-example')
print(model_def)

onnx.checker.check_model(model_def)

# save
f = model_def.SerializeToString()
file = open("conv.onnx", "wb")
file.write(f)
