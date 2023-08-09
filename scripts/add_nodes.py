

import onnx
from onnx.helper import make_node, make_tensor_value_info, get_attribute_value
from onnx import AttributeProto, TensorProto, GraphProto



onnx_path = "just_reshape.onnx"
model = onnx.load(onnx_path)
nodes = model.graph.node
cur_len = len(nodes)
tensor_num = 0
for i in range(len(nodes)):
    node = nodes[i]
    print(i, node.name, node.input, node.output, node.op_type)
    if node.op_type == "Reshape":
        outputs = node.output
        if len(outputs) == 0:
            continue


        x = make_tensor_value_info(f"tensor_{tensor_num}", TensorProto.FLOAT, node.input[1])
        tensor_num += 1
        # x2 = make_tensor_value_info(f"tensor_{tensor_num}", TensorProto.FLOAT, node.shape)
        # tensor_num += 1
        add_node = make_node("Abs", inputs=[x.name, ], outputs=outputs, name=node.name.replace("Reshape", "Abs"))
        while len(node.output) > 1:
            del node.output[0]
        
        node.output[0] = x.name
        nodes.insert(cur_len, add_node)
        cur_len += 1


for i, node in enumerate(model.graph.node):
    print(i, node.name, node.input, node.output)

f = model.SerializeToString()
file = open("abs_reshape.onnx", "wb")
file.write(f)
