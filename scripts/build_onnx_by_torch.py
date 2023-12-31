import torch


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()

    def forward(self, x):
        # x = torch.abs(x)
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))
        # shapes = int(x.shape[0]), int(x.shape[1]), int(x.shape[3]), int(x.shape[2])
        # return x.view(shapes)


net = JustReshape()
model_name = 'just_reshape.onnx'
dummy_input = torch.randn(2, 3, 4, 5)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])

