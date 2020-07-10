import highway
import torch


def test_shapes():
    embedding_size = 3
    batch_size = 4
    conv_out = torch.tensor([[1, 2, 3], [0.4, 0.8, 0.2], [1, 0.5, 0], [1, 2.0, 4.0]])  # batch_size x embedding size
    assert conv_out.shape == (batch_size, embedding_size)

    highwayNet = highway.Highway(3, *[torch.nn.init.uniform] * 4)
    out = highwayNet.forward(conv_out)
    assert out.shape == (batch_size, embedding_size)


def test_values():
    highway_net = highway.Highway(3, *[torch.nn.init.zeros_] * 4)
    conv_out = torch.tensor([[1, 2, 3], [0.4, 0.8, 0.2], [1, 0.5, 0], [1, 2.0, 4.0]])  # batch_size x embedding size
    output = highway_net.forward(conv_out)
    # if we set the weight and bias matrices to zero, only the right term will be nonzero
    # which is 1 - sigmoid(0) * x_conv_out == 0.5 * x_conv_out
    assert torch.all(torch.eq(output, conv_out * 0.5))
