import cnn
import torch


def test_shapes():
    # A char is embedded with 3 dim. vector, and a word is max. five characters long
    e_char, e_word, m_word = 3, 4, 5
    batch_size = 2
    test_input = torch.tensor([
        [
            [1.0, 2.0, 0.5, 0.6, 0.7],  # the columns are the embeddings?
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 0.5, 0.6, 0.7]
        ],
        [
            [0.5, 0.4, 0.7, 0.3, 0.99],
            [1.1, 2.1, 3.1, 0.5, 0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ]
    ])
    assert test_input.shape == (batch_size, e_char, m_word)
    convnet = cnn.CNN(kernel_size=5, filters=e_word, e_char=e_char, m_word=m_word)
    result = convnet.forward(test_input)
    assert result.shape == (batch_size, e_word, 1)
