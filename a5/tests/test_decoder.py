import json

import torch

from char_decoder import CharDecoder
from nmt_model import NMT
from vocab import Vocab

# same test setup as sanity checks
# ----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 4
DROPOUT_RATE = 0.0


class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_pad = self.char2id['∏']
        self.char_unk = self.char2id['Û']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]


def mock_decoder_forward(_current_char, dec_hidden):
    s_t = [[[0] * len(DummyVocab().char2id)] * BATCH_SIZE]  # shape: 1, batch_size, vocab_length
    for row in s_t[0]:
        # set one entry to one, so that softmax will return this value as 1 probability
        # this char id should repeatedly show up in output
        row[4] = 1  # character code for lowercase "a" in DummyVocab
    s_t = torch.tensor(s_t, dtype=torch.float)
    return s_t, dec_hidden


def test_greedy_decode():
    char_vocab = DummyVocab()
    decoder = CharDecoder(
        hidden_size=HIDDEN_SIZE,
        char_embedding_size=EMBED_SIZE,
        target_vocab=char_vocab)
    max_word_length = 21
    decoder.forward = mock_decoder_forward
    initial_states = (
        torch.tensor([[[0] * HIDDEN_SIZE] * BATCH_SIZE]), torch.tensor([[[0] * HIDDEN_SIZE] * BATCH_SIZE]))
    result = decoder.decode_greedy(initialStates=initial_states, device=decoder.char_output_projection.weight.device,
                                   max_length=max_word_length)
    for decoded_word in result:
        assert decoded_word == "a" * (max_word_length - 1)
