"""class implements character-level convolutional 1D layer"""
import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerCNN(LayerBase):
    """LayerCNN implements word and char convolutional 1D layer."""
    def __init__(self, gpu, inp_dim, char_embeddings_dim, cnn_window_size,
                 max_seq_len=150):
        super(LayerCNN, self).__init__(gpu)
        self.char_embeddings_dim = char_embeddings_dim
        self.cnn_window_size = cnn_window_size
        self.output_dim = inp_dim
        self.max_seq_len = max_seq_len
        self.conv1d = nn.Conv1d(in_channels=char_embeddings_dim,
                                out_channels=char_embeddings_dim,
                                kernel_size=cnn_window_size,
                                groups=char_embeddings_dim,
                                padding=1)

    def is_cuda(self):
        return self.conv1d.weight.is_cuda

    def forward(self, char_embeddings_feature): # batch_num x max_seq_len x char_embeddings_dim x word_len
        _, real_len, _ = char_embeddings_feature.shape
        char_embeddings_feature = self.pad_sequence(char_embeddings_feature,
                                                    self.max_seq_len)
        char_embeddings_feature = char_embeddings_feature.permute(0, 2, 1)
        batch_num, max_seq_len, char_embeddings_dim = char_embeddings_feature.shape
        conv_out = self.conv1d(char_embeddings_feature).permute(0, 2, 1)
        return conv_out[:, :real_len] # shape: batch_num x max_seq_len x filter_num*char_embeddings_dim

    def pad_sequence(self, inp, max_seq_len):
        return torch.nn.functional.pad(inp, (0, 0, 0,
                                             max_seq_len - inp.shape[1], 0, 0))
