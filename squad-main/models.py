"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BasicLSTM(nn.Module):
    """
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BasicLSTM, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
        self.enc_single = layers.RNNSingle(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        # attention layer removed
        # may need to change input_size of RNNEncoder
        # what is the output of the attention layer?
        """
        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)
        
        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)
        """
        # TODO: add new output layer
        # print('hidden_size', hidden_size)
        self.out = layers.LinearOutput(hidden_size=hidden_size,
                                       drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        # print(cw_idxs.shape, cw_idxs)
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs # location of the nonzero elements
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1) # lengths of each context in the batch
        max_c_len = cw_idxs.shape[1]
        
        # concat_words = torch.cat([c_emb, q_emb], dim=1)
        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        # concat_emb = torch.cat([c_emb, q_emb], dim=1) # concatenate embeddings, pass all thru LSTM

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        # q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        q_enc = self.enc_single(q_emb, q_len)    
        q_enc = q_enc[0].transpose(0, 1) # get hidden state, swap axes. # (batch_size, 2, 2 * hidden_size)
        q_enc = q_enc.sum(dim=1, keepdim=True) # combine two states
        # print('rnn', q_enc.shape)

        q_enc = q_enc.repeat((1,max_c_len,1))
        # print('c_enc', c_enc.shape)
        # print('q_enc', q_enc.shape)

        concat_enc = torch.cat([c_enc, q_enc], dim=2)    # (batch_size, c_len, 3 * hidden_size)
        # concat_enc = c_enc
        # print('shape', concat_enc.shape)
        
        # TODO: get output
        out_1, out_2 = self.out(concat_enc, c_mask) # 2 tensors, each (batch_size, c_len)
        
        return (out_1, out_2)

class LSTMOld(nn.Module):
    """
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(TestModel, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        # attention layer removed
        # may need to change input_size of RNNEncoder
        # what is the output of the attention layer?
        """
        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)
        
        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)
        """
        # TODO: add new output layer
        self.out = layers.LinearOutput(hidden_size=hidden_size,
                                       drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        
        # c_len is a vector
        # then what does c_len in the dimension mean??
        # is the tensor jagged?
        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        # c_enc = c_enc.flatten(start_dim=1) # flatten
        # q_enc = q_enc.flatten(start_dim=1) # flatten

        c_len_scalar = c_enc.size()[1]
        q_len_scalar = q_enc.size()[1]
        
        """
        Attention layer removed
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        """
        """
        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        """
        # TODO: concatenate the tensors
        # This doesn't look right! number of features (size of hidden layer) should change
        # since we use question and context to predict
        # let's use it for now
        concat_enc = torch.cat([c_enc, q_enc], dim=1) # (batch_size, c_len + q_len, 2 * hidden_size)
        concat_mask = torch.cat([c_mask, q_mask], dim=1) # (batch_size, c_len + q_len, 2 * hidden_size)
        # batch_size = c_enc.size(0)
        
        # TODO: get output
        out_1, out_2 = self.out(concat_enc, concat_mask) # 2 tensors, each (batch_size, c_len + q_len)
        
        # want output shape to be the same as BiDAF which is (batch_size, c_len)
        # we slice the matrix by column to get new matrix with size (batch_size, c_len)
        # this probably won't work since this is super simple, but kind of makes sense 
        # since logits and log probabilities for the questions concatenated after
        # the contexts (batch_size, c_len:) are garbage 
        out_1_trunc = torch.narrow(out_1, dim=1, start=0, length=c_len_scalar)
        out_2_trunc = torch.narrow(out_2, dim=1, start=0, length=c_len_scalar)
        return (out_1, out_2)
