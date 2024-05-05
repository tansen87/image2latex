import sys

import torch
from torch.autograd import Variable
import numpy as np

from modules.Attention_RNN import AttnDecoderRNN
from modules.Densenet_torchvision import densenet121


sys.path.append(r"../")

gpu = [0]
dictionary = ['./modules/dictionary.txt']
hidden_size = 256
batch_size_t = 1
max_len = 100


def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    return lexicon


word_dict = load_dict(dictionary[0])
word_dicts = [None] * len(word_dict)
for keys, values in word_dict.items():
    word_dicts[values] = keys


def image_solve(x_t):
    h_mask_t = []
    w_mask_t = []
    encoder = densenet121()

    attn_decoder = AttnDecoderRNN(hidden_size, 112, dropout_p=0.5)

    # encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
    # attn_decoder = torch.nn.DataParallel(attn_decoder, device_ids=gpu)
    device = torch.device("cuda:0")
    encoder = encoder.to(device).cuda()
    attn_decoder = attn_decoder.to(device).cuda()
    # encoder = encoder.cuda()
    # attn_decoder = attn_decoder.cuda()

    encoder.load_state_dict(torch.load('./model/encoder_w.pkl'))
    attn_decoder.load_state_dict(torch.load('./model/attn_decoder_w.pkl'))

    encoder.eval()
    attn_decoder.eval()

    x_t = Variable(x_t.cuda())
    x_mask = torch.ones(x_t.size()[0], x_t.size()[1], x_t.size()[2], x_t.size()[3]).cuda()
    x_t = torch.cat((x_t, x_mask), dim=1)
    x_real_high = x_t.size()[2]
    x_real_width = x_t.size()[3]
    h_mask_t.append(int(x_real_high))
    w_mask_t.append(int(x_real_width))
    x_real = x_t[0][0].view(x_real_high, x_real_width)
    output_highfeature_t = encoder(x_t)

    x_mean_t = torch.mean(output_highfeature_t)
    x_mean_t = float(x_mean_t)
    output_area_t1 = output_highfeature_t.size()
    output_area_t = output_area_t1[3]
    dense_input = output_area_t1[2]

    decoder_input_t = torch.LongTensor([111] * batch_size_t)
    decoder_input_t = decoder_input_t.cuda()

    decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda()

    decoder_hidden_t = decoder_hidden_t * x_mean_t
    decoder_hidden_t = torch.tanh(decoder_hidden_t)

    prediction = torch.zeros(batch_size_t, max_len)
    decoder_attention_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t).cuda()
    attention_sum_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t).cuda()
    decoder_attention_t_cat = []

    for i in range(max_len):
        decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder(
            decoder_input_t,
            decoder_hidden_t,
            output_highfeature_t,
            output_area_t,
            attention_sum_t,
            decoder_attention_t,
            dense_input,
            batch_size_t,
            h_mask_t,
            w_mask_t,
            gpu)

        decoder_attention_t_cat.append(decoder_attention_t[0].data.cpu().numpy())
        topv, topi = torch.max(decoder_output, 2)
        if torch.sum(topi) == 0:
            break
        decoder_input_t = topi
        decoder_input_t = decoder_input_t.view(batch_size_t)

        # prediction
        prediction[:, i] = decoder_input_t

    k = np.array(decoder_attention_t_cat)
    x_real = np.array(x_real.cpu().data)

    prediction = prediction[0]

    prediction_real = []
    for ir in range(len(prediction)):
        if int(prediction[ir]) == 0:
            break
        prediction_real.append(word_dicts[int(prediction[ir])])
    prediction_real.append('<eol>')

    prediction_real_show = np.array(prediction_real)

    return k, prediction_real_show
