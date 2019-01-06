# coding:utf-8
# author: lu yf
# create date: 2018/2/6
# Based on openKePyTorch: https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch

import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

    def get_postive_IRs(self):
        """
        sample positive IRs triples
        :return:
        """
        self.postive_h_IRs = Variable(torch.from_numpy(self.config.batch_h_IRs[0:self.config.batch_size_IRs])).cuda()
        self.postive_t_IRs = Variable(torch.from_numpy(self.config.batch_t_IRs[0:self.config.batch_size_IRs])).cuda()
        self.postive_r_IRs = Variable(torch.from_numpy(self.config.batch_r_IRs[0:self.config.batch_size_IRs])).cuda()
        self.postive_w_IRs = Variable(torch.from_numpy(self.config.batch_w_IRs[0:self.config.batch_size_IRs])).cuda()
        return self.postive_h_IRs, self.postive_t_IRs, self.postive_r_IRs, self.postive_w_IRs

    def get_negtive_IRs(self):
        """
        sample negative IRs triples
        :return:
        """
        self.negtive_h_IRs = Variable(
            torch.from_numpy(self.config.batch_h_IRs[self.config.batch_size_IRs:self.config.batch_seq_size_IRs])).cuda()
        self.negtive_t_IRs = Variable(
            torch.from_numpy(self.config.batch_t_IRs[self.config.batch_size_IRs:self.config.batch_seq_size_IRs])).cuda()
        self.negtive_r_IRs = Variable(
            torch.from_numpy(self.config.batch_r_IRs[self.config.batch_size_IRs:self.config.batch_seq_size_IRs])).cuda()
        self.negtive_w_IRs = Variable(
            torch.from_numpy(self.config.batch_w_IRs[self.config.batch_size_IRs:self.config.batch_seq_size_IRs])).cuda()

        return self.negtive_h_IRs, self.negtive_t_IRs, self.negtive_r_IRs, self.negtive_w_IRs

    def get_postive_ARs(self):
        self.postive_h_ARs = Variable(torch.from_numpy(self.config.batch_h_ARs[0:self.config.batch_size_ARs])).cuda()
        self.postive_t_ARs = Variable(torch.from_numpy(self.config.batch_t_ARs[0:self.config.batch_size_ARs])).cuda()
        self.postive_r_ARs = Variable(torch.from_numpy(self.config.batch_r_ARs[0:self.config.batch_size_ARs])).cuda()
        self.postive_w_ARs = Variable(torch.from_numpy(self.config.batch_w_ARs[0:self.config.batch_size_ARs])).cuda()
        return self.postive_h_ARs, self.postive_t_ARs, self.postive_r_ARs, self.postive_w_ARs

    def get_negtive_ARs(self):
        self.negtive_h_ARs = Variable(
            torch.from_numpy(self.config.batch_h_ARs[self.config.batch_size_ARs:self.config.batch_seq_size_ARs])).cuda()
        self.negtive_t_ARs = Variable(
            torch.from_numpy(self.config.batch_t_ARs[self.config.batch_size_ARs:self.config.batch_seq_size_ARs])).cuda()
        self.negtive_r_ARs = Variable(
            torch.from_numpy(self.config.batch_r_ARs[self.config.batch_size_ARs:self.config.batch_seq_size_ARs])).cuda()
        self.negtive_w_ARs = Variable(
            torch.from_numpy(self.config.batch_w_ARs[self.config.batch_size_ARs:self.config.batch_seq_size_ARs])).cuda()

        return self.negtive_h_ARs, self.negtive_t_ARs, self.negtive_r_ARs, self.negtive_w_ARs

    def predict(self):
        pass

    def forward(self):
        pass

    def loss_func(self):
        pass
