# coding:utf-8
# author: lu yf
# create date: 2018/2/7

from Model import *
from torch.autograd import Variable


class RHINE(Model):

    def __init__(self, config):
        super(RHINE, self).__init__(config)
        self.ent_embeddings = nn.Embedding(config.total_nodes, config.hidden_size)
        self.rel_embeddings = nn.Embedding(config.total_IRs, config.hidden_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_embeddings.weight.data)

    def translation_dis(self, h, t, r):
        return torch.abs(h + r - t)

    def euclidea_dis(self, e, v):
        return torch.pow(e - v, 2)

    def loss_func(self, p_score, n_score):
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        y = Variable(torch.Tensor([-1])).cuda()
        loss = criterion(p_score, n_score, y)
        return loss

    def forward(self,mode):
        loss = 0
        if mode == 'Trans':
            pos_h, pos_t, pos_r, pos_rel_w = self.get_postive_IRs()
            neg_h, neg_t, neg_r, neg_rel_w = self.get_negtive_IRs()
            neg_rate = len(neg_h) / len(pos_h)
            neg_step = len(pos_h)

            p_h = self.ent_embeddings(pos_h)
            p_t = self.ent_embeddings(pos_t)
            p_r = self.rel_embeddings(pos_r)
            n_h = self.ent_embeddings(neg_h)
            n_t = self.ent_embeddings(neg_t)
            n_r = self.rel_embeddings(neg_r)
            _p_score = self.translation_dis(p_h, p_t, p_r)
            _n_score = self.translation_dis(n_h, n_t, n_r)
            p_score = torch.sum(_p_score, 1)
            n_score = torch.sum(_n_score, 1)
            pos_rel_w = pos_rel_w.float()
            neg_rel_w = neg_rel_w.float()
            # norm = torch.norm(pos_rel_w, p=2).detach()
            # norm_pos_rel_w = pos_rel_w.div(norm.expand_as(pos_rel_w))
            # norm = torch.norm(pos_rel_w, p=2).detach()
            # norm_neg_rel_w = neg_rel_w.div(norm.expand_as(neg_rel_w))
            trans_loss = 0
            for i in xrange(neg_rate):
                # trans_loss += self.loss_func(norm_pos_rel_w*p_score, norm_neg_rel_w[i*neg_step:(i+1)*neg_step]*n_score[i*neg_step:(i+1)*neg_step])
                trans_loss += self.loss_func(pos_rel_w * p_score,
                                             neg_rel_w[i * neg_step:(i + 1) * neg_step] * n_score[i * neg_step:(i + 1) * neg_step])
            loss = trans_loss
        elif mode == 'Euc':
            pos_e, pos_v, pos_a, pos_attr_w = self.get_postive_ARs()
            neg_e, neg_v, neg_a, neg_attr_w = self.get_negtive_ARs()
            neg_rate = len(neg_e) / len(pos_e)
            neg_step = len(pos_e)

            p_e = self.ent_embeddings(pos_e)
            p_v = self.ent_embeddings(pos_v)
            n_e = self.ent_embeddings(neg_e)
            n_v = self.ent_embeddings(neg_v)
            _p_score = self.euclidea_dis(p_e, p_v)
            _n_score = self.euclidea_dis(n_e, n_v)
            p_score = torch.sum(_p_score, 1)
            n_score = torch.sum(_n_score, 1)
            pos_attr_w = pos_attr_w.float()
            neg_attr_w = neg_attr_w.float()
            # norm = torch.norm(pos_attr_w,p=2).detach()
            # norm_pos_attr_w = pos_attr_w.div(norm.expand_as(pos_attr_w))
            # norm = torch.norm(pos_attr_w, p=2).detach()
            # norm_neg_attr_w = neg_attr_w.div(norm.expand_as(neg_attr_w))
            cl_loss = 0
            for i in xrange(neg_rate):
                # cl_loss = self.cl_loss_func(norm_pos_attr_w*p_score, norm_neg_attr_w*n_score)
                cl_loss += self.loss_func(pos_attr_w * p_score,
                                             neg_attr_w[i * neg_step:(i + 1) * neg_step] * n_score[i * neg_step:(i + 1) * neg_step])
            loss = cl_loss
        return loss
