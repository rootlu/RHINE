# coding: utf-8
# author: lu yf
# create date: 2017/12/29

from __future__ import division

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score
import json
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class Evaluation:
    def __init__(self):
        self.entity_name_emb_dict = {}
        np.random.seed(1)

    def load_emb(self, emb_name):
        """
        load embeddings
        :param emb_name:
        :return:
        """
        with open(emb_name, 'r') as emb_file:
            emb_dict = json.load(emb_file)
        return emb_dict

    def evaluation(self,emb_dict):
        entity_emb = emb_dict['ent_embeddings.weight']
        with open('../data/dblp/node2id.txt','r') as e2i_file:
            lines = e2i_file.readlines()

        paper_id_name_dict = {}
        for i in xrange(1,len(lines)):
            tokens = lines[i].strip().split('\t')
            if lines[i][0] == 'p':
                paper_id_name_dict[tokens[1]] = tokens[0]

        for p_id,p_name in paper_id_name_dict.items():
            p_emb = map(lambda x: float(x),entity_emb[int(p_id)])
            self.entity_name_emb_dict[p_name] = p_emb

        x_paper = []
        y_paper = []
        with open('../data/dblp/paper_label.txt', 'r') as paper_name_label_file:
            paper_name_label_lines = paper_name_label_file.readlines()
        for line in paper_name_label_lines:
            tokens = line.strip().split('\t')
            x_paper.append(self.entity_name_emb_dict['p' + tokens[0]])
            y_paper.append(int(tokens[1]))
        self.kmeans_nmi(x_paper, y_paper, k=4)
        self.classification(x_paper, y_paper)

    def kmeans_nmi(self, x, y, k):
        km = KMeans(n_clusters=k)
        km.fit(x,y)
        y_pre = km.predict(x)

        nmi = normalized_mutual_info_score(y, y_pre)
        print('NMI: {}'.format(nmi))

    def classification(self,x,y):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2,random_state=9)

        lr = LogisticRegression()
        lr.fit(x_train,y_train)

        y_valid_pred = lr.predict(x_valid)
        micro_f1 = f1_score(y_valid, y_valid_pred,average='micro')
        macro_f1 = f1_score(y_valid, y_valid_pred,average='macro')
        print ('Macro-F1: {}'.format(macro_f1))
        print ('Micro-F1: {}'.format(micro_f1))


if __name__ == '__main__':
    exp = Evaluation()
    emb1 = exp.load_emb('../res/dblp/embedding.ap_pt_apt+pc_apc.json')
    exp.evaluation(emb1)
