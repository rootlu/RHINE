# coding:utf-8
# author: lu yf
# create date: 2017/12/29

from __future__ import division
import os
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import csv


class DataHelper:
    def __init__(self,input_data_dir,output_data_dir):
        self.input_data_fold = input_data_dir
        self.output_data_fold = output_data_dir
        self.node2id_dict = {}
        self.relation2id_dict = {}

    def load_data(self):
        """
        transform num to id, and build adj_matrix
        :return:
        """
        print ('loading data form {}...'.format(self.input_data_fold))
        self.author_list = set([])
        self.paper_list = set([])
        self.conf_list = set([])
        self.term_list = set([])
        with open(os.path.join(self.input_data_fold, 'paper_author.txt')) as pa_file:
            pa_line = pa_file.readlines()
        for line in pa_line:
            token = line.strip('\n').split('\t')
            self.paper_list.add(token[0])
            self.author_list.add(token[1])
        with open(os.path.join(self.input_data_fold, 'paper_conf.txt')) as pc_file:
            pc_line = pc_file.readlines()
        for line in pc_line:
            token = line.strip('\n').split('\t')
            self.paper_list.add(token[0])
            self.conf_list.add(token[1])
        with open(os.path.join(self.input_data_fold, 'paper_term.txt')) as pt_file:
            pt_line = pt_file.readlines()
        for line in pt_line:
            token = line.strip('\n').split('\t')
            self.paper_list.add(token[0])
            self.term_list.add(token[1])
        self.paper_list = list(self.paper_list)
        self.author_list = list(self.author_list)
        self.conf_list = list(self.conf_list)
        self.term_list = list(self.term_list)

        print ('#authors: {}'.format(len(self.author_list)))
        print ('#papers: {}'.format(len(self.paper_list)))
        print ('#confs: {}'.format(len(self.conf_list)))
        print ('#terms: {}'.format(len(self.term_list)))

        print ('building adj_matrix...')
        self.pa_adj_matrix = np.zeros([len(self.paper_list), len(self.author_list)], dtype=float)
        for line in pa_line:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pa_adj_matrix[row][col] = 1

        self.pc_adj_matrix = np.zeros([len(self.paper_list), len(self.conf_list)], dtype=float)
        for line in pc_line:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pc_adj_matrix[row][col] = 1

        self.pt_adj_matrix = np.zeros([len(self.paper_list), len(self.term_list)], dtype=float)
        for line in pt_line:
            token = line.strip('\n').split('\t')
            row = int(token[0])
            col = int(token[1])
            self.pt_adj_matrix[row][col] = 1

        self.ap_adj_matrix = np.transpose(self.pa_adj_matrix)
        self.apc_adj_matrix = np.matmul(self.ap_adj_matrix, self.pc_adj_matrix)
        self.apt_adj_matrix = np.matmul(self.ap_adj_matrix, self.pt_adj_matrix)

    def write_node_and_relation_2_files(self,relation_list):
        print ('writing node2id to file...')
        with open(os.path.join(self.output_data_fold,'node2id.txt'), 'w') as n2i_file:
            num_node = len(self.author_list) + len(self.paper_list) + len(self.conf_list) + len(self.term_list)
            n2i_file.write(str(num_node))
            n2i_file.write('\n')
            idx = 0
            for i in self.author_list:
                node = 'a' + i
                n2i_file.write(node)
                n2i_file.write('\t' + str(idx))
                n2i_file.write('\n')
                self.node2id_dict[node] = idx
                idx += 1

            for i in self.paper_list:
                node = 'p' + i
                n2i_file.write(node)
                n2i_file.write('\t' + str(idx))
                n2i_file.write('\n')
                self.node2id_dict[node] = idx
                idx += 1
            for i in self.conf_list:
                node = 'c' + i
                n2i_file.write(node)
                n2i_file.write('\t' + str(idx))
                n2i_file.write('\n')
                self.node2id_dict[node] = idx
                idx += 1

            for i in self.term_list:
                node = 't' + i
                n2i_file.write(node)
                n2i_file.write('\t' + str(idx))
                n2i_file.write('\n')
                self.node2id_dict[node] = idx
                idx += 1
        print ('writing relation2id to file...')
        with open(os.path.join(self.output_data_fold,'relation2id.txt'), 'w') as r2i_file:
            num_relation = len(relation_list)
            r2i_file.write(str(num_relation))
            r2i_file.write('\n')
            for i, r in enumerate(relation_list):
                r2i_file.write(str(r) + '\t')
                r2i_file.write(str(i) + '\n')
                self.relation2id_dict[r] = i

    def generate_triples(self,adj_matrix,relation_type):
        print ('gernerating triples for relation {}...'.format(relation_type))
        ridx, cidx = np.nonzero(adj_matrix)
        ridx = list(ridx)
        cidx = list(cidx)
        num_triples = len(ridx)
        train_data = open(os.path.join(self.output_data_fold,'train2id_'+relation_type+'.txt'),'w')

        for i in xrange(num_triples):
            n1 = self.node2id_dict[relation_type[0]+str(ridx[i])]
            n2 = self.node2id_dict[relation_type[-1]+str(cidx[i])]
            r = self.relation2id_dict[relation_type]
            w = int(adj_matrix[ridx[i]][cidx[i]])
            train_data.write(str(n1) + '\t' + str(n2) + '\t' + str(r) + '\t' + str(w) + '\n')

        train_data.close()

    def merge_triples(self,relations_list,relation_category):
        print ('merging triples for {}...'.format(relation_category))
        merged_data = open(os.path.join(self.output_data_fold,'train2id_'+relation_category+'.txt'),'w+')
        line_num = 0
        content = ''
        for r in relations_list:
            for line in open(os.path.join(self.output_data_fold,'train2id_'+r+'.txt')):
                content += line
                line_num += 1
        merged_data.writelines(str(line_num)+'\n'+content)


if __name__ == "__main__":
    data_helper = DataHelper(input_data_dir='../../data/dblp/originData/',output_data_dir='../../data/dblp/output/')
    data_helper.load_data()
    relation_list = ['ap','pc','pt','apc','apt']
    data_helper.write_node_and_relation_2_files(relation_list)

    data_helper.generate_triples(data_helper.ap_adj_matrix,'ap')
    data_helper.generate_triples(data_helper.pc_adj_matrix,'pc')
    data_helper.generate_triples(data_helper.pt_adj_matrix,'pt')
    data_helper.generate_triples(data_helper.apc_adj_matrix,'apc')
    data_helper.generate_triples(data_helper.apt_adj_matrix,'apt')

    data_helper.merge_triples(['ap','pt','apt'],'IRs')
    data_helper.merge_triples(['pc','apc'],'ARs')