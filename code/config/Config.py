# coding:utf-8
# author: lu yf
# create date: 2018/2/6
# Based on openKePyTorch: https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch

import torch
import numpy as np
import torch.optim as optim
import ctypes
import json
import sys
from tqdm import tqdm

sys.path.append("..")
import evaluation

exp = evaluation.Evaluation()


class Config(object):

    def __init__(self):
        self.lib_IRs = ctypes.cdll.LoadLibrary("./release/Sample_IRs.so")
        self.lib_IRs.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        self.lib_IRs.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib_IRs.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.lib_ARs = ctypes.cdll.LoadLibrary("./release/Sample_ARs.so")
        self.lib_ARs.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        self.lib_ARs.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib_ARs.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.evaluation_flag = False
        self.in_path = "./"
        self.out_path = "./"
        self.hidden_size = 100
        self.ent_size = self.hidden_size
        self.rel_size = self.hidden_size
        self.train_times = 0
        self.margin = 1.0
        self.IRs_nbatches = 100
        self.ARs_nbatches = 100
        self.negative_ent = 1
        self.negative_rel = 0
        self.workThreads = 1
        self.alpha = 0.001
        self.log_on = 1
        self.lr_decay = 0.000
        self.weight_decay = 0.000
        self.exportName = None
        self.importName = None
        self.export_steps = 0
        self.opt_method = "SGD"
        self.optimizer = None

    def init(self):
        """
        init. parameters
        :return:
        """
        self.trainModel = None
        if self.in_path != None:
            # sample IRs
            self.lib_IRs.setInPath(ctypes.create_string_buffer(self.in_path, len(self.in_path) * 2))
            self.lib_IRs.setWorkThreads(self.workThreads)
            self.lib_IRs.randReset()
            self.lib_IRs.importTrainFiles()
            self.total_IRs = self.lib_IRs.getRelationTotal()
            self.total_nodes = self.lib_IRs.getEntityTotal()
            self.train_total_IRs_triple = self.lib_IRs.getTrainTotal()
            self.batch_size_IRs = self.lib_IRs.getTrainTotal() / self.IRs_nbatches
            print ('# IRs triples: {}'.format(self.train_total_IRs_triple))
            print('IRs triple batch size: {}'.format(self.batch_size_IRs))
            self.batch_seq_size_IRs = self.batch_size_IRs * (1 + self.negative_ent+self.negative_rel)
            self.batch_h_IRs = np.zeros(self.batch_size_IRs * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_t_IRs = np.zeros(self.batch_size_IRs * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_r_IRs = np.zeros(self.batch_size_IRs * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_w_IRs = np.ones(self.batch_size_IRs * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)

            self.batch_h_addr_IRs = self.batch_h_IRs.__array_interface__['data'][0]
            self.batch_t_addr_IRs = self.batch_t_IRs.__array_interface__['data'][0]
            self.batch_r_addr_IRs = self.batch_r_IRs.__array_interface__['data'][0]
            self.batch_w_addr_IRs = self.batch_w_IRs.__array_interface__['data'][0]

            # sample ARs
            self.lib_ARs.setInPath(ctypes.create_string_buffer(self.in_path, len(self.in_path) * 2))
            self.lib_ARs.setWorkThreads(self.workThreads)
            self.lib_ARs.randReset()
            self.lib_ARs.importTrainFiles()
            self.train_total_ARs_triple = self.lib_ARs.getTrainTotal()
            self.batch_size_ARs = self.lib_ARs.getTrainTotal() / self.ARs_nbatches
            print ('# ARs triples: {}'.format(self.train_total_ARs_triple))
            print('ARs triple batch size: {}'.format(self.batch_size_ARs))
            self.batch_seq_size_ARs = self.batch_size_ARs * (1 + self.negative_ent+self.negative_rel)
            self.batch_h_ARs = np.zeros(self.batch_size_ARs * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_t_ARs = np.zeros(self.batch_size_ARs * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_r_ARs = np.zeros(self.batch_size_ARs * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)
            self.batch_w_ARs = np.ones(self.batch_size_ARs * (1 + self.negative_ent + self.negative_rel), dtype=np.int64)

            self.batch_h_addr_ARs = self.batch_h_ARs.__array_interface__['data'][0]
            self.batch_t_addr_ARs = self.batch_t_ARs.__array_interface__['data'][0]
            self.batch_r_addr_ARs = self.batch_r_ARs.__array_interface__['data'][0]
            self.batch_w_addr_ARs = self.batch_w_ARs.__array_interface__['data'][0]

    def set_opt_method(self, method):
        self.opt_method = method

    def set_log_on(self, flag):
        self.log_on = flag

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_in_path(self, path):
        self.in_path = path

    def set_out_files(self, path):
        self.out_path = path

    def set_dimension(self, dim):
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_train_times(self, times):
        self.train_times = times

    def set_IRs_nbatches(self, nbatches):
        self.IRs_nbatches = nbatches

    def set_ARs_nbatches(self, nbatches):
        self.ARs_nbatches = nbatches

    def set_margin(self, margin):
        self.margin = margin

    def set_work_threads(self, threads):
        self.workThreads = threads

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_import_files(self, path):
        self.importName = path

    def set_export_files(self, path):
        self.exportName = path

    def set_export_steps(self, steps):
        self.export_steps = steps

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_evaluation(self, evaluation_flag):
        self.evaluation_flag = evaluation_flag

    def sampling_IRs(self):
        self.lib_IRs.sampling(self.batch_h_addr_IRs, self.batch_t_addr_IRs, self.batch_r_addr_IRs, self.batch_w_addr_IRs,
                              self.batch_size_IRs,self.negative_ent, self.negative_rel)

    def sampling_ARs(self):
        self.lib_ARs.sampling(self.batch_h_addr_ARs, self.batch_t_addr_ARs, self.batch_r_addr_ARs, self.batch_w_addr_ARs,
                              self.batch_size_ARs,self.negative_ent, self.negative_rel)

    def save_pytorch(self):
        torch.save(self.trainModel.state_dict(), self.exportName)

    def restore_pytorch(self):
        self.trainModel.load_state_dict(torch.load(self.importName))

    def get_parameter_lists(self):
        return self.trainModel.cpu().state_dict()

    def get_parameters(self, mode="numpy"):
        res = {}
        lists = self.get_parameter_lists()
        for var_name in lists:
            if mode == "numpy":
                res[var_name] = lists[var_name].numpy()
            if mode == "list":
                res[var_name] = lists[var_name].numpy().tolist()
            else:
                res[var_name] = lists[var_name]
        return res

    def save_parameters(self, path=None):
        if path is None:
            path = self.out_path
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def set_parameters_by_name(self, var_name, tensor):
        self.trainModel.state_dict().get(var_name).copy_(torch.from_numpy(np.array(tensor)))

    def set_parameters(self, lists):
        for i in lists:
            self.set_parameters_by_name(i, lists[i])

    def set_model(self, model):
        self.model = model
        self.trainModel = self.model(config=self)
        self.trainModel.cuda()
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.alpha,
                                           lr_decay=self.lr_decay, weight_decay=self.weight_decay)
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr=self.alpha)
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr=self.alpha)
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr=self.alpha)

    def run(self):
        if self.importName is not None:
            self.restore_pytorch()
        for epoch in range(self.train_times):
            res = 0.0
            for batch in range(self.IRs_nbatches):
                self.sampling_IRs()
                self.optimizer.zero_grad()
                loss = self.trainModel('Trans')
                res += loss.data[0]
                loss.backward()
                self.optimizer.step()
            for batch in range(self.ARs_nbatches):
                self.sampling_ARs()
                self.optimizer.zero_grad()
                loss = self.trainModel('Euc')
                res += loss.data[0]
                loss.backward()
                self.optimizer.step()

            if self.exportName is not None and (self.export_steps != 0 and epoch % self.export_steps == 0):
                self.save_pytorch()
            if self.log_on == 1:
                print ('Epoch: {}, loss: {}'.format(epoch, res))
            if self.evaluation_flag and epoch != 0 and epoch % 100 == 0:
                emb_json = self.get_parameters("list")
                exp.evaluation(emb_json)
                self.trainModel.cuda()

        if self.out_path is not None:
            self.save_parameters(self.out_path)
