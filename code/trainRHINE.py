# coding:utf-8
# author: lu yf
# create date: 2018/2/6

import config
import model
import evaluation
import time
import warnings

warnings.filterwarnings('ignore')


def train_model(data_set, mode):
    """
    train model
    :param data_set:
    :param mode: relation categories
    :return:
    """
    con = config.Config()
    con.set_in_path("../data/" + data_set + "/")

    con.set_work_threads(16)
    con.set_train_times(400)
    con.set_IRs_nbatches(100)
    con.set_ARs_nbatches(100)
    con.set_alpha(0.005)
    con.set_margin(1)
    con.set_dimension(100)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("SGD")
    con.set_evaluation(False)

    con.set_export_files("../res/" + data_set + "/model.vec." + mode + ".tf")
    con.set_out_files("../res/" + data_set + "/embedding.vec." + mode + ".json")
    con.init()
    con.set_model(model.RHINE)

    con.run()

    print ('evaluation...')
    exp = evaluation.Evaluation()
    emb_dict = exp.load_emb("../res/" + data_set + "/embedding.vec." + mode + ".json")
    exp.evaluation(emb_dict)


if __name__ == "__main__":
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    data_set = 'dblp'
    mode = 'ap_pt_apt+pc_apc'

    print ('mode: {}'.format(mode))
    train_model(data_set, mode)
