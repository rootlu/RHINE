# RHINE
Source code for AAAI 2019 paper ["**Relation Structure-Aware Heterogeneous Information Network Embedding**"](https://arxiv.org/abs/1905.08027)

# Requirements

- Python 2.7
- numpy

- scipy
- PyTorch (0.3.0)
- My machine with two GPUs (NVIDIA GTX-1080 *2) and two CPUs (Intel Xeon E5-2690 * 2)

# Description

```
RHINE/
├── code
│   ├── config
│   │   ├── Config.py：configs for model.
│   │   └──_init_.py
│   ├── evaluation.py: evaluate the performance of learned embeddings w.r.t clustering and classification
│   ├── models
│   │   ├── _init_.py
│   │   ├── Model.py: the super model with some functions
│   │   └── RHINE.py: our model
│   ├── preData
│   │   └── dblpDataHelper.py: data preparation for our mode
│   ├── release
│   │   ├── Sample_ARs.so: sampling with dll
│   │   └── Sample_IRs.so
│   └── trainRHINE.py: train model
├── data
│   └── dblp
│       ├── node2id.txt: the first line is the number of nodes, (node_type+node_name, node_id)
│       ├── paper_label.txt: (node_name, label)
│       ├── relation2id.txt: the first line is the number of relations, 		   (relation_name, relation_id)
│       ├── train2id_apc.txt:  (node1_id, node2_id, relation_id, weight)
│       ├── train2id_pc.txt
│       ├── train2id_ap.txt
│       ├── train2id_pt.txt
│       ├── train2id_apt.txt
│       ├── train2id_ARs.txt: the first line is the number of ARs triples, (node1_id, node2_id, relation_id, weight)
│       └── train2id_IRs.txt
├── README.md
└── res
    └── dblp
        └── embedding.vec.ap_pt_apt+pc_apc.json: the learned embeddings 
```

# Reference

```
@inproceedings{Yuanfu2019RHINE,
  title={Relation Structure-Aware Heterogeneous Information Network Embedding},
  author={Yuanfu Lu, Chuan Shi, Linmei Hu, Zhiyuan Liu.}
  booktitle={Proceedings of AAAI},
  year={2019}
}

```

