# GF_Attack
This repository is the official Tensorflow implementation of "A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models".

Heng Chang, Yu Rong, Tingyang Xu, Wenbing Huang, Honglei Zhang, Peng Cui, Wenwu Zhu, Junzhou Huang, [A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models](https://arxiv.org/abs/1908.01297), AAAI 2020.

![framework](https://tva1.sinaimg.cn/large/006tNbRwgy1ga0buxk4wcj31mi0ig454.jpg)

## Requirements
The script has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):
* tensorflow (tested on 1.14.0)
* scipy (tested on 1.2.1)
* numpy (tested on 1.17.2)

## Run
- 2 order of graph filter, selecting Top-128 smallest eigen-values/vectors.
```bash
python main.py --dataset cora --K 2 --T 128
```
We only did Top-128 and Top-Half largest eigen-values/vectors to get the results in paper. To get better performance, tuning the hyper-parameters is highly encouraged.

## Acknowledgement
This repo is modified from [NETTACK](https://github.com/danielzuegner/nettack), and we sincerely thank them for their contributions.

## Reference
- If you find ``GF-Attack`` useful in your research, please cite the following in your manuscript:

```
@inproceedings{chang2020restricted,
  title={A restricted black-box adversarial framework towards attacking graph embedding models},
  author={Chang, Heng and Rong, Yu and Xu, Tingyang and Huang, Wenbing and Zhang, Honglei and Cui, Peng and Zhu, Wenwu and Huang, Junzhou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={3389--3396},
  year={2020}
}
```

