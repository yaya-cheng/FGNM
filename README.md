# Fast Gradient Non-sign Methods

## This repo is the official **Tensorflow code** implementation of our paper [Fast Gradient Non-sign Methods](https://arxiv.org/pdf/2110.12734.pdf). 

## In our paper, we give a theoretical analysis of the side-effect of 'sign' which is adopted in current methods, and further give a correction to 'sign' as well as propose the methods FGNM.

## Condensed Abstract
Adversarial attacks make their success in “fooling” DNNs and among them, gradient-based algorithms become one of the main streams. Based on the linearity hypothesis [12], under ℓ∞ constraint, 𝑠𝑖𝑔𝑛 operation applied to the gradients is a good choice for generating perturbations. However, the side-effect from such operation exists since it leads to the bias of direction between the real gradients and the perturbations. In other words, current methods contain a gap between real gradients and actual noises, which leads to biased and inefficient attacks. Therefore in this paper, based on the Taylor expansion, the bias is analyzed theoretically and the correction of sign, i.e., Fast Gradient Non-sign Method (FGNM), is further proposed. Notably, FGNM is a general routine, which can seamlessly replace the conventional 𝑠𝑖𝑔𝑛 operation in gradient-based attacks with negligible extra computational cost. Extensive experiments demonstrate the effectiveness of our methods. Specifically, ours outperform them by 27.5% at most and 9.5% on average. 

## Effectiveness of our FGNM
<p align="center">
  <img src="https://i.loli.net/2021/04/20/7gQ4JkiMRfOewhN.png" alt="Results"/>
</p>

## Visualization of the adversarial examples
<p align="center">
  <img src="https://i.loli.net/2021/04/20/OqB7WfVnHt5Izj3.png" alt="Visualization"/>
</p>

- Download the models

  - [Normlly trained models](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained) (DenseNet can be found in [here](https://github.com/flyyufelix/DenseNet-Keras))
  - [Ensemble  adversarial trained models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models?spm=5176.12282029.0.0.3a9e79b7cynrQf)

- Then put these models into ".models/"

- Run IFGNM_N:

  ```python
  python attack.py --method if --mode affine 

## Citing this work

If you find this work is useful in your research, please consider citing:

```
@article{cheng2021fast,
  title={Fast Gradient Non-sign Methods},
  author={Cheng, Yaya and Zhu, Xiaosu and Zhang, Qilong and Gao, Lianli and Song, Jingkuan},
  journal={arXiv preprint arXiv:2110.12734},
  year={2021}
}
```

