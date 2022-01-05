# Fast Gradient Non-sign Methods

## This repo is the official **Tensorflow code** implementation of our paper [Fast Gradient Non-sign Methods](https://arxiv.org/pdf/2110.12734.pdf). In our paper, we give a theoretical analysis of the side-effect of 'sign' which is adopted in current methods, and further give a correction to 'sign' as well as propose the methods FGNM.

## Condensed Abstract
Adversarial attacks make their success in “fooling” DNNs and among them, gradient-based algorithms become one of the main streams. Based on the linearity hypothesis [12], under ℓ∞ constraint, 𝑠𝑖𝑔𝑛 operation applied to the gradients is a good choice for generating perturbations. However, the side-effect from such operation exists since it leads to the bias of direction between the real gradients and the perturbations. In other words, current methods contain a gap between real gradients and actual noises, which leads to biased and inefficient attacks. Therefore in this paper, based on the Taylor expansion, the bias is analyzed theoretically and the correction of sign, i.e., Fast Gradient Non-sign Method (FGNM), is further proposed. Notably, FGNM is a general routine, which can seamlessly replace the conventional 𝑠𝑖𝑔𝑛 operation in gradient-based attacks with negligible extra computational cost. Extensive experiments demonstrate the effectiveness of our methods. Specifically, ours outperform them by 27.5% at most and 9.5% on average. 

## Effectiveness of our FGNM
<p align="center">
  <img src="https://i.loli.net/2021/04/20/7gQ4JkiMRfOewhN.png" alt="Results"/>
</p>

<p align="center">
  <img src="https://i.loli.net/2021/04/20/OqB7WfVnHt5Izj3.png" alt="Visualization"/>
</p>

tSuc and tTR performance w.r.t. relative layer depth for multiple transfer scenarios. The figure is split into four phases, corresponding to black-box attacks transferring from Den121, Inc-v3, VGG19, and Res50. All of our proposed methods outperform AA in most cases, which indicates the effectiveness of statistic alignment on various layers.


## Visualization of the adversarial examples
![image](https://github.com/yaya-cheng/PAA-GAA/blob/main/visualization%20of%20adversarial%20examples/all.png)

Visualization of adversarial examples with Den121 as the white-box. Original class: goldfish, targeted class: axolotl. Fromleft to right: Raw, MIFGSM, AA and PAAp.

- Download the models

  - [Normlly trained models](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained) (DenseNet can be found in [here](https://github.com/flyyufelix/DenseNet-Keras))
  - [Ensemble  adversarial trained models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models?spm=5176.12282029.0.0.3a9e79b7cynrQf)

- Then put these models into ".models/"

- Run PAAp on Den121 under 2nd:

  ```python
  python fp_attack_den121.py --method 1 --kernel_type poly --kernel_for_furthe l_poly --byRank 1 --targetcls 2 
  ```

- Run GAA on Den121 under 2nd:

  ```python
  python fp_attack_den121.py --mmdMethod 2 --GAA 1 --byRank 1 --targetcls 2 
  ```

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

