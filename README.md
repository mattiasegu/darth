# DARTH
This repository provides the official implementation of ["DARTH: Holistic Test-time Adaptation for Multiple Object Tracking"](https://openaccess.thecvf.com/content/ICCV2023/html/Segu_DARTH_Holistic_Test-time_Adaptation_for_Multiple_Object_Tracking_ICCV_2023_paper.html) (ICCV 2023).


![](resources/darth_banner.png)

> [**DARTH: Holistic Test-time Adaptation for Multiple Object Tracking**](),            
> Mattia Segu, Bernt Schiele, Fisher Yu,
> ICCV 2023
> *Project Website ([DARTH](http://vis.xyz/pub/darth/))* 
> *Paper ([arXiv 2310.01926](https://arxiv.org/abs/2310.01926))*

## Teaser
https://github.com/mattiasegu/darth/assets/44324619/5b08c8a2-c3f0-4d8c-a0c6-e85107c8f508

## Abstract
Multiple object tracking (MOT) is a fundamental component of perception systems for autonomous driving, and its robustness to unseen conditions is a requirement to avoid life-critical failures. Despite the urge of safety in driving systems, no solution to the MOT adaptation problem to domain shift in test-time conditions has ever been proposed. However, the nature of a MOT system is manifold - requiring object detection and instance association - and adapting all its components is non-trivial. In this paper, we analyze the effect of domain shift on appearance-based trackers, and introduce DARTH, a holistic test-time adaptation framework for MOT. We propose a detection consistency formulation to adapt object detection in a self-supervised fashion, while adapting the instance appearance representations via our novel patch contrastive loss. We evaluate our method on a variety of domain shifts - including sim-to-real, outdoor-to-indoor, indoor-to-outdoor - and substantially improve the source model performance on all metrics.


## Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for installation and to [DATASET.md](docs/DATASET.md) for datasets preparation.

## Get Started
Please see [GET_STARTED.md](docs/GET_STARTED.md) for the basic usage of DARTH.

## Citation
If you find this project useful in your research, please consider citing:

```latex
@inproceedings{segu2023darth,
  title={DARTH: Holistic Test-time Adaptation for Multiple Object Tracking},
  author={Segu, Mattia and Schiele, Bernt and Yu, Fisher},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9717--9727},
  year={2023}
}
```

```latex
@inproceedings{sun2022shift,
  title={SHIFT: a synthetic driving dataset for continuous multi-task domain adaptation},
  author={Sun, Tao and Segu, Mattia and Postels, Janis and Wang, Yuxuan and Van Gool, Luc and Schiele, Bernt and Tombari, Federico and Yu, Fisher},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21371--21382},
  year={2022}
}
```