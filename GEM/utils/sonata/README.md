# Sonata
This repo is the official project repository of the paper **_Sonata: Self-Supervised Learning of Reliable Point Representations_** and is mainly used for providing pre-trained models, inference code and visualization demo. For reproduce pre-training process of Sonate, please refer to our **[Pointcept](https://github.com/Pointcept/Pointcept)** codebase.  
[ Pretrain ] [Sonata] - [ [arXiv]() ] [ [Bib]() ] 

<div align='left'>
<img src="assets/teaser.png" alt="teaser" width="800" />
</div>

## Highlights
- *Jan xx, 2025*: We released our project repo for PTv3, if you have any questions related to our work, please feel free to open an issue.

## Overview
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)

## Installation
This repo provide two ways of installation: **standalone mode** and **package mode**.  
- The **standalone mode** is recommended for users who want to use the code for quick inference and visualization. We provide a most easily way to install the environment by using `conda` environment file. The whole environment including `cuda` and `pytorch` can be easily installed by running the following command:
  ```bash
  # Create and activate conda environment named as 'sonata'
  # cuda: 12.4, pytorch: 2.5.0
  conda env create -f environment.yml
  conda activate sonata
  ```

- The **package mode** is recommended for users who want to inject our model into their own codebase. We provide a `setup.py` file for installation. You can install the package by running the following command:
  ```bash
  # Ensure Cuda and Pytorch are already installed in your local environment
  
  # CUDA_VERSION: cuda version of local environment (e.g., 124)
  # TORCH_VERSION: torch version of local environment (e.g., 2.5.0)
  pip install spconv-cu${CUDA_VERSION}
  pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu${CUDA_VERSION}.html
  pip install git+https://github.com/Dao-AILab/flash-attention.git
  pip install huggingface_hub timm
  
  # (optional, or directly copy the sonata folder to your project)
  python setup.py install
  ```
  Additionally, for running our **demo code**, the following packages are also required:
  ```bash
  pip install open3d fast_pytorch_kmeans psutil numpy==1.26.4  # currently, open3d does not support numpy 2.x
  ```

## Quick Start
- **Data.** Organize your data in a dictionary with the following format:
  ```python
  point = {
    "coord": numpy.array,  # (N, 3)
    "color": numpy.array,  # (N, 3)
    "normal": numpy.array,  # (N, 3)
    "segment": numpy.array,  # (N,) optional
  }
  ```
  One example of the data can be loaded by running the following command:
  ```python
  point = sonata.data.load("example1")
  ```
- **Transform.** The data transform pipeline is shared as the one used in Pointcept codebase. You can use the following code to construct the transform pipeline:
  ```python
  config = [
      dict(type="CenterShift", apply_z=True),
      dict(
          type="GridSample",
          grid_size=0.02,
          hash_type="fnv",
          mode="train",
          return_grid_coord=True,
      ),
      dict(type="NormalizeColor"),
      dict(type="ToTensor"),
      dict(
          type="Collect",
          keys=("coord", "grid_coord", "color"),
          feat_keys=("coord", "color", "normal"),
      ),
  ]
  transform = sonata.transform.Compose(config)
  ```
  The above default inference augmentation pipeline can also be acquired by running the following command:
  ```python
  transform = sonata.transform.default()
  ```
- **Model.** Load the pre-trained model by running the following command:
  ```python
  # Load the pre-trained model from Huggingface
  # supported models: "sonata"
  # ckpt is cached in ~/.cache/sonata/ckpt, and the path can be customized by setting 'download_root'
  model = sonata.model.load("sonata").cuda()
  
  # Load the pre-trained model from local path
  # assume the ckpt file is stored in the 'ckpt' folder
  model = sonata.model.load("ckpt/sonata.pth").cuda()
  
  # the ckpt file store the config and state_dict of pretrained model
  ```
- **Inference.** Run the inference by running the following command:
  ```python
  point = transform(point)
  if isinstance(data[key], torch.Tensor):
      data[key] = data[key].cuda(non_blocking=True)
  point = model(point)
  ```
  As Sonata is a pre-trained **encoder-only** PTv3, the default output of the model is point cloud after hieratical encoding. The encoded point feature can be mapping back to original scale with the following code:
  ```python
  for _ in range(2):
      assert "pooling_parent" in point.keys()
      assert "pooling_inverse" in point.keys()
      parent = point.pop("pooling_parent")
      inverse = point.pop("pooling_inverse")
      parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
      point = parent
  while "pooling_parent" in point.keys():
      assert "pooling_inverse" in point.keys()
      parent = point.pop("pooling_parent")
      inverse = point.pop("pooling_inverse")
      parent.feat = point.feat[inverse]
      point = parent
  ```
- **Visualization.** We provide the similarity heatmap and PCA visualization demo in the `demo` folder. You can run the following command to visualize the result:
  ```bash
  python demo/similarity.py
  python demo/pca.py
  python demo/fair_embody_demo.py
  ```

## Citation
If you find _Sonata_ useful to your research, please consider citing our work as an acknowledgment. (੭ˊ꒳​ˋ)੭✧
```bib
@inproceedings{wu2024ptv3,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2024ppt,
    title={Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training},
    author={Wu, Xiaoyang and Tian, Zhuotao and Wen, Xin and Peng, Bohao and Liu, Xihui and Yu, Kaicheng and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2023masked,
  title={Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning},
  author={Wu, Xiaoyang and Wen, Xin and Liu, Xihui and Zhao, Hengshuang},
  journal={CVPR},
  year={2023}
}
```
```bib
@inproceedings{wu2022ptv2,
    title={Point transformer V2: Grouped Vector Attention and Partition-based Pooling},
    author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
    booktitle={NeurIPS},
    year={2022}
}
```
```bib
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished={\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```