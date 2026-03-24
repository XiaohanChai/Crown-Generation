# OctFusion for Crown Generation

![crown generation with octfusion](https://github.com/user-attachments/assets/5ab72901-3747-471b-a01f-dc49bfe66bdb)

In this study, we have adopted the algorithm from the paper “OctFusion: Octree-based Diffusion Models for 3D Shape Generation” and applied it to the generation of dental crowns in the field of stomatology, achieving preliminary results.

The model is used for dental crown generation. The training data consists of three crowns: the second premolar, the first molar, and the second molar, and unconditional training and generation are performed. As shown in the figure, the first row illustrates the process of dental crown generation during training. The second row shows the model of octree generation. The third row presents examples of generated crown models. It can be observed that the generated crown models exhibit reasonable anatomical morphology and distinct personalized features. For instance, details such as the Carabelli cusp and wear facets can also be generated.

This study was trained on a dataset of 705 dental crown samples. Due to the sensitivity of medical data, the dental crown dataset is currently not open-source.

Code release for the paper "OctFusion: Octree-based Diffusion Models for 3D Shape Generation". Computer Graphics Forum (presented at SGP 2025)

[[`arXiv`](https://arxiv.org/abs/2408.14732)]
[[`BibTex`](#citation)]


## 1. Installation
1. Clone this repository
```bash
git clone https://github.com/XiaohanChai/PKUSS-JingwenYang-tooth-crown_generation-octfusion.git
cd octfusion
```
2. Create a `Conda` environment.
```bash
conda create -n octfusion python=3.11 -y && conda activate octfusion
```

3. Install PyTorch with Conda
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. Install other requirements.
```bash
pip3 install -r requirements.txt 
```

## 2. Train from scratch
### 2.1 Data Preparation

1. Convert the meshes in `Crown` to signed distance fields (SDFs).
We use the similar data preparation as [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN.git). 
```bash
python tools/repair_mesh_cxh.py --run convert_mesh_to_sdf
```

### 2.2 Train OctFusion
1. VAE Training.
```bash
sh scripts/run_snet_vae.sh train vae crown
```

2. Train the first stage model. 
```bash
sh scripts/run_snet_uncond.sh train lr crown
```

3. Load the pretrained first stage model and train the second stage.  
```bash
sh scripts/run_snet_uncond.sh train hr crown
```
# <a name="citation"></a> Citation

If you find this code helpful, please consider citing:

Arxiv version of the original article of "Octfusion: Octree-based diffusion models for 3d shape generation"
```BibTeX
@article{xiong2024octfusion,
  title={Octfusion: Octree-based diffusion models for 3d shape generation},
  author={Xiong, Bojun and Wei, Si-Tong and Zheng, Xin-Yang and Cao, Yan-Pei and Lian, Zhouhui and Wang, Peng-Shuai},
  journal={arXiv preprint arXiv:2408.14732},
  year={2024}
}
```

# Acknowledgement
This code borrows heavely from [OctFusion](https://github.com/octree-nn/octfusion), [SDFusion](https://github.com/yccyenchicheng/SDFusion), [LAS-Diffusion](https://github.com/Zhengxinyang/LAS-Diffusion), [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN). We thank the authors for their great work. The followings packages are required to compute the SDF: [mesh2sdf](https://github.com/wang-ps/mesh2sdf).
