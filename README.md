DINOv2 Vision Toolkit

A practical computer vision toolkit built around DINOv2 for extracting robust visual features and building image understanding applications.

DINOv2 is a self-supervised vision foundation model developed by Meta AI Research. It learns useful visual representations from images without requiring human-provided labels during pretraining.

This project provides a simple and reusable interface for using DINOv2 embeddings in downstream computer vision workflows such as image similarity, visual search, classification, clustering, and feature extraction.

✨ Features

🧠 DINOv2-based visual feature extraction

🖼️ Image embedding generation

🔍 Image-to-image similarity

🔎 Visual image search

📊 Feature visualization and analysis

🏷️ Support for downstream classification

🧩 Easy integration into custom computer vision pipelines

⚡ GPU/CPU inference support

🔌 PyTorch-based implementation

🧠 What is DINOv2?

DINOv2 is a self-supervised Vision Transformer (ViT) model designed to learn general-purpose visual representations.

Instead of directly predicting a class such as:

Cat → 0
Dog → 1
Car → 2


DINOv2 learns a numerical representation of an image:

Image
   ↓
DINOv2
   ↓
Visual Embedding
   ↓
[0.12, -0.31, 0.84, ...]


These embeddings capture useful visual information and can be reused for many different computer vision tasks.

🚀 Use Cases
1. Image Similarity

Compare two images based on their visual features.

Image A ──→ DINOv2 ──→ Embedding A
                           │
                           ├── Similarity
                           │
Image B ──→ DINOv2 ──→ Embedding B


Useful for:

duplicate image detection

finding visually similar products

content recommendation

image matching

2. Visual Search

Create a search system where a user provides an image and the system retrieves visually similar images from a dataset.

Query Image
     ↓
DINOv2 Embedding
     ↓
Vector Database
     ↓
Similar Images

3. Image Classification

DINOv2 can be used as a pretrained backbone.

A lightweight classifier can be placed on top of the extracted features:

Image
  ↓
DINOv2
  ↓
Feature Vector
  ↓
Linear / MLP Classifier
  ↓
Class Prediction


This allows downstream models to reuse pretrained visual representations.

4. Image Clustering

Images can be converted into embeddings and grouped according to visual similarity.

Images
  ↓
DINOv2
  ↓
Embeddings
  ↓
Clustering
  ↓
Visual Groups


Possible algorithms include:

K-Means

DBSCAN

Hierarchical Clustering

🏗️ Architecture
                    ┌─────────────────┐
                    │      Image      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     DINOv2      │
                    │  Vision Encoder │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Visual Features │
                    │   / Embedding   │
                    └────────┬────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
        Similarity       Classification   Clustering
             │               │               │
             ▼               ▼               ▼
        Visual Search    Prediction      Image Groups

📦 Installation

Clone the repository:

git clone https://github.com/<YOUR_USERNAME>/dinov2-vision-toolkit.git
cd dinov2-vision-toolkit


Create a virtual environment:

python -m venv .venv


Activate it on Linux/macOS:

source .venv/bin/activate


Windows:

.venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


For GPU inference, install a PyTorch version compatible with your CUDA environment.

⚡ Quick Start

Load a pretrained DINOv2 backbone:

import torch

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)

model.eval()


Extract features:

with torch.no_grad():
    features = model(image)


The resulting tensor can be used as a visual representation for downstream applications.

📁 Project Structure
dinov2-main/
│
├── .gitignore
│   CODE_OF_CONDUCT.md
│   conda-extras.yaml
│   conda.yaml
│   CONTRIBUTING.md
│   hubconf.py
│   LICENSE
│   MODEL_CARD.md
│   pyproject.toml
│   README.md
│   requirements-dev.txt
│   requirements-extras.txt
│   requirements.txt
│   setup.cfg
│   setup.py
│   
├───.github
│   └───workflows
│           lint.yaml
│           
├───dinov2
│   │   __init__.py
│   │   
│   ├───configs
│   │   │   ssl_default_config.yaml
│   │   │   __init__.py
│   │   │   
│   │   ├───eval
│   │   │       vitb14_pretrain.yaml
│   │   │       vitb14_reg4_pretrain.yaml
│   │   │       vitg14_pretrain.yaml
│   │   │       vitg14_reg4_pretrain.yaml
│   │   │       vitl14_pretrain.yaml
│   │   │       vitl14_reg4_pretrain.yaml
│   │   │       vits14_pretrain.yaml
│   │   │       vits14_reg4_pretrain.yaml
│   │   │       
│   │   └───train
│   │           vitg14.yaml
│   │           vitl14.yaml
│   │           vitl16_short.yaml
│   │           
│   ├───data
│   │   │   adapters.py
│   │   │   augmentations.py
│   │   │   collate.py
│   │   │   loaders.py
│   │   │   masking.py
│   │   │   samplers.py
│   │   │   transforms.py
│   │   │   __init__.py
│   │   │   
│   │   └───datasets
│   │           decoders.py
│   │           extended.py
│   │           image_net.py
│   │           image_net_22k.py
│   │           __init__.py
│   │           
│   ├───distributed
│   │       __init__.py
│   │       
│   ├───eval
│   │   │   knn.py
│   │   │   linear.py
│   │   │   log_regression.py
│   │   │   metrics.py
│   │   │   setup.py
│   │   │   utils.py
│   │   │   __init__.py
│   │   │   
│   │   ├───depth
│   │   │   │   __init__.py
│   │   │   │   
│   │   │   ├───models
│   │   │   │   │   builder.py
│   │   │   │   │   __init__.py
│   │   │   │   │   
│   │   │   │   ├───backbones
│   │   │   │   │       vision_transformer.py
│   │   │   │   │       __init__.py
│   │   │   │   │       
│   │   │   │   ├───decode_heads
│   │   │   │   │       decode_head.py
│   │   │   │   │       dpt_head.py
│   │   │   │   │       linear_head.py
│   │   │   │   │       __init__.py
│   │   │   │   │       
│   │   │   │   ├───depther
│   │   │   │   │       base.py
│   │   │   │   │       encoder_decoder.py
│   │   │   │   │       __init__.py
│   │   │   │   │       
│   │   │   │   └───losses
│   │   │   │           gradientloss.py
│   │   │   │           sigloss.py
│   │   │   │           __init__.py
│   │   │   │           
│   │   │   └───ops
│   │   │           wrappers.py
│   │   │           __init__.py
│   │   │           
│   │   ├───segmentation
│   │   │   │   __init__.py
│   │   │   │   
│   │   │   ├───hooks
│   │   │   │       optimizer.py
│   │   │   │       __init__.py
│   │   │   │       
│   │   │   ├───models
│   │   │   │   │   __init__.py
│   │   │   │   │   
│   │   │   │   ├───backbones
│   │   │   │   │       vision_transformer.py
│   │   │   │   │       __init__.py
│   │   │   │   │       
│   │   │   │   └───decode_heads
│   │   │   │           linear_head.py
│   │   │   │           __init__.py
│   │   │   │           
│   │   │   └───utils
│   │   │           colormaps.py
│   │   │           __init__.py
│   │   │           
│   │   └───segmentation_m2f
│   │       │   __init__.py
│   │       │   
│   │       ├───core
│   │       │   │   __init__.py
│   │       │   │   
│   │       │   ├───anchor
│   │       │   │       builder.py
│   │       │   │       point_generator.py
│   │       │   │       __init__.py
│   │       │   │       
│   │       │   ├───box
│   │       │   │   │   builder.py
│   │       │   │   │   __init__.py
│   │       │   │   │   
│   │       │   │   └───samplers
│   │       │   │           base_sampler.py
│   │       │   │           mask_pseudo_sampler.py
│   │       │   │           mask_sampling_result.py
│   │       │   │           sampling_result.py
│   │       │   │           __init__.py
│   │       │   │           
│   │       │   └───utils
│   │       │           dist_utils.py
│   │       │           misc.py
│   │       │           __init__.py
│   │       │           
│   │       ├───models
│   │       │   │   builder.py
│   │       │   │   __init__.py
│   │       │   │   
│   │       │   ├───backbones
│   │       │   │       adapter_modules.py
│   │       │   │       drop_path.py
│   │       │   │       vit.py
│   │       │   │       vit_adapter.py
│   │       │   │       __init__.py
│   │       │   │       
│   │       │   ├───decode_heads
│   │       │   │       mask2former_head.py
│   │       │   │       __init__.py
│   │       │   │       
│   │       │   ├───losses
│   │       │   │       cross_entropy_loss.py
│   │       │   │       dice_loss.py
│   │       │   │       match_costs.py
│   │       │   │       __init__.py
│   │       │   │       
│   │       │   ├───plugins
│   │       │   │       msdeformattn_pixel_decoder.py
│   │       │   │       __init__.py
│   │       │   │       
│   │       │   ├───segmentors
│   │       │   │       encoder_decoder_mask2former.py
│   │       │   │       __init__.py
│   │       │   │       
│   │       │   └───utils
│   │       │           assigner.py
│   │       │           point_sample.py
│   │       │           positional_encoding.py
│   │       │           transformer.py
│   │       │           __init__.py
│   │       │           
│   │       └───ops
│   │           └───modules
│   │                   ms_deform_attn.py
│   │                   __init__.py
│   │                   
│   ├───fsdp
│   │       __init__.py
│   │       
│   ├───hub
│   │   │   backbones.py
│   │   │   classifiers.py
│   │   │   depthers.py
│   │   │   dinotxt.py
│   │   │   utils.py
│   │   │   __init__.py
│   │   │   
│   │   ├───depth
│   │   │       decode_heads.py
│   │   │       encoder_decoder.py
│   │   │       ops.py
│   │   │       __init__.py
│   │   │       
│   │   └───text
│   │           dinotxt_model.py
│   │           dinov2_wrapper.py
│   │           text_tower.py
│   │           text_transformer.py
│   │           tokenizer.py
│   │           vision_tower.py
│   │           
│   ├───layers
│   │       attention.py
│   │       block.py
│   │       dino_head.py
│   │       drop_path.py
│   │       layer_scale.py
│   │       mlp.py
│   │       patch_embed.py
│   │       swiglu_ffn.py
│   │       __init__.py
│   │       
│   ├───logging
│   │       helpers.py
│   │       __init__.py
│   │       
│   ├───loss
│   │       dino_clstoken_loss.py
│   │       ibot_patch_loss.py
│   │       koleo_loss.py
│   │       __init__.py
│   │       
│   ├───models
│   │       vision_transformer.py
│   │       __init__.py
│   │       
│   ├───run
│   │   │   submit.py
│   │   │   __init__.py
│   │   │   
│   │   ├───eval
│   │   │       knn.py
│   │   │       linear.py
│   │   │       log_regression.py
│   │   │       
│   │   └───train
│   │           train.py
│   │           
│   ├───thirdparty
│   │   └───CLIP
│   │       │   LICENSE
│   │       │   
│   │       └───clip
│   │               simple_tokenizer.py
│   │               
│   ├───train
│   │       ssl_meta_arch.py
│   │       train.py
│   │       __init__.py
│   │       
│   └───utils
│           cluster.py
│           config.py
│           dtype.py
│           param_groups.py
│           utils.py
│           __init__.py
│           
├───docs
│       ChannelAdaptiveDINO.png
│       README_CHANNEL_ADAPTIVE_DINO.md
│       
├───notebooks
│       depth_estimation.ipynb
│       dinotxt.ipynb
│       semantic_segmentation.ipynb
│       
└───scripts
        lint.sh


Run Project
1. Go to Project Directory
cd dinov2-main

2. Install Dependencies
pip install -r requirements.txt

3. Run the Project
python test_dinov2.py

4. Windows
py test_dinov2.py

Expected Result

The script loads the DINOv2 ViT-S/14 model and runs a test image through it. If successful, it displays the extracted feature vector and confirms that DINOv2 is working.

Loading DINOv2 ViT-S/14 model...
Feature vector shape: torch.Size([1, 384])
✅ DINOv2 is working!


🔬 Supported DINOv2 Models

Depending on available compute, the project can use different DINOv2 backbones:

Model	Approx. Parameters
ViT-S/14	21M
ViT-B/14	86M
ViT-L/14	300M
ViT-g/14	1.1B

For most development and experimentation, a smaller backbone can be used to reduce memory and inference cost.

🧪 Example Workflow

A typical visual-search pipeline can look like:

                 Offline
                   │
              Dataset Images
                   │
                   ▼
                DINOv2
                   │
                   ▼
             Image Embeddings
                   │
                   ▼
             Vector Database
                   │
                   │
───────────────────┼──────────────────
                   │
                 Online
                   │
              Query Image
                   │
                   ▼
                DINOv2
                   │
                   ▼
             Query Embedding
                   │
                   ▼
            Similarity Search
                   │
                   ▼
             Search Results

🎯 Why Use DINOv2?

DINOv2 provides strong general-purpose visual representations learned through self-supervised training.

This means you can often start with pretrained visual features instead of training a vision model completely from scratch for every new application.

The model can therefore act as a reusable vision backbone for downstream computer vision systems.

📚 Research

This project builds upon the DINOv2 research from Meta AI Research.

DINOv2: Learning Robust Visual Features without Supervision

Vision Transformers Need Registers

DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment

For the original implementation and pretrained models, refer to the official DINOv2 repository.

⚠️ Disclaimer

This repository is intended for research, experimentation, and educational purposes.

DINOv2 pretrained models, code, and associated components are subject to their respective licenses and terms. Check the upstream project before redistributing model weights or using them in production.

📄 License

This project's original code is released under the license specified in the LICENSE file.

The DINOv2 code and pretrained weights are provided by their respective upstream project and are subject to their applicable license terms.

🙏 Acknowledgements

This project builds on the work of the DINOv2 authors and the Meta AI Research / FAIR team.

Special thanks to the open-source computer vision community for making pretrained vision models accessible for research and development.

⭐ Contributing

Contributions, issues, and suggestions are welcome.

If you find this project useful, consider giving it a ⭐ on GitHub.

📖 Citation

If your work uses DINOv2, please cite the original research:

@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}

In short

DINOv2 Vision Toolkit = DINOv2 ko ek reusable computer-vision feature engine ki tarah use karke image embeddings, similarity, search, classification aur clustering jaise applications banana.







:new: [2025-08-14] *Please check out the more recent [DINOv3](https://github.com/facebookresearch/dinov3) effort continuing this line of work.*

[2025-06-11] *Added dino.txt inference code, following [DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](https://arxiv.org/abs/2412.16334).*

[2023-10-26] *Added DINOv2 backbones with registers, following [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).*

# DINOv2: Learning Robust Visual Features without Supervision

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

Maxime Oquab,
Timothée Darcet,
Théo Moutakanni,
Huy V. Vo,
Marc Szafraniec,
Vasil Khalidov,
Patrick Labatut,
Armand Joulin,
Piotr Bojanowski

[[`Paper #1`](https://arxiv.org/abs/2304.07193)] [`Paper #2`](https://arxiv.org/abs/2309.16588)] [[`Blog`](https://ai.facebook.com/blog/dino-v2-computer-vision-self-supervised-learning/)] [[`Demo`](https://dinov2.metademolab.com)] [[`BibTeX`](#citing-dinov2)]

PyTorch implementation and pretrained models for DINOv2. For details, see the papers: **[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)** and **[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)**.

DINOv2 models produce high-performance visual features that can be directly employed with classifiers as simple as linear layers on a variety of computer vision tasks; these visual features are robust and perform well across domains without any requirement for fine-tuning. The models were pretrained on a dataset of 142 M images without using any labels or annotations.

https://github.com/facebookresearch/dinov2/assets/60359573/f168823e-7922-415a-b429-578badf5c356

<div align="center">
  Visualization of the three first principal components of the patch features of all frames, mapped to RGB values.
</div>

## Pretrained models

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th>with<br />registers</th>
      <th>ImageNet<br />k-NN</th>
      <th>ImageNet<br />linear</th>
      <th>download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td align="right">21 M</td>
      <td align="center">:x:</td>
      <td align="right">79.0%</td>
      <td align="right">81.1%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td align="right">21 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">79.1%</td>
      <td align="right">80.9%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-B/14 distilled</td>
      <td align="right">86 M</td>
      <td align="center">:x:</td>
      <td align="right">82.1%</td>
      <td align="right">84.5%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-B/14 distilled</td>
      <td align="right">86 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">82.0%</td>
      <td align="right">84.6%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">:x:</td>
      <td align="right">83.5%</td>
      <td align="right">86.3%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td align="right">300 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">83.8%</td>
      <td align="right">86.7%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="center">:x:</td>
      <td align="right">83.5%</td>
      <td align="right">86.5%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth">backbone only</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="center">:white_check_mark:</td>
      <td align="right">83.7%</td>
      <td align="right">87.1%</td>
      <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth">backbone only</a></td>
    </tr>
  </tbody>
</table>

### Pretrained backbones (via PyTorch Hub)

Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch (the only required dependency for loading the model). Installing PyTorch with CUDA support is strongly recommended.

A corresponding [model card](MODEL_CARD.md) is included in the repository.

```python
import torch

# DINOv2
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# DINOv2 with registers
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
```

### Pretrained heads - Image classification

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">backbone</th>
      <th rowspan="2">with<br />registers</th>
      <th>download</th>
    </tr>
    <tr>
      <th>ImageNet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td align="center">:x:</td>
      <td>
        linear head (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear4_head.pth">4 layers</a>)
      </td>
    </tr>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td align="center">:white_check_mark:</td>
      <td>
        linear head (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear4_head.pth">4 layers</a>)
      </td>
    </tr>
    <tr>
      <td>ViT-B/14 distilled</td>
      <td align="center">:x:</td>
      <td>
        linear head (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear4_head.pth">4 layers</a>)
    </tr>
    <tr>
      <td>ViT-B/14 distilled</td>
      <td align="center">:white_check_mark:</td>
      <td>
        linear head (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear4_head.pth">4 layers</a>)
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td align="center">:x:</td>
      <td>
        linear head (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear4_head.pth">4 layers</a>)
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td align="center">:white_check_mark:</td>
      <td>
        linear head (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_linear4_head.pth">4 layers</a>)
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="center">:x:</td>
      <td>
        linear head (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear4_head.pth">4 layers</a>)
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="center">:white_check_mark:</td>
      <td>
        linear head (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_lreg4_inear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_linear4_head.pth">4 layers</a>)
    </tr>
  </tbody>
</table>

The (full) classifier models can be loaded via PyTorch Hub:

```python
import torch

# DINOv2
dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
dinov2_vitl14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
dinov2_vitg14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc')

# DINOv2 with registers
dinov2_vits14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
dinov2_vitb14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
dinov2_vitl14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')
dinov2_vitg14_reg_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg_lc')
```

### Pretrained heads - Depth estimation

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">backbone</th>
      <th colspan="2">download head</th>
    </tr>
    <tr>
      <th>NYUd</th>
      <th>KITTI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td>
        linear (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_linear4_head.pth">4 layers</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_dpt_head.pth">DPT</a>
      </td>
      <td>
        linear (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_linear4_head.pth">4 layers</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_kitti_dpt_head.pth">DPT</a>
      </td>
    </tr>
    <tr>
      <td>ViT-B/14 distilled</td>
      <td>
        linear (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_nyu_linear4_head.pth">4 layers</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_nyu_dpt_head.pth">DPT</a>
      </td>
      <td>
        linear (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_linear4_head.pth">4 layers</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_kitti_dpt_head.pth">DPT</a>
      </td>
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td>
        linear (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_nyu_linear4_head.pth">4 layers</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_nyu_dpt_head.pth">DPT</a>
      </td>
      <td>
        linear (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_linear4_head.pth">4 layers</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_kitti_dpt_head.pth">DPT</a>
      </td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>
        linear (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_nyu_linear4_head.pth">4 layers</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_nyu_dpt_head.pth">DPT</a>
      </td>
      <td>
        linear (<a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_linear_head.pth">1 layer</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_linear4_head.pth">4 layers</a>),
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_kitti_dpt_head.pth">DPT</a>
      </td>
    </tr>
  </tbody>
</table>

### Pretrained heads - Semantic segmentation

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">backbone</th>
      <th>download model</th>
      <th colspan="2">download head</th>
    </tr>
    <tr>
      <th>ADE20K</th>
      <th>ADE20K</th>
      <th>VOC2012</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14 distilled</td>
      <td></td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_linear_head.pth">linear</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_ade20k_ms_head.pth">multi-scale</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_voc2012_linear_head.pth">linear</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_voc2012_ms_head.pth">multi-scale</a>
      </td>
    </tr>
    <tr>
      <td>ViT-B/14 distilled</td>
      <td></td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_linear_head.pth">linear</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_ms_head.pth">multi-scale</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_voc2012_linear_head.pth">linear</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_voc2012_ms_head.pth">multi-scale</a>
      </td>
    </tr>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td></td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_linear_head.pth">linear</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_ade20k_ms_head.pth">multi-scale</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_voc2012_linear_head.pth">linear</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_voc2012_ms_head.pth">multi-scale</a>
      </td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth">Mask2Former</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_linear_head.pth">linear</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_ade20k_ms_head.pth">multi-scale</a>
      </td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_voc2012_linear_head.pth">linear</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_voc2012_ms_head.pth">multi-scale</a>
      </td>
    </tr>
  </tbody>
</table>


### Pretrained heads - Zero-shot tasks with dino.txt

<table style="margin: auto">
  <thead>
    <tr>
      <th rowspan="2">backbone</th>
      <th rowspan="2">with<br />registers</th>
      <th>download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-L/14 distilled</td>
      <td align="center">:white_check_mark:</td>
      <td>
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth">vision head</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth">text model</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/thirdparty/bpe_simple_vocab_16e6.txt.gz">vocabulary</a>,
        <a href="https://dl.fbaipublicfiles.com/dinov2/thirdparty/LICENSE">vocabulary license</a>
      </td>
    </tr>
  </tbody>
</table>

The (full) dino.txt model can be loaded via PyTorch Hub:

```python
import torch

# DINOv2
dinov2_vitl14_reg4_dinotxt_tet1280d20h24l = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l')
```


## Installation

The training and evaluation code requires PyTorch 2.0 and [xFormers](https://github.com/facebookresearch/xformers) 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Clone the repository and then create and activate a `dinov2` conda environment using the provided environment definition:

```shell
conda env create -f conda.yaml
conda activate dinov2
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)* - Clone the repository and then use the provided `requirements.txt` to install the dependencies:

```shell
pip install -r requirements.txt
```

For dense tasks (depth estimation and semantic segmentation), there are additional dependencies (specific versions of `mmcv` and `mmsegmentation`) which are captured in the `extras` dependency specifications:

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)**:

```shell
conda env create -f conda-extras.yaml
conda activate dinov2-extras
```

*[pip](https://pip.pypa.io/en/stable/getting-started/)*:

```shell
pip install -r requirements.txt -r requirements-extras.txt
```

## Data preparation

### ImageNet-1k

The root directory of the dataset should hold the following contents:

- `<ROOT>/test/ILSVRC2012_test_00000001.JPEG`
- `<ROOT>/test/[..]`
- `<ROOT>/test/ILSVRC2012_test_00100000.JPEG`
- `<ROOT>/train/n01440764/n01440764_10026.JPEG`
- `<ROOT>/train/[...]`
- `<ROOT>/train/n15075141/n15075141_9993.JPEG`
- `<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG`
- `<ROOT>/val/[...]`
- `<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG`
- `<ROOT>/labels.txt`

The provided dataset implementation expects a few additional metadata files to be present under the extra directory:

- `<EXTRA>/class-ids-TRAIN.npy`
- `<EXTRA>/class-ids-VAL.npy`
- `<EXTRA>/class-names-TRAIN.npy`
- `<EXTRA>/class-names-VAL.npy`
- `<EXTRA>/entries-TEST.npy`
- `<EXTRA>/entries-TRAIN.npy`
- `<EXTRA>/entries-VAL.npy`

These metadata files can be generated (once) with the following lines of Python code:

```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

Note that the root and extra directories do not have to be distinct directories.

### ImageNet-22k

Please adapt the [dataset class](dinov2/data/datasets/image_net_22k.py) to match your local setup.

<br />

:warning: To execute the commands provided in the next sections for training and evaluation, the `dinov2` package should be included in the Python module search path, i.e. simply prefix the command to run with `PYTHONPATH=.`.

## Training

### Fast setup: training DINOv2 ViT-L/16 on ImageNet-1k

Run DINOv2 training on 4 A100-80GB nodes (32 GPUs) in a SLURM cluster environment with submitit:

```shell
python dinov2/run/train/train.py \
    --nodes 4 \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

Training time is approximately 1 day and the resulting checkpoint should reach 81.6% on k-NN eval and 82.9% on linear eval.

The training code saves the weights of the teacher in the `eval` folder every 12500 iterations for evaluation.

### Long setup: training DINOv2 ViT-L/14 on ImageNet-22k

Run DINOv2 training on 12 A100-80GB nodes (96 GPUs) in a SLURM cluster environment with submitit:

```shell
python dinov2/run/train/train.py \
    --nodes 12 \
    --config-file dinov2/configs/train/vitl14.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet22k:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

Training time is approximately 3.3 days and the resulting checkpoint should reach 82.0% on k-NN eval and 84.5% on linear eval.

The training code saves the weights of the teacher in the `eval` folder every 12500 iterations for evaluation.


## Evaluation

The training code regularly saves the teacher weights. In order to evaluate the model, run the following evaluation on a single node:

### k-NN classification on ImageNet-1k

```shell
python dinov2/run/eval/knn.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/knn \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### Logistic regression classification on ImageNet-1k

```shell
python dinov2/run/eval/log_regression.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/logreg \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### Linear classification with data augmentation on ImageNet-1k

```shell
python dinov2/run/eval/linear.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/linear \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

We release the weights from evaluating the different models:

<table style="margin: auto">
  <tr>
    <th>model</th>
    <th>with<br />registers</th>
    <th>ImageNet<br />top-1</th>
    <th>linear evaluation</th>
  </tr>
  <tr>
    <td>ViT-S/14 distilled</td>
    <td align="center">:x:</td>
    <td align="right">81.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth">linear head weights</a></td>
  </tr>
  <tr>
    <td>ViT-S/14 distilled</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">80.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth">linear head weights</a></td>
  </tr>
  <tr>
    <td>ViT-B/14 distilled</td>
    <td align="center">:x:</td>
    <td align="right">84.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth">linear head weights</a></td>
  </tr>
  <tr>
    <td>ViT-B/14 distilled</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">84.4%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth">linear head weights</a></td>
  </tr>
  <tr>
    <td>ViT-L/14 distilled</td>
    <td align="center">:x:</td>
    <td align="right">86.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth">linear head weights</a></td>
  </tr>
  <tr>
    <td>ViT-L/14 distilled</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">86.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_linear_head.pth">linear head weights</a></td>
  </tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="center">:x:</td>
    <td align="right">86.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth">linear head weights</a></td>
  </tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="center">:white_check_mark:</td>
    <td align="right">87.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_linear_head.pth">linear head weights</a></td>
  </tr>
</table>

The performance of the provided pretrained model weights can be evaluated as follows on ImageNet-1k:

```shell
python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitg14_pretrain.yaml \
    --pretrained-weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

## Notebooks

A few notebooks are provided to help the community leverage the models and code:

<ul>
  <li><a href="https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb">Depth estimation</a> - How to load and use the depth heads in combination with a matching backbone via mmcv</li>
  <li><a href="https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb">Semantic segmentation</a> - How to load and use the segmentation heads in combination with a matching backbone via mmcv, and also how to load and use the Mask2Former-based segmentation model trained on ADE20K</li>
</ul>

## License

DINOv2 code and model weights are released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citing DINOv2

If you find this repository useful, please consider giving a star :star: and citation :t-rex::

```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

```
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```

```
@misc{jose2024dinov2meetstextunified,
  title={DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment}, 
  author={Cijo Jose and Théo Moutakanni and Dahyun Kang and Federico Baldassarre and Timothée Darcet and Hu Xu and Daniel Li and Marc Szafraniec and Michaël Ramamonjisoa and Maxime Oquab and Oriane Siméoni and Huy V. Vo and Patrick Labatut and Piotr Bojanowski},
  journal={arXiv:2412.16334},
  year={2024}
}
```
