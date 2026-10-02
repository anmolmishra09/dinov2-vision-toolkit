# Model Card for DINOv2-S/B/L/g

These are Vision Transformer models trained following the methods described in the papers:

* **DINOv2: Learning Robust Visual Features without Supervision**
* **Vision Transformers Need Registers**

We provide 8 models:

* 1 ViT-g model trained from scratch, with 3 ViT-S/B/L models distilled from the ViT-g, without registers.
* 1 ViT-g model trained from scratch, with 3 ViT-S/B/L models distilled from the ViT-g, with registers.

## Model Details

The model takes an image as input and returns a class token and patch tokens, and optionally 4 register tokens.

The embedding dimensions are:

* **384** for ViT-S
* **768** for ViT-B
* **1024** for ViT-L
* **1536** for ViT-g

The models follow a Vision Transformer architecture with a patch size of 14.

For models with registers, 4 learned register tokens are added to the input sequence after the patch embedding.

For a `224 × 224` image, this results in:

* 1 class token
* 256 patch tokens
* 4 additional register tokens for register-based models

The models can accept larger images provided that their height and width are multiples of the patch size (`14`). If this condition is not satisfied, the model crops the image to the closest smaller multiple of 14.

### Model Description

* **Developed by:** Meta AI
* **Model type:** Vision Transformer
* **License:** Apache License 2.0
* **Repository:** https://github.com/facebookresearch/dinov2
* **Paper:** https://arxiv.org/abs/2304.07193
* **Demo:** https://dinov2.metademolab.com/

## Uses

The models are vision backbones that provide general-purpose visual features for downstream tasks.

### Direct Use

The models can be used without fine-tuning with relatively simple downstream classifiers, such as linear layers, to obtain competitive results on tasks including:

* Depth estimation
* Semantic segmentation
* Image classification
* Image retrieval

Examples include:

* k-NN classifiers using the class token
* Logistic regression classifiers using the class token
* Linear classifiers using the class token and the average of patch tokens
* Nearest-neighbor image retrieval

### Downstream Use

Fine-tuning the models is technically possible and can provide additional performance improvements. The authors measured approximately a 2% improvement on ImageNet-1k classification.

However, fine-tuning is recommended only when necessary because the pretrained features already provide strong performance out of the box.

## Bias, Risks, and Limitations

Despite improvements resulting from the use of training methods that do not rely on manual annotations, the models still exhibit significant biases toward rich households from Western countries.

### Recommendations

Fine-tuning may increase biases in the resulting features when the fine-tuning dataset contains similar biases.

Users should evaluate the model on data representative of their intended deployment environment.

## How to Get Started with the Model

### DINOv2

```python
import torch

# DINOv2
dinov2_vits14 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

dinov2_vitb14 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
)

dinov2_vitl14 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitl14"
)

dinov2_vitg14 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitg14"
)
```

### DINOv2 with Registers

```python
import torch

dinov2_vits14_reg = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14_reg"
)

dinov2_vitb14_reg = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14_reg"
)

dinov2_vitl14_reg = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitl14_reg"
)

dinov2_vitg14_reg = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitg14_reg"
)
```

## Training Details

### Training Data

* **Training dataset:** LVD-142M
* **Training regime:** FP16 using PyTorch FSDP mixed precision

### Training Procedure

The training objective combines:

* DINO self-distillation loss with multi-crop augmentation
* iBOT masked-image modeling loss
* KoLeo regularization on `[CLS]` tokens

### Architectures

| Model    | Parameters | Patch Size | Embedding Dimension | Attention Heads | FFN    |
| -------- | ---------: | ---------: | ------------------: | --------------: | ------ |
| ViT-S/14 |        21M |         14 |                 384 |               6 | MLP    |
| ViT-B/14 |        86M |         14 |                 768 |              12 | MLP    |
| ViT-L/14 |       0.3B |         14 |                1024 |              16 | MLP    |
| ViT-g/14 |       1.1B |         14 |                1536 |              24 | SwiGLU |

### Distillation

Distillation follows the standard DINOv2 pretraining procedure, except that the teacher is a pretrained ViT-g model that remains frozen during distillation.

## Evaluation

The evaluation results below are reported from the associated DINOv2 research papers.

| Model    | Registers | ImageNet-1k k-NN | ImageNet-1k Linear | ImageNet-1k Linear V2 | NYU-Depth v2 RMSE | SUN-RGBD RMSE | ADE20k mAP | iNaturalist 2018 | Oxford-H mAP |
| -------- | :-------: | ---------------: | -----------------: | --------------------: | ----------------: | ------------: | ---------: | ---------------: | -----------: |
| ViT-S/14 |     ❌     |            79.0% |              81.1% |                 70.8% |             0.417 |         0.431 |       47.2 |            69.5% |         43.2 |
| ViT-S/14 |     ✅     |            79.1% |              80.9% |                 71.0% |               N/A |           N/A |        N/A |            67.6% |         39.5 |
| ViT-B/14 |     ❌     |            82.1% |              84.5% |                 74.9% |             0.362 |         0.400 |       51.3 |            76.3% |         49.5 |
| ViT-B/14 |     ✅     |            82.0% |              84.6% |                 75.6% |               N/A |           N/A |        N/A |            73.8% |         51.0 |
| ViT-L/14 |     ❌     |            83.5% |              86.3% |                 77.6% |             0.333 |         0.396 |       53.1 |            79.8% |         54.0 |
| ViT-L/14 |     ✅     |            83.8% |              86.7% |                 78.5% |               N/A |           N/A |        N/A |            80.9% |         55.7 |
| ViT-g/14 |     ❌     |            83.5% |              86.5% |                 78.4% |             0.298 |         0.362 |       53.0 |            81.6% |         52.3 |
| ViT-g/14 |     ✅     |            83.7% |              87.1% |                 78.8% |               N/A |           N/A |        N/A |            81.5% |         58.2 |

### Evaluation Protocols

The evaluation tasks use the following protocols:

* **ImageNet-1k k-NN:** k-nearest-neighbor classification using the class token.
* **ImageNet-1k Linear:** linear classification using frozen features.
* **ImageNet-1k Linear V2:** an alternative linear evaluation protocol.
* **NYU-Depth v2:** depth estimation using frozen features and linear layers.
* **SUN-RGBD:** depth estimation evaluation.
* **ADE20k:** semantic segmentation using a multiscale evaluation protocol.
* **iNaturalist 2018:** image classification using a linear classifier.
* **Oxford-H:** image retrieval using nearest-neighbor features.

For complete evaluation protocols, please refer to the associated papers.

## Environmental Impact

### Hardware

* **GPU:** NVIDIA A100
* **Cloud provider:** Private infrastructure
* **Compute region:** USA

### Training Compute

| Model              | GPU Hours |
| ------------------ | --------: |
| ViT-g training     |    22,000 |
| ViT-S distillation |     4,500 |
| ViT-B distillation |     5,300 |
| ViT-L distillation |     8,000 |

* **Estimated carbon emissions:** 7 tCO₂eq

## Software

* PyTorch 2.0
* xFormers 0.0.18

## License

The models are released under the **Apache License 2.0**.

## References

### DINOv2

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```

### Vision Transformers Need Registers

```bibtex
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Timothée and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```

## Acknowledgements

This model is based on research conducted by Meta AI and the contributors to the DINOv2 project.

For additional information, implementation details, checkpoints, and evaluation methodology, please refer to the official DINOv2 repository and associated research papers.
