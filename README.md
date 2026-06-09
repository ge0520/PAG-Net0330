#  PAG-Net: An Novel Image Dehazing Method using Patch-aware Guidance
## About Paper

We are delighted to inform everyone that our paper has been successfully accepted by IJCNN (2026)

## Abstract

Image dehazing is a fundamental task in low-level computer vision, aiming to restore haze-free images from hazy observations. However, existing approaches often face a trade-off between preserving global structural consistency and recovering fine-grained local textures. To address this challenge, we propose a patch-aware strategy to explicitly guide the dehazing process.
Specifically, we introduce the Patch-Aware Modulation Module(PAMM), which is composed of patch-aware guidance and sensitivity-guided modulation. First, the patch-aware guidance generates a spatially varying sensitivity map that characterizes the global distribution of haze density across the image. Subsequently, this sensitivity map is employed to modulate the global contextual features extracted by a dedicated large-kernel convolution, thereby achieving a more harmonious integration of global structure and local detail. Extensive experiments on
multiple benchmark datasets demonstrate that our patch-aware strategy consistently improves both quantitative metrics and qualitative visual fidelity over state-of-the-art methods.

## Data

1. [RESIDE Dataset](https://sites.google.com/site/boyilics/website-builder/reside)


## Code

Partial implementation and the pre-train models is available in the [code](PAG-Net0330/code) folder.  

## Performance

1. RESIDE-IN

![](MFF-Net/picture/MMFlood_result.png)

2. RESIDE-OUT

![](MFF-Net/picture/sen1flood.jpg)

3. NH-HAZE

![](MFF-Net/picture/ETCI.jpg)
