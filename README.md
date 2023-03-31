# Editing Part & Commands
To run inpainting:
```
conda env create -f environment.yml
source activate ddrm
```
```
python main.py --ni --config cat.yml<create_config_file> --doc cat<Dataset that should be under datasets folder> --timesteps 20 --eta 0.85 --etaB 1 --deg inp<the_name_of_the_mask> --sigma_0 0.05 -i bedroom_inpaint<destination_folder>
```
---
### Custom Folder Experiment 1:
- prepared Dataset using: `python prepareDataset.py`
    - desired_size : 256
- Degradation: **Inpainting Lolcat**
- Config File: ImageNet_256_ood.config 
    - changed:
        - subset_1k :False
        - out_of_dist : True
- Dataset Structure:
    - exp/
        - datasets/
            - folder_name/
                - class_name1/
                    - Image1.png
                - class_name2/
                    - Image1.png
                
    
```
python main.py --ni --config imagenet_256_ood.yml --doc imageNet_ood_butterfly --timesteps 20 --eta 0.85 --etaB 1 --deg inp_lolcat --sigma_0 0.05 -i mimic_pair_inp_lolcat_imagenet_256
```
---
### Custom Folder Experiment 2:
[same as before ]
- Created Mask From Mask generator: left_down.npy
- Degradation: **Inpainting left down**
- Added condition for mask in Diffusion.py in runners folder
                   
```
python main.py --ni --config imagenet_256_ood.yml --doc imageNet_ood_butterfly --timesteps 20 --eta 0.85 --etaB 1 --deg inp_left_down --sigma_0 0.05 -i mimic_pair_inp_left_down_imagenet_256
```
---
### Custom Folder Experiment 3:
[same as before ]
- Created Mask From Mask generator: both_down.npy
- Degradation: **Inpainting both down**
- Added condition for mask in Diffusion.py in runners folder

```
python main.py --ni --config imagenet_256_ood.yml --doc imageNet_ood_butterfly --timesteps 20 --eta 0.85 --etaB 1 --deg inp_both_down --sigma_0 0.05 -i mimic_pair_inp_both_down_imagenet_256
```
---
### Custom Folder Experiment 4:
[same as before ]
- Created Mask From Mask generator: left_up.npy
- Degradation: **Inpainting left up**
- Added condition for mask in Diffusion.py in runners folder

```
python main.py --ni --config imagenet_256_ood.yml --doc imageNet_ood_butterfly --timesteps 20 --eta 0.85 --etaB 1 --deg inp_left_up --sigma_0 0.05 -i mimic_pair_inp_left_up_imagenet_256
```
---
### Custom Folder Experiment 5:
[same as before ]
- Created Mask From Mask generator: both_up.npy
- Degradation: **Inpainting both up**
- Added condition for mask in Diffusion.py in runners folder

```
python main.py --ni --config imagenet_256_ood.yml --doc imageNet_ood_butterfly --timesteps 20 --eta 0.85 --etaB 1 --deg inp_both_up --sigma_0 0.05 -i mimic_pair_inp_both_up_imagenet_256
```
---
# Denoising Diffusion Restoration Models (DDRM)

[arXiv](https://arxiv.org/abs/2201.11793) | [PDF](https://ddrm-ml.github.io/DDRM-paper.pdf) | [Project Website](https://ddrm-ml.github.io/)

[Bahjat Kawar](https://bahjat-kawar.github.io/)<sup>1</sup>, [Michael Elad](https://elad.cs.technion.ac.il/)<sup>1</sup>, [Stefano Ermon](http://cs.stanford.edu/~ermon)<sup>2</sup>, [Jiaming Song](http://tsong.me)<sup>2</sup><br />
<sup>1</sup> Technion, <sup>2</sup>Stanford University

DDRM uses pre-trained [DDPMs](https://hojonathanho.github.io/diffusion/) for solving general linear inverse problems. It does so efficiently and without problem-specific supervised training.

<img src="figures/ddrm-overview.png" alt="ddrm-overview" style="width:800px;"/>

## Running the Experiments
The code has been tested on PyTorch 1.8 and PyTorch 1.10. Please refer to `environment.yml` for a list of conda/mamba environments that can be used to run the code. 

### Pretrained models
We use pretrained models from [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion), [https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) and [https://github.com/ermongroup/SDEdit](https://github.com/ermongroup/SDEdit)

We use 1,000 images from the ImageNet validation set for comparison with other methods. The list of images is taken from [https://github.com/XingangPan/deep-generative-prior/](https://github.com/XingangPan/deep-generative-prior/)

The models and datasets are placed in the `exp/` folder as follows:
```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files
│   ├── celeba # all CelebA files
│   ├── imagenet # all ImageNet files
│   ├── ood # out of distribution ImageNet images
│   ├── ood_bedroom # out of distribution bedroom images
│   ├── ood_cat # out of distribution cat images
│   └── ood_celeba # out of distribution CelebA images
├── logs # contains checkpoints and samples produced during training
│   ├── celeba
│   │   └── celeba_hq.ckpt # the checkpoint file for CelebA-HQ
│   ├── diffusion_models_converted
│   │   └── ema_diffusion_lsun_<category>_model
│   │       └── model-x.ckpt # the checkpoint file saved at the x-th training iteration
│   ├── imagenet # ImageNet checkpoint files
│   │   ├── 256x256_classifier.pt
│   │   ├── 256x256_diffusion.pt
│   │   ├── 256x256_diffusion_uncond.pt
│   │   ├── 512x512_classifier.pt
│   │   └── 512x512_diffusion.pt
├── image_samples # contains generated samples
└── imagenet_val_1k.txt # list of the 1k images used in ImageNet-1K.
```

We note that some models may not generate high-quality samples in unconditional image synthesis; this is especially the case for the pre-trained CelebA model.

### Sampling from the model

The general command to sample from the model is as follows:
```
python main.py --ni --config {CONFIG}.yml --doc {DATASET} --timesteps {STEPS} --eta {ETA} --etaB {ETA_B} --deg {DEGRADATION} --sigma_0 {SIGMA_0} -i {IMAGE_FOLDER}
```
where the following are options
- `ETA` is the eta hyperparameter in the paper. (default: `0.85`)
- `ETA_B` is the eta_b hyperparameter in the paper. (default: `1`)
- `STEPS` controls how many timesteps used in the process.
- `DEGREDATION` is the type of degredation allowed. (One of: `cs2`, `cs4`, `inp`, `inp_lolcat`, `inp_lorem`, `deno`, `deblur_uni`, `deblur_gauss`, `deblur_aniso`, `sr2`, `sr4`, `sr8`, `sr16`, `sr_bicubic4`, `sr_bicubic8`, `sr_bicubic16` `color`)
- `SIGMA_0` is the noise observed in y.
- `CONFIG` is the name of the config file (see `configs/` for a list), including hyperparameters such as batch size and network architectures.
- `DATASET` is the name of the dataset used, to determine where the checkpoint file is found.
- `IMAGE_FOLDER` is the name of the folder the resulting images will be placed in (default: `images`)

For example, for sampling noisy 4x super resolution from the ImageNet 256x256 unconditional model using 20 steps:
```
python main.py --ni --config imagenet_256.yml --doc imagenet --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
```
The generated images are place in the `<exp>/image_samples/{IMAGE_FOLDER}` folder, where `orig_{id}.png`, `y0_{id}.png`, `{id}_-1.png` refer to the original, degraded, restored images respectively.

The config files contain a setting controlling whether to test on samples from the trained dataset's distribution or not.

### Images for Demonstration Purposes
A list of images for demonstration purposes can be found here: [https://github.com/jiamings/ddrm-exp-datasets](https://github.com/jiamings/ddrm-exp-datasets). Place them under the `<exp>/datasets` folder, and these commands can be excecuted directly:

CelebA noisy 4x super-resolution:
```
python main.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05
```

General content images uniform deblurring:
```
python main.py --ni --config imagenet_256.yml --doc imagenet_ood --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.0 -i imagenet_sr4_sigma_0.0
```

Bedroom noisy 4x super-resolution:
```
python main.py --ni --config bedroom.yml --doc bedroom --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i bedroom_sr4_sigma_0.05
```

## References and Acknowledgements
```
@inproceedings{kawar2022denoising,
    title={Denoising Diffusion Restoration Models},
    author={Bahjat Kawar and Michael Elad and Stefano Ermon and Jiaming Song},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```

This implementation is based on / inspired by:
- [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) (the DDPM TensorFlow repo),
- [https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) (PyTorch helper that loads the DDPM model), and
- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) (code structure)
