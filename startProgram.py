import os

os.system("python main.py --ni --config imagenet_256_ood.yml \
--doc color_swatch --timesteps 20 --eta 0.85 --etaB 1 --deg\
 inp_half --sigma_0 0.05 -i color_checker_inp_half_imagenet_256")