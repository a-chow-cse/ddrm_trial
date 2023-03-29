import numpy as np 
import argparse
from PIL import Image

## instructions to run: python see_mask.py -name "npy file name"

def make_image_of_mask(args):
    npy_file=np.load(args.name)
    print("Shape: ",npy_file.shape)
    img=Image.fromarray(np.uint8(npy_file*255), 'L')
    img.save(args.name+"_img.jpeg")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-name",type=str,default="lorem3.npy")
    args=parser.parse_args()
    print(args.name)
    make_image_of_mask(args)


main()


