import numpy as np
import os
from PIL import Image
def left_down_mask(im):
    im_arr = np.array(im)

    mask=np.ones((256,256))

    for i in range(145,165):
        for j in range(65,120):
            im_arr[i,j]=1
            mask[i,j]=0

    im= Image.fromarray(im_arr)
    np.save('left_down.npy', mask)  
    im.save(os.getcwd()+"/editedLuscombei.png")

def both_down_mask(im):
    im_arr = np.array(im)

    mask=np.ones((256,256))

    for i in range(145,165):
        for j in range(65,120):
            im_arr[i,j]=1
            mask[i,j]=0
        for k in range(137,192):
            im_arr[i,k]=1
            mask[i,k]=0

    im= Image.fromarray(im_arr)
    np.save('both_down.npy', mask)  
    im.save(os.getcwd()+"/editedLuscombei.png")

def left_up_mask(im):
    im_arr = np.array(im)

    mask=np.ones((256,256))

    for i in range(65,95):
        for j in range(0,30):
            im_arr[i,j]=1
            mask[i,j]=0

    im= Image.fromarray(im_arr)
    np.save('left_up.npy', mask)  
    im.save(os.getcwd()+"/editedLuscombei.png")

def both_up_mask(im):
    im_arr = np.array(im)

    mask=np.ones((256,256))

    for i in range(77,95):
        for j in range(6,30):
            im_arr[i,j]=1
            mask[i,j]=0
        for k in range(224,248):
            im_arr[i,k]=1
            mask[i,k]=0

    im= Image.fromarray(im_arr)
    np.save('both_up.npy', mask)  
    im.save(os.getcwd()+"/editedLuscombei.png")

def both_but_one_sided_mask(im):
    im_arr = np.array(im)

    mask=np.ones((256,256))

    for i in range(73,95):
        for j in range(4,30):
            im_arr[i,j]=1
            mask[i,j]=0

    for i in range(145,165):
        for k in range(137,192):
            im_arr[i,k]=1
            mask[i,k]=0

    im= Image.fromarray(im_arr)
    np.save('opposite_oneSided.npy', mask)  
    im.save(os.getcwd()+"/editedLuscombei.png")

def half_mask(im):
    im_arr = np.array(im)

    mask=np.ones((256,256))

    for i in range(0,100):
        for j in range(0,256):
            im_arr[i,j]=1
            mask[i,j]=0

    im= Image.fromarray(im_arr)
    np.save('half.npy', mask)  
    im.save(os.getcwd()+"/whole_edited.png")

im = Image.open('../exp/datasets/color_swatch/whole.png')
#both_but_one_sided_mask(im)
half_mask(im)