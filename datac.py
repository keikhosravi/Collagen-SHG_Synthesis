from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, img_as_float, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython import display
# Ignore warnings
import warnings
import csv
import copy
from skimage import exposure

warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class HE_SHG_Dataset(Dataset):

    def __init__(self, csv_file, transform=None):
     
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        he_name = os.path.join(self.files_list.iloc[idx, 0])
        shg_name = os.path.join(self.files_list.iloc[idx, 1])

        he_image = io.imread(he_name)
        he_image = img_as_float(he_image)
        shg_image= io.imread(shg_name)
        shg_image= img_as_float(shg_image)
        
        sample = {'input': he_image, 'output': shg_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Compose(object):


    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class Rescale(object):


    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        input, output = sample['input'], sample['output']

        h, w = input.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        he_img = transform.resize(input, (new_h, new_w))
        shg_img = transform.resize(output, (new_h, new_w))

  

        return {'input': he_img, 'output': shg_img}


class RandomCrop(object):
    

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        input, output = sample['input'], sample['output']

        h, w = input.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input = input[top: top + new_h,
                      left: left + new_w]

        output = output[top: top + new_h,
                      left: left + new_w]

        return {'input': input, 'output': output}


class ToTensor(object):

    def __call__(self, sample):
        input, output = sample['input'], sample['output']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input = input.transpose((2, 0, 1))
        output = output.transpose((0, 1))

        return {'input': torch.from_numpy(input),
                'output': torch.from_numpy(output)}
    
class Normalize(object):
    

#     def __init__(self, mean, std):
#          self.mean = mean
#          self.std = std

    def __call__(self, sample):
        
        #nparray
        input, output = sample['input'], sample['output']
        
#         gray = color.rgb2gray(input)
#         gray = exposure.equalize_hist(gray)
#         output = np.multiply(gray, output)
#         input[:, :, 0] = output
#         input[:, :, 1] = output
#         input[:, :, 2] = output
        output = output-0.02
        output = np.clip(output, a_min=0, a_max=1)
        output = exposure.adjust_gamma(output, 0.5)

        
        
        return {'input': input, 'output': output}

    
def get_default_image_length():
    return 100
def get_default_input_channels():
    return 3
def get_default_batch_size():
    return 32
def get_default_num_workers():
    return 1

def get_csv_path():
    path = os.path.join(os.getcwd(), 'Data', 'image_files.csv')
    return path

def generate_csv():
    cwd = os.getcwd()
    csvFilePath = os.path.join(cwd, 'Data', 'image_files.csv')
    hePath = os.path.join(cwd, 'Data', 'HE_JPEG')
    shgPath = os.path.join(cwd, 'Data', 'SHG_JPEG')
    numOfPatch = len([name for name in os.listdir(hePath) if 
                      os.path.isfile(os.path.join(hePath, name))])
    with open(csvFilePath, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', 
                                quoting=csv.QUOTE_MINIMAL)
        for i in range(numOfPatch):
            he = os.path.join(hePath, str(i+1) + '.jpeg')
            shg = os.path.join(shgPath, str(i+1) + '.jpeg')
            filewriter.writerow([he, shg])

def unnormalize_img(batch, mean, std):
    for img in batch:
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
    return batch 

def show_patch(dataloader, index = 3):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['input'].size(), 
              sample_batched['output'].size())

        # observe 4th batch and stop.
        if i_batch == index:
            plt.figure()
            input_batch, label_batch = sample_batched['input'], sample_batched['output']
            batch_size = len(input_batch)
            im_size = input_batch.size(2)
            label_batch=label_batch.reshape([batch_size,1,im_size,im_size])
            print(label_batch.size())
#             input_batch = unnormalize_img(input_batch, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             label_batch = unnormalize_img(label_batch, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

            grid = utils.make_grid(input_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.figure()

            grid = utils.make_grid(label_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
        
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)



def match_histograms(image, reference, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : ndarray
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    shape = image.shape
    image_dtype = image.dtype

    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference '
                             'image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched           