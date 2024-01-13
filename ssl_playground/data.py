# import os
# from os import makedirs
# from os.path import exists, join
# from itertools import groupby
# import struct
# import random
# from tqdm import tqdm
# import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import zeros_like
from torchvision.transforms import ToTensor, Pad, Compose, RandomAffine, Lambda
from torchvision.transforms.functional import affine
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class MNIST64:

    transform = Compose([ToTensor(), Pad(18)])
    one_hot_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    rotation_range = [-180, 180]
    translate_range = [0.4, 0.4]
    scale_range = [0.5, 2.0]

    def __init__(self, mnist_path):
        self.data_dir = mnist_path
        return
    
    def make_data(self, one_hot_labels=False):
        if one_hot_labels:
            self.train_data = MNIST(self.data_dir, train=True, transform=MNIST64.transform, 
                                    target_transform=MNIST64.one_hot_transform, 
                                    download=True)
            self.test_data = MNIST(self.data_dir, train=False, transform=MNIST64.transform, 
                                   target_transform=MNIST64.one_hot_transform, 
                                   download=True)
        else:
            self.train_data = MNIST(self.data_dir, train=True, transform=MNIST64.transform, download=True)
            self.test_data = MNIST(self.data_dir, train=False, transform=MNIST64.transform, download=True)

    def make_loaders(self, batch_size):
        self.train_data_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_data, batch_size=batch_size)

    def make_data_and_loaders(self, batch_size, one_hot_labels=False):
        self.make_data(one_hot_labels=one_hot_labels)
        self.make_loaders(batch_size)

    def transform_images_uniform(images, angle, translate, scale):
        return affine(images, angle, translate, scale, 0)
    
    def transform_images(images, angles, translates, scales):
        assert images.size(0) == len(angles) == len(translates[0]) == len(scales), "Transform parameters must match batch size!"
        output = zeros_like(images)
        for i in range(images.size(0)):
            output[i] = affine(images[i], angles[i], [translates[0][i], translates[1][i]], scales[i], 0)
        return output
    
    def random_transform_images_uniform(images):
        angle, translate, scale, _ = RandomAffine.get_params(degrees=MNIST64.rotation_range, translate=MNIST64.translate_range, scale_ranges=MNIST64.scale_range, shears=[0.0, 0.0], img_size=[64, 64])
        return MNIST64.transform_images_uniform(images, angle, translate, scale), (angle, translate, scale)
    
    def random_transform_images(images):
        angles = np.random.uniform(MNIST64.rotation_range[0], MNIST64.rotation_range[1], size=images.size(0))
        translates = np.stack([np.random.uniform(-64*MNIST64.translate_range[0], 64*MNIST64.translate_range[0], size=images.size(0)), 
                              np.random.uniform(-64*MNIST64.translate_range[1], 64*MNIST64.translate_range[1], size=images.size(0))], axis=0)
        scales = np.random.uniform(MNIST64.scale_range[0], MNIST64.scale_range[1], size=images.size(0)) #TODO: Make this logarithmic!
        return MNIST64.transform_images(images, angles.tolist(), translates.tolist(), scales.tolist()), (angles, translates, scales)
    
    def random_transform_pairs_uniform(images):
        images_1, (angle_1, translate_1, scale_1) = MNIST64.random_transform_images_uniform(images)
        images_2, (angle_2, translate_2, scale_2) = MNIST64.random_transform_images_uniform(images)
        return images_1, images_2, (angle_2 - angle_1, (translate_2[0] - translate_1[0], translate_2[1] - translate_1[1]), scale_2 - scale_1)
    
    def random_transform_pairs(images):
        images_1, params_1 = MNIST64.random_transform_images(images)
        images_2, params_2 = MNIST64.random_transform_images(images)
        return images_1, images_2, (params_2[0] - params_1[0], params_2[1] - params_1[1], params_2[2] - params_1[2])

    def show_image(img, label=None):
        if label is not None:
            print(label)
        plt.matshow(img, cmap="gray")
        plt.show()


# # Code for SmallNORBExample and SmallNORBDataset taken with minor modifications from
# # https://github.com/ndrplz/small_norb
# class SmallNORBExample:

#     def __init__(self):
#         self.image_lt  = None
#         self.image_rt  = None
#         self.category  = None
#         self.instance  = None
#         self.elevation = None
#         self.azimuth   = None
#         self.lighting  = None

#     def __lt__(self, other):
#         return self.category < other.category or \
#                 (self.category == other.category and self.instance < other.instance)

#     def show(self, subplots, category_label, elevation_, azimuth_):
#         fig, axes = subplots
#         fig.suptitle(
#             f"Category: {category_label} - Instance: {self.instance:02d} - Elevation: {elevation_:02d} - Azimuth: {azimuth_:03d} - Lighting: {self.lighting:02d}")
#         axes[0].imshow(self.image_lt, cmap='gray')
#         axes[1].imshow(self.image_rt, cmap='gray')

#     @property
#     def pose(self):
#         return np.array([self.elevation, self.azimuth, self.lighting], dtype=np.float32)

# class SmallNORBDataset:

#     # Number of examples in both train and test set
#     n_examples = 24300
#     n_categories = 5
#     n_instances = 10
#     n_elevations = 9
#     n_azimuths = 18
#     n_lightings = 5

#     # Categories present in small NORB dataset
#     categories = ['animal', 'human', 'airplane', 'truck', 'car']

#     elevations = np.arange(30, 75, 5)
#     azimuths = 20*np.arange(18)

#     def __init__(self, dataset_root):
#         """
#         Initialize small NORB dataset wrapper
        
#         Parameters
#         ----------
#         dataset_root: str
#             Path to directory where small NORB archives have been extracted.
#         """

#         self.dataset_root = dataset_root
#         self.initialized  = False

#         # Store path for each file in small NORB dataset (for compatibility the original filename is kept)
#         self.dataset_files = {
#             'train': {
#                 'cat':  join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'),
#                 'info': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat'),
#                 'dat':  join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
#             },
#             'test':  {
#                 'cat':  join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'),
#                 'info': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat'),
#                 'dat':  join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
#             }
#         }

#         # Initialize both train and test data structures
#         self.data = {
#             'train': [SmallNORBExample() for _ in range(SmallNORBDataset.n_examples)],
#             'test':  [SmallNORBExample() for _ in range(SmallNORBDataset.n_examples)]
#         }

#         # Fill data structures parsing dataset binary files
#         for data_split in ['train', 'test']:
#             self._fill_data_structures(data_split)

#         self.initialized = True

#     def explore_random_examples(self, dataset_split):
#         """
#         Visualize random examples for dataset exploration purposes
        
#         Parameters
#         ----------
#         dataset_split: str
#             Dataset split, can be either 'train' or 'test'

#         Returns
#         -------
#         None
#         """
#         if self.initialized:
#             subplots = plt.subplots(nrows=1, ncols=2)
#             i = random.choice(range(SmallNORBDataset.n_examples))
#             example = self.data[dataset_split][i]
#             example.show(subplots, SmallNORBDataset.categories[example.category],
#                          SmallNORBDataset.elevations[example.elevation], 
#                          SmallNORBDataset.azimuths[example.azimuth//2])

#     def export_to_jpg(self, export_dir):
#         """
#         Export all dataset images to `export_dir` directory
        
#         Parameters
#         ----------
#         export_dir: str
#             Path to export directory (which is created if nonexistent)
            
#         Returns
#         -------
#         None
#         """
#         if self.initialized:
#             for split_name in ['train', 'test']:

#                 split_dir = join(export_dir, split_name)
#                 if not exists(split_dir):
#                     makedirs(split_dir)

#                 for i, norb_example in tqdm(iterable=enumerate(self.data[split_name]),
#                                             total=len(self.data[split_name]),
#                                             desc='Exporting {} images to {}'.format(split_name, export_dir)):

#                     category = SmallNORBDataset.categories[norb_example.category]
#                     instance = norb_example.instance

#                     image_lt_path = join(split_dir, '{:06d}_{}_{:02d}_lt.jpg'.format(i, category, instance))
#                     image_rt_path = join(split_dir, '{:06d}_{}_{:02d}_rt.jpg'.format(i, category, instance))

#                     imageio.imwrite(image_lt_path, norb_example.image_lt)
#                     imageio.imwrite(image_rt_path, norb_example.image_rt)

#     def group_dataset_by_category_and_instance(self, dataset_split):
#         """
#         Group small NORB dataset for (category, instance) key
        
#         Parameters
#         ----------
#         dataset_split: str
#             Dataset split, can be either 'train' or 'test'

#         Returns
#         -------
#         groups: list
#             List of 25 groups of 972 elements each. All examples of each group are
#             from the same category and instance
#         """
#         if dataset_split not in ['train', 'test']:
#             raise ValueError('Dataset split "{}" not allowed.'.format(dataset_split))

#         groups = []
#         for key, group in groupby(iterable=sorted(self.data[dataset_split]),
#                                   key=lambda x: (x.category, x.instance)):
#             groups.append(list(group))

#         return groups

#     def _fill_data_structures(self, dataset_split):
#         """
#         Fill SmallNORBDataset data structures for a certain `dataset_split`.
        
#         This means all images, category and additional information are loaded from binary
#         files of the current split.
        
#         Parameters
#         ----------
#         dataset_split: str
#             Dataset split, can be either 'train' or 'test'

#         Returns
#         -------
#         None

#         """
#         dat_data  = self._parse_NORB_dat_file(self.dataset_files[dataset_split]['dat'])
#         cat_data  = self._parse_NORB_cat_file(self.dataset_files[dataset_split]['cat'])
#         info_data = self._parse_NORB_info_file(self.dataset_files[dataset_split]['info'])
#         for i, small_norb_example in enumerate(self.data[dataset_split]):
#             small_norb_example.image_lt   = dat_data[2 * i]
#             small_norb_example.image_rt   = dat_data[2 * i + 1]
#             small_norb_example.category  = cat_data[i]
#             small_norb_example.instance  = info_data[i][0]
#             small_norb_example.elevation = info_data[i][1]
#             small_norb_example.azimuth   = info_data[i][2]
#             small_norb_example.lighting  = info_data[i][3]

#     @staticmethod
#     def matrix_type_from_magic(magic_number):
#         """
#         Get matrix data type from magic number
#         See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.

#         Parameters
#         ----------
#         magic_number: tuple
#             First 4 bytes read from small NORB files 

#         Returns
#         -------
#         element type of the matrix
#         """
#         convention = {'1E3D4C51': 'single precision matrix',
#                       '1E3D4C52': 'packed matrix',
#                       '1E3D4C53': 'double precision matrix',
#                       '1E3D4C54': 'integer matrix',
#                       '1E3D4C55': 'byte matrix',
#                       '1E3D4C56': 'short matrix'}
#         magic_str = bytearray(reversed(magic_number)).hex().upper()
#         return convention[magic_str]

#     @staticmethod
#     def _parse_small_NORB_header(file_pointer):
#         """
#         Parse header of small NORB binary file
        
#         Parameters
#         ----------
#         file_pointer: BufferedReader
#             File pointer just opened in a small NORB binary file

#         Returns
#         -------
#         file_header_data: dict
#             Dictionary containing header information
#         """
#         # Read magic number
#         magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

#         # Read dimensions
#         dimensions = []
#         num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
#         for _ in range(num_dims):
#             dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

#         file_header_data = {'magic_number': magic,
#                             'matrix_type': SmallNORBDataset.matrix_type_from_magic(magic),
#                             'dimensions': dimensions}
#         return file_header_data

#     @staticmethod
#     def _parse_NORB_cat_file(file_path):
#         """
#         Parse small NORB category file
        
#         Parameters
#         ----------
#         file_path: str
#             Path of the small NORB `*-cat.mat` file

#         Returns
#         -------
#         examples: ndarray
#             Ndarray of shape (24300,) containing the category of each example
#         """
#         with open(file_path, mode='rb') as f:
#             header = SmallNORBDataset._parse_small_NORB_header(f)

#             num_examples, = header['dimensions']

#             struct.unpack('<BBBB', f.read(4))  # ignore this integer
#             struct.unpack('<BBBB', f.read(4))  # ignore this integer

#             examples = np.zeros(shape=num_examples, dtype=np.int32)
#             for i in tqdm(range(num_examples), desc='Loading categories...'):
#                 category, = struct.unpack('<i', f.read(4))
#                 examples[i] = category

#             return examples

#     @staticmethod
#     def _parse_NORB_dat_file(file_path):
#         """
#         Parse small NORB data file

#         Parameters
#         ----------
#         file_path: str
#             Path of the small NORB `*-dat.mat` file

#         Returns
#         -------
#         examples: ndarray
#             Ndarray of shape (48600, 96, 96) containing images couples. Each image couple
#             is stored in position [i, :, :] and [i+1, :, :]
#         """
#         with open(file_path, mode='rb') as f:

#             header = SmallNORBDataset._parse_small_NORB_header(f)

#             num_examples, channels, height, width = header['dimensions']

#             examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

#             for i in tqdm(range(num_examples * channels), desc='Loading images...'):

#                 # Read raw image data and restore shape as appropriate
#                 image = struct.unpack('<' + height * width * 'B', f.read(height * width))
#                 image = np.uint8(np.reshape(image, newshape=(height, width)))

#                 examples[i] = image

#         return examples

#     @staticmethod
#     def _parse_NORB_info_file(file_path):
#         """
#         Parse small NORB information file

#         Parameters
#         ----------
#         file_path: str
#             Path of the small NORB `*-info.mat` file

#         Returns
#         -------
#         examples: ndarray
#             Ndarray of shape (24300,4) containing the additional info of each example.
            
#              - column 1: the instance in the category (0 to 9)
#              - column 2: the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 
#                degrees from the horizontal respectively)
#              - column 3: the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
#              - column 4: the lighting condition (0 to 5)
#         """
#         with open(file_path, mode='rb') as f:

#             header = SmallNORBDataset._parse_small_NORB_header(f)

#             struct.unpack('<BBBB', f.read(4))  # ignore this integer

#             num_examples, num_info = header['dimensions']

#             examples = np.zeros(shape=(num_examples, num_info), dtype=np.int32)

#             for r in tqdm(range(num_examples), desc='Loading info...'):
#                 for c in range(num_info):
#                     info, = struct.unpack('<i', f.read(4))
#                     examples[r, c] = info

#         return examples

# class SmallNORB:

#     def __init__(self, norb_path):
#         self.data_dir = norb_path

#         self.file_links = ["https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
#                   "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
#                   "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
#                   "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
#                   "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
#                   "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz"]

#         file_link_prefix = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
#         self.file_names = [f.replace(file_link_prefix, "") for f in self.file_links]
#         self.dataset = None

#     def _download_small_norb(self):
#         for l in self.file_links:
#             os.system(f"wget -P {self.data_dir} {l}")

#     def _unzip_small_norb(self):
#         for n in self.file_names:
#             os.system(f"gunzip {self.data_dir}/{n}")

#     def _load_data_from_mat(self):
#         self.dataset = SmallNORBDataset(self.data_dir)
    
#     def create_small_norb(self, download=True, unzip=True, extract=True):
#         if download: self._download_small_norb()
#         if unzip: self._unzip_small_norb()
#         if extract: self._load_data_from_mat()