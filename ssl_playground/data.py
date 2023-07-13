import os
from torchvision.transforms import ToTensor, Pad, Compose
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class MNIST64:

    transform = transform = Compose([ToTensor(), Pad(18)])

    def __init__(self, mnist_path):
        self.data_dir = mnist_path
        return
    
    def make_data(self):
        train_data = MNIST(self.data_dir, train=True, transform=self.transform, download=True)
        test_data = MNIST(self.data_dir, train=False, transform=self.transform, download=True)
        return train_data, test_data

    def make_loaders(self, train_data, test_data, batch_size):
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_data, batch_size=batch_size)
        return train_data_loader, test_data_loader

class SmallNORB:

    def __init__(self, norb_path):
        self.data_dir = norb_path
        self.file_links = ["https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
                  "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
                  "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
                  "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
                  "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
                  "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz"]

        file_link_prefix = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
        self.file_names = [f.replace(file_link_prefix, "") for f in self.file_links]
        file_name_prefix = "smallnorb-5x46789x9x18x6x2x96x96-"
        self.stripped_file_names = [f[len(file_name_prefix):] for f in self.file_names]

    def _download_small_norb(self):
        for l in self.file_links:
            os.system(f"wget -P {self.data_dir} {l}")

    def _unzip_small_norb(self):
        for n in self.file_names:
            os.system(f"gunzip {self.data_dir}/{n}")

    def _rename_small_norb(self):
        for n, sn in zip(self.file_names, self.stripped_file_names):
            os.system(f"mv {self.data_dir}/{n[:-3]} {self.data_dir}/{sn[:-3]}")
    
    def create_small_norb(self, download=True, unzip=True, rename=True):
        if download: self._download_small_norb()
        if unzip: self._unzip_small_norb()
        if rename: self._rename_small_norb()