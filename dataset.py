# Import libraries
import torch, torchvision, os
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn; from PIL import Image
from torchvision import transforms as T; from glob import glob
# Set the manual seed
torch.manual_seed(2023)

class CustomDataset(Dataset):

    """

    This class gets several parameters and returns dataset to train an AI model.

    Parameters:

        root             - path to data, str;
        transformations  - transformations to be applied, torchvision object.    
    
    """
    
    def __init__(self, root, transformations = None):

        # Set the transformations
        self.transformations = transformations
        # Get the image paths
        self.im_paths = [im_path for im_path in sorted(glob(f"{root}/*/*")) if "jpg" in im_path]
        # Set the class names, class counts, and other variables
        self.cls_names, self.cls_counts, count, data_count = {}, {}, 0, 0
        
        # Go through the image paths
        for idx, im_path in enumerate(self.im_paths):
            # Get the class name
            class_name = self.get_class(im_path)
            # Add the class name to the dictionary
            if class_name not in self.cls_names: self.cls_names[class_name] = count; self.cls_counts[class_name] = 1; count += 1
            # Increase the count of the class if the class is in the dictionary
            else: self.cls_counts[class_name] += 1
    
    # Function to get the class name based on the image path        
    def get_class(self, path): return os.path.dirname(path).split("/")[-1]

    # Function to get the number of images in the dataset
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        """

        This function gets an index and returns image and gt for the data.

        Parameter:

            idx         - index, int.

        Outputs:

            im          - image, tensor;
            gt          - ground truth label, int.
        
        """
        
        im_path = self.im_paths[idx]
        im = Image.open(im_path)
        gt = self.cls_names[self.get_class(im_path)]
        
        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt

def get_dls(root, transformations, bs, split = [0.8, 0.1, 0.1], ns = 4):
    
    ds = CustomDataset(root = root, transformations = transformations)
    ds_len = len(ds)
    tr_len = int(ds_len * split[0]); val_len = int(ds_len * split[1]); ts_len = ds_len - tr_len - val_len
    
    tr_ds, val_ds, ts_ds = random_split(ds, [tr_len, val_len, ts_len])
    
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, ds.cls_names
