################################################################################
# With the data augmentation functions, we can define our data loaders:

import params
import transforms

from mxnet import gluon
import zipfile, os
from gluoncv.utils import download

classes = 23

file_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/minc-2500-tiny.zip'
root_dir = '../data'
zip_file = download(file_url, path=root_dir)
with zipfile.ZipFile(zip_file, 'r') as zin:
    zin.extractall(os.path.expanduser('./'))

path = os.path.join(root_dir,'/minc-2500-tiny')
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')
test_path = os.path.join(path, 'test')

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(transforms.transform_train),
    batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(transforms.transform_test),
    batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(transforms.transform_test),
    batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)