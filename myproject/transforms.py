################################################################################
# Things to keep in mind:
#
# 1. ``epochs = 5`` is just for this tutorial with the tiny dataset. please change it to a larger number in your experiments, for instance 40.
# 2. ``per_device_batch_size`` is also set to a small number. In your experiments you can try larger number like 64.
# 3. remember to tune ``num_gpus`` and ``num_workers`` according to your machine.
# 4. A pre-trained model is already in a pretty good status. So we can start with a small ``lr``.
#
# Data Augmentation
# -----------------
#
# In transfer learning, data augmentation can also help.
# We use the following augmentation in training:
#
# 2. Randomly crop the image and resize it to 224x224
# 3. Randomly flip the image horizontally
# 4. Randomly jitter color and add noise
# 5. Transpose the data from height*width*num_channels to num_channels*height*width, and map values from [0, 255] to [0, 1]
# 6. Normalize with the mean and standard deviation from the ImageNet dataset.
#

from mxnet.gluon.data.vision import transforms


jitter_param = 0.4
lighting_param = 0.1

# Define the transformations applied to the training data.
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the transformations applied to the test data.
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
