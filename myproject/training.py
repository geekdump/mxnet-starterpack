import params
import data

import mxnet as mx
import numpy as np
import os, time, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model

################################################################################
#
# Note that only ``train_data`` uses ``transform_train``, while
# ``val_data`` and ``test_data`` use ``transform_test`` to produce deterministic
# results for evaluation.
#
# Model and Trainer
# -----------------
#
# We use a pre-trained ``ResNet50_v2`` model, which has balanced accuracy and
# computation cost.

model_name = 'ResNet50_v2'
finetune_net = get_model(model_name, pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(data.classes)
finetune_net.output.initialize(init.Xavier(), ctx = params.ctx)
finetune_net.collect_params().reset_ctx(params.ctx)
finetune_net.hybridize()

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
                        'learning_rate': params.lr, 'momentum': params.momentum, 'wd': params.wd})
metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()

################################################################################
# Here's an illustration of the pre-trained model
# and our newly defined model:
#
# |image-model|
#
# Specifically, we define the new model by::
#
# 1. load the pre-trained model
# 2. re-define the output layer for the new task
# 3. train the network
#
# This is called "fine-tuning", i.e. we have a model trained on another task,
# and we would like to tune it for the dataset we have in hand.
#
# We define a evaluation function for validation and testing.

def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

################################################################################
# Training Loop
# -------------
#
# Following is the main training loop. It is the same as the loop in
# `CIFAR10 <dive_deep_cifar10.html>`__
# and ImageNet.
#
# .. note::
#
#     Once again, in order to go through the tutorial faster, we are training on a small
#     subset of the original ``MINC-2500`` dataset, and for only 5 epochs. By training on the
#     full dataset with 40 epochs, it is expected to get accuracy around 80% on test data.

lr_counter = 0
num_batch = len(data.train_data)

for epoch in range(params.epochs):
    if epoch == params.lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate*params.lr_factor)
        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, batch in enumerate(data.train_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=params.ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=params.ctx, batch_axis=0, even_split=False)
        with ag.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        trainer.step(params.batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch

    _, val_acc = test(finetune_net, data.val_data, params.ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
             (epoch, train_acc, train_loss, val_acc, time.time() - tic))

_, test_acc = test(finetune_net, data.test_data, params.ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))