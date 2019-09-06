import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from gluoncv.model_zoo import get_model


class MyNetwork(gluon.HybridBlock):
    def __init__(self,
                 model_name='ResNet50_v2',
                 pretrained=True,
                 classes=10,
                 **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.finetune_net = get_model(model_name, pretrained=pretrained)
            self.finetune_net.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, fc3_weight):
        return self.finetune_net(x)