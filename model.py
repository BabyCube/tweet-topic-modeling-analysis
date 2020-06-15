# codeing: utf-8

from mxnet import gluon
from mxnet.gluon import HybridBlock


class CNNTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, filters=[3, 4], num_conv_layers=2,
                 num_classes=2, prefix=None, params=None):
        super(CNNTextClassifier, self).__init__(prefix=prefix, params=params)

        with self.name_scope():
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            self.encoder = gluon.nn.Conv1D(channels=16, kernel_size=4, strides=2, activation='relu')
            self.encoded_pool = gluon.nn.MaxPool1D(pool_size=2)
            self.encoder_2 = gluon.nn.Conv1D(channels=32, kernel_size=2, strides=2)
            self.encoded_pool2 = gluon.nn.GlobalMaxPool1D()
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(0.4))
                self.output.add(gluon.nn.Dense(3)) # three categories - neutral, positive or negative sentiment

    def hybrid_forward(self, F, data):
        embedded = self.embedding(data)
        encoded = self.encoder(embedded)
        encoded_pooled = self.encoded_pool(encoded)
        encoded_2 = self.encoder_2(encoded_pooled)
        encoded_pooled_2 = self.encoded_pool2(encoded_2)
        return self.output(encoded_pooled_2)
