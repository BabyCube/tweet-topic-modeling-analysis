import argparse
import load_data_classifier as ld
import model as md

import gluonnlp as nlp
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon.data import DataLoader
import sklearn.metrics as mt
import matplotlib.pyplot as plt
"""
this file will use train a text classifier and classify text to either positive or negative sentiment

the ground truth of positive or negative is based on sentiment score generated;
threshold can be adjusted based on input within argument parser
"""

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--path', type=str,
                    help='File path to COVID-19 tweets JSON with sentiment socre',
                    default='hash_tag_covid_processed.json')
parser.add_argument('--lower_bound_to_be_neutral', type=float,
                    help='The lower bound for sentiment score to be considered neutral; if lower, negative sentiment',
                    default=-0.4)
parser.add_argument('--upper_bound_to_be_neutral', type=float,
                    help='The upper bound for sentiment score to be considered neutral; if larger, positive sentiment',
                    default=0.4)
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=16)

args = parser.parse_args()


loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
def train_classifier(vocabulary, data_train, data_val, data_test, ctx=mx.cpu()):
    ## set up the data loaders for each data source
    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=True)
    test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)

    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape
    model = md.CNNTextClassifier(emb_input_dim, emb_output_dim)

    model.initialize(ctx=ctx)  ## initialize model parameters on the context ctx
    model.embedding.weight.set_data(
        vocab.embedding.idx_to_vec)  ## set the embedding layer parameters to the pre-trained embedding in the vocabulary

    model.hybridize()  ## OPTIONAL for efficiency - perhaps easier to comment this out during debugging

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    for epoch in range(args.epochs):
        print("- - - - - \n")
        print("this is the ", epoch, "th training loop")
        epoch_cum_loss = 0
        for i, (label, data) in enumerate(train_dataloader):
            # print (i, data, label)
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = model(data)  ## should have shape (batch_size,)
                l = loss_fn(output, label).mean()  # get the average loss over the batch
            l.backward()
            trainer.step(1)  ## update weights
            epoch_cum_loss += l.asscalar()  ## needed to convert mx.nd.array value back to Python float
        val_accuracy = evaluate(model, test_dataloader)
        ## display and/or collect validation accuracies after each training epoch
        print("accuracy", val_accuracy)


def evaluate(model, dataloader, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    acc = 0
    total_correct = 0
    total = 0
    labels = []  # store the ground truth labels
    scores = []  # store the predictions/scores from the model
    confidence = []  # store the higher probability of prediction from the model
    for i, (label, data) in enumerate(dataloader):
        out = model(data)
        predictionArray = mx.nd.softmax(out)
        predictionArray_ = predictionArray.asnumpy()

        for row in predictionArray_:
            row = row.tolist()
            confidence.append(row[1])  # append the probability estimate of the positive class
            if row.index(max(row)) == 0:
                scores.append(-1)
            elif row.index(max(row)) == 1:
                scores.append(0)
            else:
                scores.append(1)

        for j in range(out.shape[0]):  ## out.shape[0] refers to the batch size
            lab = int(label[j].asscalar())
            labels.append(lab)
            ## .. gather predictions for each item here
    for i, k in enumerate(scores):
        total += 1
        if labels[i] == scores[i]:
            total_correct += 1  ## if correct
    print("here is the total", total)
    print("here is the total correct", total_correct)

    acc = mt.accuracy_score(labels, scores)
    # acc = total_correct / float(total)
    return acc


if __name__ == '__main__':

    tweets_file = args.path
    upper = args.upper_bound_to_be_neutral
    lower = args.lower_bound_to_be_neutral

    vocab, train_dataset, val_dataset, test_dataset = ld.load_dataset(tweets_file, upper, lower)

    glove_twitter = nlp.embedding.create('glove', source='glove.twitter.27B.50d')
    vocab.set_embedding(glove_twitter)
    #
    ctx = mx.cpu()  ## or mx.gpu(N) if GPU device N is available
    #
    train_classifier(vocab, train_dataset, val_dataset, test_dataset, ctx)