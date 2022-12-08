import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import math
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow_confusion_metrics import tf_confusion_metrics
from tensorflow_confusion_metrics import tf_confusion_metrics_2, Macro_calculate_measures_tf, calculate_output
import random
import pandas as pd
import datetime
from sklearn.metrics import auc, roc_curve
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cat = 5


def normalize(data):

    normalized_data = Normalizer(norm='l2').fit_transform(data)

    return normalized_data

def getBatch(list, batchSize):

    try:
        for i in range(0, len(list), batchSize):
            yield list[i:i + batchSize]
    except Exception as E:
        print(E)

def accuracytestNN(sess):

    return tf_confusion_metrics(Pred_AE, y, sess, {x: test_data, y: test_labels})

def accuracytestPL(sess):

    return tf_confusion_metrics(Pred_PL, y, sess, {x: test_data, y: test_labels})

def calc_roc_curve(y_true,y_pred, DN):
    # Compute ROC curve and ROC area for each class
    y_true = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    y_pred = label_binarize(y_pred, classes=[0, 1, 2, 3, 4])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(cat):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
##################################Plot of a ROC curve for a specific class##############################################
    # plt.figure()
    lw = 2
    # plt.plot(fpr[2], tpr[2], color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
#######################################Compute macro-average ROC curve and ROC area#########################################
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(cat)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(cat):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= cat

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'mediumvioletred','lime'])
    for i, color in zip(range(cat), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of '+DN)
    plt.legend(loc="lower right")
    # plt.show()
    # plt.savefig('Fig_ROC'+DN+'.eps', format='eps')

currentDT_1 = datetime.datetime.now()
print(str(currentDT_1))

data = np.genfromtxt(
    "Data/Static_Dynamic/Hybrid_Feature_Vector_flatten.csv", delimiter=",")

print("Number of data before cleaning", len(data))

col_num = data.shape[1]

labels = np.array(data[:, col_num - 1])
labels = labels.astype(int)
print(np.unique(labels))

# dropping the labels' columns
data = data[:, :col_num - 1]

data = normalize(data)

# 1-d array to one-hot conversion
onehot_labels = np.zeros((labels.shape[0], cat))
onehot_labels[np.arange(labels.size), labels - 1] = 1

train_data, test_data, train_labels, test_labels = train_test_split(
    data, onehot_labels, test_size=0.3)

print(type(train_labels))

s = np.array([np.where(r == 1)[0][0] for r in train_labels])
print(np.unique(s))
print("Adware_training=", (s == 0).sum())
print("Banking_training=", (s == 1).sum())
print("SMS_training", (s == 2).sum())
print("Riskware_training=", (s == 3).sum())
print("Benign_training", (s == 4).sum())

print(type(test_labels))
s = np.array([np.where(r == 1)[0][0] for r in test_labels])
# s = s.astype(int)
print("Adware_testing=", (s == 0).sum())
print("Banking_testing=", (s == 1).sum())
print("SMS_testing", (s == 2).sum())
print("Riskware_testing=", (s == 3).sum())
print("Benign_testing", (s == 4).sum())

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

# Neural Network parameters
iteration_list = np.zeros(1)
neural_network_accuracy_list = np.zeros(1)
pseudo_label_accuracy_list = np.zeros(1)
neural_network_cost_list = np.zeros(1)
pseudo_label_cost_list = np.zeros(1)

neural_network_accuracy = 0
pseudo_label_accuracy = 0

learningRate = 0.007
trainingEpochs = 1500
lbl_samples = 8118

# Network Parameters
num_hidden_1 = 450 # 1st layer num features
num_hidden_2 = 300 # 2nd layer num features (the latent dim)
num_hidden_3 = 150 # 3rd layer num features (the latent dim)
num_hidden_4 = 10 # 4th layer num features (the latent dim)
# num_input = 470
# num_input = 497
num_input = 564
outputN = cat
batchSize = 100
num_train_samples = train_data.shape[0]
PLbatchSize = math.ceil(
    ((num_train_samples-lbl_samples)*batchSize)/lbl_samples)

iteration = 0
epoch = 0
cPL = 0
T1 = 100
T2 = 400
a = 0.
af = 1.5

# iteration = 0
# epoch = 0
# cPL = 0
# T1 = 200
# T2 = 800
# a = 0.
# af = 1.5

# print("HiddenLayer1:", hiddenN, "HiddenLayer2:", hiddenN2, "HiddenLayer3:",
#       hiddenN3, "HiddenLayer4:", hiddenN4, "HiddenLayer5:", hiddenN5)

x = tf.placeholder("float", [None, num_input])
y = tf.placeholder("float", [None, outputN])
PLx = tf.placeholder("float", [None, num_input])
PLy = tf.placeholder("float", [None, outputN])
alpha = tf.placeholder("float", )

weightsAE = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], stddev=1.0, mean=0.0)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], stddev=1.0, mean=0.0)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3], stddev=1.0, mean=0.0)),
    'encoder_h4': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_4], stddev=1.0, mean=0.0)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_4, num_hidden_3], stddev=1.0, mean=0.0)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2], stddev=1.0, mean=0.0)),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], stddev=1.0, mean=0.0)),
    'decoder_h4': tf.Variable(tf.random_normal([num_hidden_1, num_input], stddev=1.0, mean=0.0)),
}
biasesAE = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([num_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([num_input])),
}
weightsSoft = {
    'out_layer_h' : tf.Variable(tf.random_normal([num_hidden_4, outputN])),
}
biasesSoft= {
    'out_layer_b' : tf.Variable(tf.random_normal([outputN])),
}
weightsPL = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], stddev=1.0, mean=0.0)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], stddev=1.0, mean=0.0)),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3], stddev=1.0, mean=0.0)),
    'encoder_h4': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_4], stddev=1.0, mean=0.0)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_4, num_hidden_3], stddev=1.0, mean=0.0)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2], stddev=1.0, mean=0.0)),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], stddev=1.0, mean=0.0)),
    'decoder_h4': tf.Variable(tf.random_normal([num_hidden_1, num_input], stddev=1.0, mean=0.0)),
}
biasesPL = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([num_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([num_input])),
}
weightsSoft_PL = {
    'out_layer_h' : tf.Variable(tf.random_normal([num_hidden_4, outputN])),
}
biasesSoft_PL= {
    'out_layer_b' : tf.Variable(tf.random_normal([outputN])),
}
# Building the encoder
def encoder(x, weights, biases):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                   biases['encoder_b4']))
    return layer_4
# Building the decoder
def decoder(x, weights, biases):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4= tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4
def logits(x, weights, biases):
    out_logits = tf.add(tf.matmul(x, weights['out_layer_h']),
           biases['out_layer_b'])
    return out_logits

currentDT_2 = datetime.datetime.now()
print(str(currentDT_2))

dateTimeDifference = currentDT_2 - currentDT_1

print('Run Time in Seconds_preprocessing: ',dateTimeDifference.total_seconds())
######################################Pre-training###################################################################
currentDT_1 = datetime.datetime.now()
print(str(currentDT_1))

encoder_op_Pre = encoder(x, weightsAE, biasesAE)
decoder_op_Pre = decoder(encoder_op_Pre, weightsAE, biasesAE)

costPre = tf.reduce_mean(tf.square(decoder_op_Pre - x))
optimizerPre = tf.train.RMSPropOptimizer(learningRate).minimize(costPre)

encoder_op_Pre_PL = encoder(x, weightsPL, biasesPL)
decoder_op_Pre_PL = decoder(encoder_op_Pre_PL, weightsPL, biasesPL)

costPre_PL = tf.reduce_mean(tf.square(decoder_op_Pre_PL - x))
optimizerPre_PL = tf.train.RMSPropOptimizer(learningRate).minimize(costPre_PL)

init = tf.global_variables_initializer()
pretrain_saver = tf.train.Saver([weightsAE['encoder_h1'], weightsAE['encoder_h2'], weightsAE['encoder_h3'],weightsAE['encoder_h4'],
                                 biasesAE['encoder_b1'], biasesAE['encoder_b2'], biasesAE['decoder_b3'],biasesAE['decoder_b4'],
                                 weightsPL['encoder_h1'], weightsPL['encoder_h2'], weightsPL['encoder_h3'],weightsPL['encoder_h4'],
                                 biasesPL['encoder_b1'], biasesPL['encoder_b2'], biasesPL['decoder_b3'], biasesPL['decoder_b4']])
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(trainingEpochs):
        n_batches = lbl_samples // batchSize
        c = list(zip(train_data, train_labels))
        random.shuffle(c)

        train_data, train_labels = zip(*c)

        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)

        data_batches = list(getBatch(train_data[0:lbl_samples, :], batchSize,))
        labels_batches = list(
            getBatch(train_labels[0:lbl_samples, :], batchSize))
        for iter in range(n_batches):
            print("\r{}%".format(100 * iter // n_batches), end="")
            batch_x = data_batches[iter]
            sess.run(optimizerPre, feed_dict={x: batch_x})
            sess.run(optimizerPre_PL, feed_dict={x: batch_x})
        # accuracy_val = accuracy.eval(feed_dict={x: batch_x})
        # accuracy_val_Pre = accuracyPL.eval(feed_dict={x: batch_x})
        # print("\r{}".format(epoch), "Train accuracy:", accuracy_val,accuracy_val_Pre, end=" ")

        yy = sess.run(decoder_op_Pre, feed_dict={x: batch_x})
        xx = sess.run(encoder_op_Pre, feed_dict={x: batch_x})
        saver.save(sess, "./my_model_supervised.ckpt")
        # accuracy_val = accuracy.eval(feed_dict={x: test_data, y: test_labels})
        # print("Test accuracy:", accuracy_val)

currentDT_2 = datetime.datetime.now()
print(str(currentDT_2))

dateTimeDifference = currentDT_2 - currentDT_1

print('Run Time in Seconds_pre_training: ',dateTimeDifference.total_seconds())
#######################################Fine-Tuning#####################################################################
currentDT_1 = datetime.datetime.now()
print(str(currentDT_1))
encoder_op_AE = encoder(x, weightsAE, biasesAE)
Pred_AE = logits(encoder_op_AE, weightsSoft, biasesSoft)
encoder_op_PL = encoder(x, weightsPL, biasesPL)
Pred_PL = logits(encoder_op_PL, weightsSoft_PL, biasesSoft_PL)
encoder_op_PL1 = encoder(PLx, weightsPL, biasesPL)
Pred_PL1 = logits(encoder_op_PL1, weightsSoft_PL, biasesSoft_PL)

costAE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_AE,
                                                                labels=y))
costPL = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_PL,
                                                                       labels=y)),
                (alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_PL1,
                                                                                labels=PLy))))
# Gradient Descent
optimizerAE = tf.train.RMSPropOptimizer(learningRate).minimize(costAE)
optimizerPL = tf.train.RMSPropOptimizer(learningRate).minimize(costPL)

# Initializing the variables
init = tf.global_variables_initializer()

epoch = 0
with tf.Session() as sess:
    sess.run(init)
    pretrain_saver.restore(sess, "./my_model_supervised.ckpt")
    # Training cycle
    avg_costNN = 1.
    avg_costPL = 1.
    while (avg_costNN > 0.05 and epoch < trainingEpochs):

        avg_costNN = 0.
        avg_costPL = 0.

        total_batch = int(lbl_samples / batchSize)

        c = list(zip(train_data, train_labels))
        random.shuffle(c)

        train_data, train_labels = zip(*c)

        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)

        data_batches = list(getBatch(train_data[0:lbl_samples, :], batchSize,))
        labels_batches = list(
            getBatch(train_labels[0:lbl_samples, :], batchSize))

        PLdata_batches = list(
            getBatch(train_data[lbl_samples:, :], PLbatchSize))
        PLlabels_batches = list(
            getBatch(train_labels[lbl_samples:, :], PLbatchSize))

        # Loop over all batches
        try:
            for i in range(total_batch):

                batch_x = data_batches[i]
                batch_y = labels_batches[i]

                _, cNN = sess.run([optimizerAE, costAE], feed_dict={x: batch_x,
                                                                    y: batch_y})

                # implementation of alpha calculation formula
                if iteration >= T1 and iteration<=T2:
                    a = ((iteration - T1) / (T2 - T1)) * af
                if iteration >= T2:
                    a = af

                batch_xpred = PLdata_batches[i]
                batch_ypred = sess.run([Pred_PL], feed_dict={x: batch_xpred})
                batch_ypred = batch_ypred[0]
                batch_ypred = batch_ypred.argmax(1)
                batch_ypre = np.zeros((len(batch_ypred), cat))
                for ii in range(len(batch_ypred)):
                    batch_ypre[ii, batch_ypred[ii]] = 1

                _, cPL = sess.run([optimizerPL, costPL], feed_dict={x: batch_x,
                                                                    y: batch_y,
                                                                    PLx: batch_xpred,
                                                                    PLy: batch_ypre,
                                                                    alpha: a})
                iteration = iteration + 1
                # Compute average loss
                avg_costNN += cNN
                avg_costPL += cPL

        except Exception as E:
            print(E)
        avg_costNN += avg_costNN / total_batch
        avg_costPL += avg_costPL / total_batch

        if iteration % 100 == 0:
            print('t=', iteration)
            neural_network_accuracy = accuracytestNN(sess)

            pseudo_label_accuarcy = accuracytestPL(sess)

            iteration_list = np.append(iteration_list, iteration)

            neural_network_cost_list = np.append(
                neural_network_cost_list, avg_costNN)
            pseudo_label_cost_list = np.append(
                pseudo_label_cost_list, avg_costPL)

        epoch += 1

    x_NN = accuracytestNN(sess)
    x_PL = accuracytestPL(sess)

    conf = tf_confusion_metrics_2(Pred_PL, y, sess, {x: test_data, y: test_labels})
    Macro_calculate_measures_tf(Pred_PL, y, sess, {x: test_data, y: test_labels})
    sum_conf = np.sum(conf, axis=1)

    lst = []
    for i in range(len(sum_conf)):
        lst.append(np.round((conf[i, :] / sum_conf[i]), 2))

    arr = np.array(lst)

    print("Optimization Finished!")

    print("Confusion Matrix:")
    print(conf)
    print(arr)
    print(sum_conf)
    print("NN:", x_NN)
    print("+PL:", x_PL)

    pd.concat([x_NN, x_PL], axis=1).to_csv('test_5_layers.csv')
    y_true, y_pred = calculate_output(Pred_PL, y, sess, {x: test_data, y: test_labels})
    calc_roc_curve(y_true, y_pred, 'PLSAE')
    y_true, y_pred = calculate_output(Pred_AE, y, sess, {x: test_data, y: test_labels})
    calc_roc_curve(y_true, y_pred, 'SAE')

    #########################Cost Plot############################

    currentDT_2 = datetime.datetime.now()
    print(str(currentDT_2))

    dateTimeDifference = currentDT_2 - currentDT_1

    print('Run Time in Seconds_fine_tuning: ',dateTimeDifference.total_seconds())
    plt.figure()
    plt.plot(iteration_list, pseudo_label_cost_list, 'r--', label='PLSAE')
    plt.plot(iteration_list, neural_network_cost_list, 'b--', label='SAE')

    plt.legend(loc='upper left')
    plt.xlabel("Iterations")
    plt.ylabel("Average Training Cost")
    plt.ylim(0,26)
    plt.xlim(0,16000)
    # plt.show()
    # plt.savefig('Fig_avg_cost.eps', format='eps')
    # roc_curve(sess)
