import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def createBerlinReport(net, data, device):
    berlin_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 'Allotment', 'Commercial Area', 'Water']

    print("Berlin Start!")
    return createReport(net, data, berlin_class_names, device)
    

def createReport(net, data, class_names, device):
    global cate
    net.eval()
    count = 0
    for hsi, x, hsi_pca, test_labels,h,w in data:
        hsi=hsi.cuda(device)
        hsi_pca = hsi_pca.to(device)
        x = x.to(device)
        _ , outputs = net(hsi_pca.unsqueeze(1), x)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred = outputs
            y_true = test_labels
            count = 1
        else:
            y_pred = np.concatenate((y_pred, outputs))
            y_true = np.concatenate((y_true, test_labels))

    classification = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)
    confusion = confusion_matrix(y_true, y_pred)
    oa = accuracy_score(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_true, y_pred)

    classification = str(classification)
    confusion = str(confusion)
    oa = oa * 100
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = kappa * 100

    logging.info(f'\n{classification}')
    logging.info(f'Overall accuracy (%) {oa}')
    logging.info(f'Average accuracy (%) {aa}')
    logging.info(f'Kappa accuracy (%){kappa}')
    logging.info(f'\n{confusion}')
    
    return oa,aa,kappa,each_acc