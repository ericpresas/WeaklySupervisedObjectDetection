from utils import utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from pandas import DataFrame
import numpy as np

stage = 'train'

ANNOTATIONS_PATH = "data/coco/annotations"
ANN_COCO_PATH = f"{ANNOTATIONS_PATH}/instances_{stage}2017"
FILE_PATH = f"{ANNOTATIONS_PATH}/pseudo_labels_{stage}.pkl"
CATS_PATH = f"{ANNOTATIONS_PATH}/categories.pkl"

if __name__ == "__main__":
    pseudo_labels = utils.load_pickle(FILE_PATH)
    categories = utils.load_pickle(CATS_PATH)

    categories_list = [category['name'] for category in categories]
    map_category = {category['id']:category['name'] for category in categories}

    y_true = []
    y_pred = []
    correct = 0
    for label_info in pseudo_labels:
        pseudo_name = map_category[label_info['pseudo_label_id']]
        y_true.append(pseudo_name)
        if pseudo_name in label_info['categories']:
            y_pred.append(pseudo_name)
            correct += 1
        else:
            y_pred.append(label_info['categories'][0])

    conf_m = confusion_matrix(y_true, y_pred, labels=categories_list)
    #conf_m = conf_m.astype('float') / conf_m.sum(axis=1)[:, np.newaxis]
    df_cm = DataFrame(conf_m, index=categories_list, columns=categories_list)
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
    plt.show()
    print(f"Accuracy: {correct/len(pseudo_labels)}")
    #plt.
