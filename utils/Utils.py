from matplotlib import pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Utils:

    @staticmethod
    def plot_image(image, title):
        plt.title(title)
        plt.imshow(image)
        plt.show()
        plt.close()

    @staticmethod
    def save_pickle(data, path):
        with open(f'{path}', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(path):
        return pickle.load(open(path, "rb"))

    @staticmethod
    def get_feature_vector_class(all_features_info, idx):
        positions_class = all_features_info['labels'] == idx
        feature_vector = all_features_info['features'][positions_class]
        return feature_vector

    @staticmethod
    def compute_class_cosine_similarity(features, class_features):
        cos_sim_instances = cosine_similarity(features.reshape(1, -1), class_features)
        cos_sim = np.sum(cos_sim_instances)/cos_sim_instances.shape[1]
        return cos_sim


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count