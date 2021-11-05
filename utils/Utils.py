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
        #cos_sim = np.max(cos_sim_instances)
        return cos_sim

    @staticmethod
    def convertToOneHot(vector, num_classes=None):
        """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """

        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0

        if num_classes is None:
            num_classes = np.max(vector) + 1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)

        result = np.zeros(shape=(len(vector), num_classes), dtype=np.float32)
        result[np.arange(len(vector)), vector] = np.float32(1.0)
        return result


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