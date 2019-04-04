import numpy as np
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics


# attentions = list of attention arrays
# correct = list of correct word indexes
def attention_metric(attentions, correct):
    correct_predictions = 0
    for i in range(len(correct)):
        if np.argmax(attentions[i]) == correct[i]:
            correct_predictions += 1

    return correct_predictions / len(correct)

# vectors = list of arrays shape (num_vectors_in_class, vector_len) with length num_classes 
def centroid_metric(vectors, metric='euclidean'):
    centroids = []
    for vector_class in vectors:
        centroids.append(np.average(vector_class, axis=0))
    centroids = np.array(centroids) # (num_classes, vector_len)
    distances = distance.cdist(centroids, centroids, metric)
    return np.average(distances) # not really average distance but other properties are ok

# input_type = 0 => vectors = list of arrays shape (num_vectors_in_class, vector_len) with length num_classes 
# input_type = 1 => vectors = array (num_vectors, vector_len)
def cluster_metric(vectors, y=None, input_type=0):
    if input_type == 0:
        X = vectors[0]
        y = np.zeros(len(vectors[0])) # len(x) == x.shape[0] if x is np.array

        for idx, vec_class in enumerate(vectors[1:]):
            X = np.concatenate((X, vec_class))
            y = np.concatenate((y, np.ones(len(vec_class)) * (idx + 1)))
    else:
        X = vectors

    clustering = KMeans(n_clusters=len(np.unique(y))).fit(X)
    labels = clustering.labels_
    print(labels)
    homogeneity = metrics.homogeneity_score(y, labels)
    completeness = metrics.completeness_score(y, labels)
    v_measure = metrics.v_measure_score(y, labels)
    return homogeneity, completeness, v_measure

# vectors = list of arrays shape (num_vectors_in_class, vector_len) with length num_classes 
# input_type = 1 => vectors = array (num_vectors, vector_len), y = array (num_vectors) of labels
def classifier_metric(vectors, y=None, val_vectors=None, val_y=None, input_type=0):
    clf = LogisticRegression(max_iter=500)
    if input_type == 0:
        X = vectors[0]
        y = np.zeros(len(vectors[0])) # len(x) == x.shape[0] if x is np.array

        for idx, vec_class in enumerate(vectors[1:]):
            X = np.concatenate((X, vec_class))
            y = np.concatenate((y, np.ones(len(vec_class)) * (idx + 1)))

        if val_vectors == None:
            val_vectors = vectors
            val_X = X
            val_y = y
        else:
            val_X = val_vectors[0]
            val_y = np.zeros(len(val_vectors[0])) # len(x) == x.shape[0] if x is np.array
            for idx, vec_class in enumerate(val_vectors[1:]):
                val_X = np.concatenate((val_X, vec_class))
                val_y = np.concatenate((val_y, np.ones(len(vec_class)) * (idx + 1)))
    else:
        X = vectors
        if val_vectors == None:
            val_X = X
            val_y = y
        else:
            val_X = val_vectors

    clf.fit(X, y)
    return clf.score(val_X, val_y)

# vectors = list of arrays shape (num_vectors_in_class, vector_len) with length num_classes 
def distance_metrics(vectors, metric='euclidean'):
    distances = []
    for idx, vecs1 in enumerate(vectors):
        for vecs2 in vectors[idx+1:]:
            distances += list(distance.cdist(vecs1, vecs2, metric).reshape(-1))
    distances = np.array(distances)
    return np.min(distances), np.max(distances), np.average(distances)