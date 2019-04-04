import numpy as np
import subprocess
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../supervisedPCA-Python')
from supervised_pca import SupervisedPCARegressor
from metrics import *
import csv

labels = []
with open('100_texts_tagged.csv', 'r') as f:
    with open('texts', 'w') as dest:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # if i == 52:
            #     break
            labels.append(int(row[0]))
            dest.write(row[1] + '\n')


bert = '../cased_L-12_H-768_A-12/'
layers = '0,1,2,3,4,5,6,7,8,9,10,11'

args = ['python', '../bert/extract_features.py']
args.append('--input_file=texts')
args.append('--output_file=sentiment')
args.append('--vocab_file=' + bert + 'vocab.txt')
args.append('--bert_config_file=' + bert + 'bert_config.json')
args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
args.append('--layers=' + layers)
args.append('--max_seq_length=128')
args.append('--batch_size=8')
args.append('--do_lower_case=False')
args.append('--attention=False')
args.append('--mask_underscore=False')
subprocess.run(args)




layer_cls_metrics = []
layer_v_metrics = []
for layer in range(12):
    layer_outputs = np.load('sentiment_layer_' + str(layer) + '.npz')
    outputs = np.array([layer_outputs['arr_' + str(x)][0] for x in range(len(labels))]) # arrays of shape (num_texts, num_neurons) 


    pca = SupervisedPCARegressor(n_components = np.min(outputs.shape))
    # pca = SupervisedPCARegressor(threshold=0.1)
    pca.fit(outputs, labels)

    pca_outputs = pca.get_transformed_data(outputs)
    print(pca_outputs.shape)

    # class_vectors = [[], []]
    # for idx, label in enumerate(labels):
    #     class_vectors[label].append(pca_outputs[idx])
    # layer_metrics.append(classifier_metric(class_vectors))
    layer_cls_metrics.append(classifier_metric(outputs, labels, input_type=1))
    layer_v_metrics.append(cluster_metric(pca_outputs, labels, input_type=1)[2])


metrics = []
for idx, cls_metric in enumerate(layer_cls_metrics):
    metrics.append((f'Layer {idx} : clas_score {cls_metric}, v_measure {layer_v_metrics[idx]}', cls_metric, layer_v_metrics[idx]))

metrics = sorted(metrics, key=lambda a: a[1], reverse=True)
for metric in metrics:
    print(metric[0])
