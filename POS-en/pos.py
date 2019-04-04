import numpy as np
import subprocess
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../supervisedPCA-Python')
from supervised_pca import BaseSupervisedPCA
from metrics import *

data = np.load('brown_100_sents.npy')# [:20] # array of lists [text, target_word_idx, correct_word_idx]
with open('sentences', 'w') as f:
    for text in data:
            string = ''
            for word in text:
                    string += word[0] + ' '
            f.write(string[:-1] + '\n')

bert = '../cased_L-12_H-768_A-12/'
layers = '0,1,2,3,4,5,6,7,8,9,10,11'

args = ['python', '../bert/extract_features.py']
args.append('--input_file=sentences')
args.append('--output_file=POS')
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

pos_to_idx = {}
max_label = 0
correct = []
for text in data:
    labels = []
    for word in text:
        if word[1] not in pos_to_idx:
            pos_to_idx[word[1]] = max_label
            max_label += 1
        labels.append(pos_to_idx[word[1]])
    correct.append(labels)


layer_metrics = []
for layer in range(12):
    layer_outputs = np.load('POS_layer_' + str(layer) + '.npz')
    text_outputs = [layer_outputs['arr_' + str(x)] for x in range(data.shape[0])] # list of arrays of shape (num_tokens, num_neurons) 


    pca = SupervisedPCARegressor(n_components = np.min(outputs.shape)) # it works only if n_components <= min(num_features, num_examples)
    # pca = SupervisedPCARegressor(threshold=0.1)
    pca.fit(outputs, labels)

    pca_outputs = pca.get_transformed_data(outputs)

    class_vectors = [[] for _ in range(max_label)]
    for t_idx, text in enumerate(data):
        for w_idx, word in enumerate(text):
            # do something with multiple tokens per word, for example all tokens are the same pos
            class_vectors[pos_to_idx[word[1]]].append(text_outputs[t_idx][w_idx])
    layer_metrics.append(classifier_metric(class_vectors))


metrics = []
for idx, metric in enumerate(layer_metrics):
    metrics.append((f'Layer {idx} : {metric}', metric))

metrics = sorted(metrics, key=lambda a: a[1], reverse=True)
for metric in metrics:
    print(metric[0])
