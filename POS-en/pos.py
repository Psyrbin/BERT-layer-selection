import numpy as np
import subprocess
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../supervisedPCA-Python')
from supervised_pca import SupervisedPCARegressor
from metrics import *
from bert.tokenization import FullTokenizer

data = np.load('brown_100_sents.npy')[:2] # array of lists [text, target_word_idx, correct_word_idx]
with open('sentences', 'w') as f:
    for text in data:
            string = ''
            for word in text:
                    string += word[0] + ' '
            f.write(string[:-1] + '\n')

bert = '../cased_L-12_H-768_A-12/'
layers = '0,1,2,3,4,5,6,7,8,9,10,11'

bert_tokens = []
token_map = []
tokenizer = FullTokenizer(vocab_file=bert + 'vocab.txt', do_lower_case=False)

for text in data:
    text_tokens = ['[CLS]']
    text_map = []
    for word in text:
        text_map.append(len(text_tokens))
        text_tokens.extend(tokenizer.tokenize(word[0]))

    token_map.append(text_map)
    bert_tokens.append(text_tokens)

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


layer_cls_metrics = []
layer_v_metrics = []
layer_avg_dist_metrics = []
for layer in range(12):
    layer_outputs = np.load('POS_layer_' + str(layer) + '.npz')
    text_outputs_raw = [layer_outputs['arr_' + str(x)] for x in range(data.shape[0])] # list of arrays of shape (num_tokens, num_neurons) 

    # remove vectors from excess tokens
    text_outputs = []
    for idx, text_output in enumerate(text_outputs_raw):
        text_outputs.append(text_output[token_map[idx]])


    # FIX PCA

    # pca = SupervisedPCARegressor(n_components = np.min(outputs.shape)) # it works only if n_components <= min(num_features, num_examples)
    # pca = SupervisedPCARegressor(threshold=0.1)
    # pca.fit(outputs, labels)

    # pca_outputs = pca.get_transformed_data(outputs)

    class_vectors = [[] for _ in range(max_label)]
    for t_idx, text in enumerate(data):
        for w_idx, word in enumerate(text):
            # do something with multiple tokens per word, for example all tokens are the same pos
            class_vectors[pos_to_idx[word[1]]].append(text_outputs[t_idx][w_idx])
    layer_cls_metrics.append(classifier_metric(class_vectors))
    layer_v_metrics.append(cluster_metrics(class_vectors)[2])
    layer_avg_dist_metrics.append(distance_metrics(class_vectors)[2])


metrics = []
for idx, cls_metric in enumerate(layer_cls_metrics):
    metrics.append((f'Layer {idx + 1} : clas_score {cls_metric}, v_measure {layer_v_metrics[idx]}, average_distance between classes {layer_avg_dist_metrics[idx]}', cls_metric, layer_v_metrics[idx], layer_avg_dist_metrics[idx]))

metrics = sorted(metrics, key=lambda a: a[1], reverse=True)
for metric in metrics:
    print(metric[0])