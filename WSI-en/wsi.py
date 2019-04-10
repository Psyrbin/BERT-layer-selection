import numpy as np
import subprocess
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../supervisedPCA-Python')
from supervised_pca import SupervisedPCARegressor
from metrics import *
from bert.tokenization import FullTokenizer


data = np.load('data.npy')#[:10]
data = [(ex[0], int(ex[1]), int(ex[2])) for ex in data]

with open('texts', 'w') as f:
    for example in data:
        f.write(example[0] + '\n')

bert = '../cased_L-12_H-768_A-12/'
layers = '0,1,2,3,4,5,6,7,8,9,10,11'

bert_tokens = []
token_map = []
tokenizer = FullTokenizer(vocab_file=bert + 'vocab.txt', do_lower_case=False)

for text in data:
    text_tokens = ['[CLS]']
    text_map = []
    for word in text[0].split(' '):
        text_map.append(len(text_tokens))
        text_tokens.extend(tokenizer.tokenize(word))

    token_map.append(text_map)
    bert_tokens.append(text_tokens)


args = ['python', '../bert/extract_features.py']
args.append('--input_file=texts')
args.append('--output_file=WSI')
args.append('--vocab_file=' + bert + 'vocab.txt')
args.append('--bert_config_file=' + bert + 'bert_config.json')
args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
args.append('--layers=' + layers)
args.append('--max_seq_length=128')
args.append('--batch_size=8')
args.append('--do_lower_case=False')
args.append('--attention=False')
args.append('--mask_underscore=False')
# subprocess.run(args)

print(len(data))
print(len(token_map))
for i, a in enumerate(token_map):
    if len(a) <= int(data[i][1]):
        print(data[i])
        print(bert_tokens[i])
        print(a)
targets = [token_map[i][example[1]] for i, example in enumerate(data)]
labels = [example[2] for example in data]


layer_cls_metrics = []
layer_v_metrics = []
layer_avg_dist_metrics = []
for layer in range(12):
    layer_outputs = np.load('WSI_layer_' + str(layer) + '.npz')
    outputs = np.array([layer_outputs['arr_' + str(x)][target] for x, target in enumerate(targets)]) # arrays of shape (num_texts, num_neurons) 


    try:
        pca = SupervisedPCARegressor(n_components = np.min(outputs.shape)) # it works only if n_components <= min(num_features, num_examples)
        pca = SupervisedPCARegressor(threshold = 0.2)
        # pca = SupervisedPCARegressor(threshold=0.1)
        pca.fit(outputs, labels)

        pca_outputs = pca.get_transformed_data(outputs)
        print(pca_outputs.shape)
    except:
        pca_outputs = outputs

    class_vectors = [[] for _ in np.unique(labels)]
    for idx, label in enumerate(labels):
        class_vectors[label].append(pca_outputs[idx])

    layer_cls_metrics.append(classifier_metric(outputs, labels, input_type=1))
    layer_v_metrics.append(cluster_metrics(pca_outputs, labels, input_type=1)[2])
    layer_avg_dist_metrics.append(distance_metrics(class_vectors)[2])


metrics = []
for idx, cls_metric in enumerate(layer_cls_metrics):
    metrics.append((f'Layer {idx + 1} : clas_score {cls_metric}, v_measure {layer_v_metrics[idx]}, average_distance between classes {layer_avg_dist_metrics[idx]}', cls_metric, layer_v_metrics[idx], layer_avg_dist_metrics[idx]))

metrics = sorted(metrics, key=lambda a: a[1], reverse=True)
for metric in metrics:
    print(metric[0])