import numpy as np
import subprocess
import sys
sys.path.insert(0, '..')
from metrics import attention_metric


# MAP INDEXES TO TOKENS PLZ
# CORRECT[i] = range(tok[i], tok[i+1])
# TARGET = -||-
# IN METRIC ARGMAX() in CORRECT[i]

data = np.load('100_sents_converted.npy')#[:20] # array of lists [text, target_word_idx, correct_word_idx]
with open('100_texts', 'w') as f:
    for example in data:
        f.write(example[0])

bert = '../cased_L-12_H-768_A-12/'
layers = '0,1,2,3,4,5,6,7,8,9,10,11'

args = ['python', '../bert/extract_features.py']
args.append('--input_file=100_texts')
args.append('--output_file=anaphora')
args.append('--vocab_file=' + bert + 'vocab.txt')
args.append('--bert_config_file=' + bert + 'bert_config.json')
args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
args.append('--layers=' + layers)
args.append('--max_seq_length=128')
args.append('--batch_size=8')
args.append('--do_lower_case=False')
args.append('--attention=True')
args.append('--mask_underscore=False')
subprocess.run(args)

correct = [int(data[i][2]) for i in range(data.shape[0])]
targets = [int(data[i][1]) for i in range(data.shape[0])]
layer_metrics = []
for layer in range(12):
    att_data = np.load('anaphora_layer_' + str(layer) + '.npz')
    text_attentions = [att_data['arr_' + str(x)] for x in range(data.shape[0])] # list of arrays of shape (num_heads, num_tokens, num_tokens) 
    head_metrics = []
    for head in range(12):
        head_attentions = [text_attentions[i][head][targets[i]] for i in range(len(text_attentions))]  
        head_metrics.append(attention_metric(head_attentions, correct))
    layer_metrics.append(head_metrics)

metrics = []
for l_idx, layer in enumerate(layer_metrics):
    for h_idx, head in enumerate(layer):
        metrics.append((f'Layer {l_idx} head {h_idx}: {head}', head))

metrics = sorted(metrics, key=lambda a: a[1], reverse=True)
for metric in metrics:
    print(metric[0])