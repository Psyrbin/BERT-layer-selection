import numpy as np
import subprocess
import sys
sys.path.insert(0, '..')
from metrics import attention_metric, attention_metric2
from bert.tokenization import FullTokenizer

if len(sys.argv) == 1:
    input_file = '100_sents_converted.npy'
    bert = '../cased_L-12_H-768_A-12/'
    layers = '0,1,2,3,4,5,6,7,8,9,10,11'
else:
    input_file = sys.argv[1]
    bert = sys.argv[2]
    layers = '0'
    for i in range(int(sys.argv[3]) - 1):
        layers += ',' + str(i+1)

data = np.load(input_file)# [:1] # array of lists [text, target_word_idx, correct_word_idx]
with open('100_texts', 'w') as f:
    for example in data:
        f.write(example[0])

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

# correct = [[int(data[i][2])] for i in range(data.shape[0])]
correct = [list(map(int, example[2:])) for example in data]
targets = [int(data[i][1]) for i in range(data.shape[0])]
layer_metrics = []
for layer in range(len(layers)):
    att_data = np.load('anaphora_layer_' + str(layer) + '.npz')
    text_attentions_raw = [att_data['arr_' + str(x)] for x in range(data.shape[0])] # list of arrays of shape (num_heads, num_tokens, num_tokens) 

    # text_attentions = []
    # for idx, text_attention in enumerate(text_attentions_raw):
    #     text_attentions.append(text_attention[:,token_map[idx]][:,:,token_map[idx]])

    text_attentions = text_attentions_raw


    head_metrics = []
    for head in range(12):

        head_attentions = [text_attentions[i][head, token_map[i][targets[i]]] for i in range(len(text_attentions))]
        # attention_metric checks if highest attention is to the first token of the correct word, input should include only first tokens of every word
        # attention_metric2 checks if highest attention is to any of the tokens of the correct word, input should include all tokens and list mapping word to its first token
        
        # head_metrics.append(attention_metric(head_attentions, correct))
        head_metrics.append(attention_metric2(head_attentions, correct, token_map))
    layer_metrics.append(head_metrics)

metrics = []
for l_idx, layer in enumerate(layer_metrics):
    for h_idx, head in enumerate(layer):
        metrics.append((f'Layer {l_idx} head {h_idx}: {head}', head))

metrics = sorted(metrics, key=lambda a: a[1], reverse=True)
for metric in metrics:
    print(metric[0])
