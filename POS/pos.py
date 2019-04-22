import numpy as np
import subprocess
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../supervisedPCA-Python')
from supervised_pca import SupervisedPCARegressor
from metrics import *
from bert.tokenization import FullTokenizer
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, Div, Plot, LinearAxis, Grid
from bokeh.models.widgets import TextInput, Button, Select
from bokeh.models.glyphs import VBar
from bokeh.io import curdoc

if len(sys.argv) == 1:
    input_file = 'brown_100_sents.npy'
    bert = '../cased_L-12_H-768_A-12/'
    layers = '0,1,2,3,4,5,6,7,8,9,10,11'
else:
    input_file = sys.argv[1]
    bert = sys.argv[2]
    layers = '0'
    for i in range(int(sys.argv[3]) - 1):
        layers += ',' + str(i+1)

data = np.load(input_file)[:30]
with open('sentences', 'w') as f:
    for text in data:
            string = ''
            for word in text:
                    string += word[0] + ' '
            f.write(string[:-1] + '\n')

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



    class_vectors = [[] for _ in range(max_label)]
    for t_idx, text in enumerate(data):
        for w_idx, word in enumerate(text):
            # do something with multiple tokens per word, for example all tokens are the same pos
            class_vectors[pos_to_idx[word[1]]].append(text_outputs[t_idx][w_idx])

    labels = []
    for idx, vectors in enumerate(class_vectors):
        labels += [idx for _ in range(len(vectors))]

    outputs = np.concatenate(class_vectors, axis=0)


    try:
        pca = SupervisedPCARegressor(n_components = np.min(outputs.shape)) # it works only if n_components <= min(num_features, num_examples)
        pca = SupervisedPCARegressor(threshold = 0.7)
        # pca = SupervisedPCARegressor(threshold=0.1)
        pca.fit(outputs, labels)

        pca_outputs = pca.get_transformed_data(outputs)
        print(pca_outputs.shape)
    except:
        pca_outputs = outputs


    layer_cls_metrics.append(classifier_metric(outputs, labels, input_type=1, validate=True, fold_count=5))
    layer_v_metrics.append(cluster_metrics(pca_outputs, labels, input_type=1)[2])
    layer_avg_dist_metrics.append(distance_metrics(class_vectors))   


metrics = []
for idx, cls_metric in enumerate(layer_cls_metrics):
    metrics.append((f'Layer {idx + 1} : avg_clas_score {cls_metric[0]}, min_val_score {cls_metric[1]}, v_measure {layer_v_metrics[idx]}, average_distance between classes {layer_avg_dist_metrics[idx]}, conf_interval {cls_metric[3]}', cls_metric[0], cls_metric[1], layer_v_metrics[idx], layer_avg_dist_metrics[idx], cls_metric[3], cls_metric[4], idx))

metric_names = ['Logistic regression accuracy', 'Logistic regression min', 'Clustering v-measure', 'Average distance ratio']
xs = []
ys = []
plots = []
sources = []
glyphs = []
renderers = []
sorters = []

def sort(idx, metric):
    if metric == 'Layer':
        metric_idx = -2
    else:
        metric_idx = metric_names.index(metric)

    tmp_metrics = metrics.copy()
    reverse = True
    if metric == 'Average distance ratio' or metric == 'Layer':
        reverse = False
    tmp_metrics = sorted(tmp_metrics, key=lambda a: a[1 + metric_idx], reverse=reverse)
    xs_tmp = [metric[-1] + 1 for metric in tmp_metrics]
    for jdx, i in enumerate(xs_tmp):
        xs[idx][i-1] = jdx+1
    sources[idx] = ColumnDataSource(dict(x=xs[idx], top=ys[idx]))

    plots[idx].renderers.remove(renderers[idx])
    plots[idx].xaxis.ticker = [i + 1 for i in range(len(xs[idx]))]

    override = {}
    for i in range(len(metrics)):
        override[xs[idx][i]] = str(i+1)
    plots[idx].xaxis.major_label_overrides = override

    renderers[idx] = plots[idx].add_glyph(sources[idx], VBar(x='x', top='top', bottom=0, width = 0.2))

for idx, name in enumerate(metric_names):
    plots.append(figure(title=name, plot_width=600, plot_height=600, toolbar_location=None))
    xs.append([metric[-1] + 1 for metric in metrics])
    ys.append([metric[1 + idx] for metric in metrics])
    sources.append(ColumnDataSource(dict(x=xs[-1], top=ys[-1])))

    glyphs.append(VBar(x='x', top='top', bottom=0, width = 0.2))
    renderers.append(plots[-1].add_glyph(sources[-1], glyphs[-1]))
    plots[-1].xaxis.ticker = [i for i in xs[-1]]
    sorters.append(Select(title='Sort by', options = ['Layer'] + metric_names))
    sorters[-1].on_change('value', lambda attr, old, new, idx=idx: sort(idx, new))



interval_plot = figure(title='Logistic regression cross validation scores', plot_width=600, plot_height=600, toolbar_location=None)
lows = [metric[-2][0] for metric in metrics]
highs = [metric[-2][1] for metric in metrics]
lows1 = [metric[-3][0] for metric in metrics]
highs1 = [metric[-3][1] for metric in metrics]
x_vals = [i + 1 for i in range(len(lows))]
interval_plot.vbar(x_vals, 0.3, highs1, lows1, line_color='blue')
interval_plot.segment(x_vals, lows, x_vals, highs, line_color='black', line_width=2)
interval_plot.rect(x_vals, lows, 0.2, 0.0001, line_color='black')
interval_plot.rect(x_vals, highs, 0.2, 0.0001, line_color='black')
interval_plot.xaxis.ticker = x_vals


layout_list = [[sorters[i], plots[i]] for i in range(len(plots))]
layout_list[0].append(interval_plot)
l = layout(layout_list, sizing_mode='fixed')

curdoc().add_root(l)
curdoc().title = "POS"
