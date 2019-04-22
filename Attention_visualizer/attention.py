from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, Div
from bokeh.models.widgets import TextInput, Button
from bokeh.io import curdoc
import sys
import subprocess
import numpy as np
from cgi import escape

if len(sys.argv) == 1:
    input_file = 'input.txt'
    bert = '../cased_L-12_H-768_A-12/'
    layers = '0,1,2,3,4,5,6,7,8,9,10,11'
elif len(sys.argv) == 3:
    input_file = 'input.txt'
    bert = sys.argv[1]
    layers = '0'
    for i in range(int(sys.argv[2]) - 1):
        layers += ',' + str(i+1)
else:
    input_file = sys.argv[1]
    bert = sys.argv[2]
    layers = '0'
    for i in range(int(sys.argv[3]) - 1):
        layers += ',' + str(i+1)

layer_idxs = [int(x) for x in layers.split(',')]
text_count = 1


text_input = TextInput()
button = Button(label='Обработать', button_type='success')
div = Div(text='')

def calculate():
    with open(input_file, 'w') as f:
        f.write(text_input.value)

    args = ['python', '../bert/extract_features.py']
    args.append('--input_file=' + input_file)
    args.append('--output_file=')
    args.append('--vocab_file=' + bert + 'vocab.txt')
    args.append('--bert_config_file=' + bert + 'bert_config.json')
    args.append('--init_checkpoint=' + bert + 'bert_model.ckpt')
    args.append('--layers=' + layers)
    args.append('--max_seq_length=128')
    args.append('--batch_size=8')
    args.append('--do_lower_case=False')
    args.append('--attention=True')
    args.append('--mask_underscore=True')
    subprocess.run(args)

    with open('tokens.txt', 'r') as f:
        texts = f.read().split('\n\n')
    if texts[-1] == '':
        texts = texts[:-1]
    print(texts, text_count)
    assert len(texts) == text_count

    string = '<div style="font-size: 18px">'

    for text_idx, text in enumerate(texts):
        tokens = []
        for i, line in enumerate(text.split('\n')):
            tokens.append(line)
            if line == '[MASK]':
                mask_idx = i
        print(mask_idx)

        # string += '<HTML>\n'
        for idx in layer_idxs:
            string += '<div>Layer ' + str(idx) + '</div><br>\n'
            layer = np.load('_layer_' + str(idx) + '.npy')
            # layer = layer['arr_' + str(text_idx)]
            for head in range(layer.shape[0]):
                string += '<div>HEAD ' + str(head) + '</div>\n'
                attentions = layer[head, mask_idx,:]
                for word_idx, attn in enumerate(attentions):
                    if attn < 0:
                        color = 'rgba(250, 10, 10, ' + str(-attn)
                    else:
                        color = 'rgba(10, 10, 250, ' + str(attn)
                    string += '<span style="background-color:' + color + ');">' + escape(tokens[word_idx]) + '</span> <span> </span>'
                string += '<br><br>\n'
            string += '<br><br><br><br>\n'

            # res.write('</HTML>')
    string += '</div>'
    div.text = string

button.on_click(calculate)

l = layout([
    [text_input], [button], [div]
], sizing_mode='scale_width')

curdoc().add_root(l)
curdoc().title = "Attention visualizer"
