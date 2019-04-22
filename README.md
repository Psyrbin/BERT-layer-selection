Для работы скриптов требуется чекпоинт BERT (можно скачать с https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip). 
Во всех скриптах можно не указывать аргументов, тогда они будут искать папку с именем `cased_L-12_H-768_A-12`, в которой должен лежать чекпоинт (по ссылке выше скачивается архив с такой папкой) и возьмут готовые входные файлы.
В папке `anaphora` скрипт `anaphora.py` запускается так:
```
python anaphora.py [input_file] [bert_checkpoint_folder] [number_of_layers]
```

В папках `POS`, `sentiment` и `WSI` скрипты с такими же названиями запускаются так (на примере `sentiment.py`):
```
bokeh serve --show sentiment.py --args [input_file] [bert_checkpoint_folder] [number_of_layers]
``` 

В папке `Attention visualizer` скрипт `attention.py` запускается так:
```
bokeh serve --show attention.py --args [bert_checkpoint_folder] [number_of_layers]
```
Он открывает страницу, на которой в поле надо ввести текст, одно из слов которого заменено на _. После нажатия кнопки "Обработать" выдастся визуализация внимения на разных слоях со слова, замененного на _.

(требуется Bokeh версии 1.0.2)