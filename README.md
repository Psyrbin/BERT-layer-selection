Для работы скриптов требуется чекпоинт BERT (можно скачать с https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip), разархивировать в текущую директорию. 

В папке `anaphora` скрипт `anaphora.py` запускается так:
```
cd anaphora
python anaphora.py 
```

В папке `sentiment` скрипт `sentiment.py`
```
cd sentiment
bokeh serve --show sentiment.py --port [port_num]
``` 
В папке `POS` скрипт `pos.py`
```
cd POS
bokeh serve --show pos.py --port [port_num]
``` 
В папке `WSI` скрипт `wsi.py`
```
cd WSI
bokeh serve --show wsi.py --port [port_num]
``` 

В папке `Attention_visualizer` скрипт `attention.py` запускается так:
```
cd Attention_visualizer
bokeh serve --show attention.py --port [port_num]
```
Он открывает страницу, на которой в поле надо ввести текст, одно из слов которого заменено на _. После нажатия кнопки "Обработать" выдастся визуализация внимения на разных слоях со слова, замененного на _.

# Требуемые библиотеки:
bokeh 1.0.2

numpy >= 1.15.4

scikit-learn >= 0.20.1