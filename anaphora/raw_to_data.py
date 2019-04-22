import numpy as np

with open('train.c.txt', 'r') as f:
    res = []
    for idx, line in enumerate(f):
        if idx % 5 == 0:
            sent = [line]
        elif idx % 5 == 1:
            sent.append(line)
        elif idx % 5 == 3:
            sent.append(line)
            res.append(sent)
np.save('data1', res)