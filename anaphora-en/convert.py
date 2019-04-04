import numpy as np

data = np.load('100_sents.npy')
converted = []
for sentence in data:
    text = sentence[0]
    target = sentence[1][:-1]
    correct = sentence[2][:-1]

    target_idx = -1
    correct_idx = -1

    for idx, word in enumerate(text.split()):
        if word == target or word[:-1] == target:
            if target_idx != -1:
                print('target',sentence, target_idx, idx)
            target_idx = idx
        if word == correct or word[:-1] == correct:
            if correct_idx != -1:
                print('correct', sentence, correct_idx, idx)
            correct_idx = idx
    converted.append([text, target_idx, correct_idx])
np.save('100_sents_converted', converted)