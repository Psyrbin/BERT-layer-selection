import numpy as np

data = np.load('data1.npy')[:100]
converted = []
for sentence in data:
    text = sentence[0]
    target = sentence[1][:-1]
    correct = sentence[2].split(' ')[-1][:-1]
    correct_len = len(sentence[2].split(' '))
    print('a', correct_len)


    target_idx = -1
    correct_idx = -1

    for idx, word in enumerate(text.split()):
        if word == target or word[:-1] == target:
            if target_idx != -1:
                print('target',sentence, target_idx, idx)
            target_idx = idx
        if word == correct or word[:-1] == correct or word[:-2] == correct:
            if correct_idx != -1:
                print('correct', sentence, correct_idx, idx)
            correct_idx = [i for i in range(idx, idx-correct_len, -1)]
            while len(correct_idx) < 3:
                correct_idx.append(correct_idx[-1])
            print(correct_idx)
    # converted.append([text, target_idx, correct_idx])
    converted.append([text, target_idx])
    converted[-1].extend(correct_idx)
np.save('100_sents_converted', converted)