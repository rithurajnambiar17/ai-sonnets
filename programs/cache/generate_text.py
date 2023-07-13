import sys
import random
import pickle
import numpy as np
# import tensorflow as tf

# model = tf.keras.models.load_model('models/sonnet_generator.h5')
model = pickle.load(open('./models/sonnet_generator.h5', 'rb'))
seq_length = 40

# Load the sonnet text file and convert to lowercase
sonnet_text = open("./data/sonnet_data/Sonnet.txt", "r").read().lower()

# Create a set of unique characters present in the text
chars = sorted(list(set(sonnet_text)))

# Create dictionaries to map characters to indices and vice versa
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}


def generated_text():
    # Generate text using the trained model
    start_index = random.randint(0, len(sonnet_text) - seq_length - 1)
    generated_text = sonnet_text[start_index:start_index+seq_length]
    # sys.stdout.write(generated_text)
    output = []
    for i in range(400):
        X_pred = np.zeros((1, seq_length, len(chars)))
        for j, char in enumerate(generated_text):
            X_pred[0, j, char_to_idx[char]] = 1
        pred = model.predict(X_pred, verbose=0)[0]
        next_char = idx_to_char[np.argmax(pred)]
        generated_text += next_char
        generated_text = generated_text[1:]
        # sys.stdout.write(next_char)
        # sys.stdout.flush()
        output.append(next_char)
    
    # convert the output list to a string
    result = ''.join(output)

    # Convert the output string to a list of lines by splitting using punctuation marks
    result = result.replace('.', '.\n')
    result = result.replace('?', '?\n')
    result = result.replace('!', '!\n')
    result = result.split('\n')

    return result