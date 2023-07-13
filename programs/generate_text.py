import pickle
import numpy as np
import tensorflow as tf

def generate_text(text):
    model = tf.keras.models.load_model('./models/sonnet_generator.h5')
    tokenizerLoaded = pickle.load(open('./tokenizer.pickle', 'rb'))

    for _ in range(10 - len(text.split())):
        token_list = tokenizerLoaded.texts_to_sequences([text])[0]
        token_list = tf.keras.utils.pad_sequences([token_list], maxlen = 129, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        
        for word, index in tokenizerLoaded.word_index.items():
            if index == predicted:
                output_word = word
                break
        text += " " + output_word
    return text