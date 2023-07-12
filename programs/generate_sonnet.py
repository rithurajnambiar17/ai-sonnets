import numpy as np
import tensorflow as tf
from programs.depickle_tokenizer import depickle_tokenizer

tokenizerLoaded = depickle_tokenizer('tokenizer.pickle')
model = tf.keras.models.load_model('models\sonnet_generator.h5')

def generate_sonnet(seed_text):
  MAX_SEQ_LEN = 163
  for i in range(14):
    token_list = tokenizerLoaded.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.text.pad_sequences([token_list], maxlen = MAX_SEQ_LEN-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""

    for word, index in tokenizerLoaded.word_index.items():
      if index == predicted:
        output_word = word
        break
    seed_text += " " + output_word
  return seed_text