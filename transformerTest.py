from transformerModel import tokenizer, END_TOKEN, preprocess_sentence, START_TOKEN, MAX_LENGTH, newModel
import tensorflow as tf


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
    output = tf.expand_dims(START_TOKEN, 0)
    for i in range(MAX_LENGTH):
        # MAKE PREDICTION ON MODEL LOADED FROM WEIGHTS
        model = newModel()
        model.load_weights("transformerWeights.h5")
        predictions = model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.equal(predicted_id, END_TOKEN[0]):
            break
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])
    return predicted_sentence
