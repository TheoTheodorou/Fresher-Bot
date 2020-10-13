import cv2
import tensorflow as tf


def predict(filepath):
    CATEGORIES = ["apple", "apricot", "avocado", "banana", "blackberry", "blueberry", "cherry", "coconut",
                  "fig", "grape", "grapefruit", "kiwifruit", "lemon", "lime", "mango", "olive", "orange",
                  "passionfruit", "peach", "pear", "pineapple", "plum", "pomegranate", "raspberry",
                  "strawberry", "tomato", "watermelon"]

    def prepare(file):
        IMG_SIZE = 100
        img_array = cv2.imread(file, cv2.IMREAD_COLOR)
        img_array = img_array / 255
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    model = tf.keras.models.load_model("CNN.model")
    image = filepath
    image = prepare(image)

    prediction = model.predict([image])
    prediction = list(prediction[0])
    # print(prediction)
    # print(np.argmax(prediction))
    # print(CATEGORIES[prediction.index(max(prediction))])

    return CATEGORIES[prediction.index(max(prediction))]
