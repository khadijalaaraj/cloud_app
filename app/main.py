from flask import Flask
import tensorflow as tf
from tensorflow import keras
import time
app = Flask(__name__)

@app.route("/")
def hello(): 
  
      
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    model = keras.models.Sequential([
       keras.layers.Flatten(input_shape=[28, 28]),
       keras.layers.Dense(300, activation="relu"),
       keras.layers.Dense(100, activation="relu"),
       keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
    start_time = time.time()
    history = model.fit(X_train, y_train, batch_size=32, epochs=5,
                    validation_data=(X_valid, y_valid))
  
    interval = time.time() - start_time 
    return interval 

if __name__ == "__main__":
    app.run(host='0.0.0.0')
