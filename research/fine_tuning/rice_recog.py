import os
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def datagen():
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen, test_datagen

def train_test_set(train_datagen, test_datagen, train_path, test_path):
    training_set = train_datagen.flow_from_directory(
          train_path,
          target_size=(64, 64),
          batch_size=32,
          class_mode='categorical')
    test_set = test_datagen.flow_from_directory(
          test_path,
          target_size=(64, 64),
          batch_size=32,
          class_mode='categorical',
          shuffle=False)  # Ensure the order of the test set remains the same
    return training_set, test_set

def model(weight_decay):
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3],
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return cnn

def main():
    print("---------------------------Starting---------------------------")
    MLFLOW_TRACKING_URI = "https://dagshub.com/karmakaragradwip02/rice_image_detection_cnn.mlflow"
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'karmakaragradwip02'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '9ccb0f28354fcca6469017b32544fa0704b9c343'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("CNN Classifier")
    print("mlflow tracking set")
    print("---------------------------Mlflow URI set---------------------------------------")
    weight_decay = 1e-4  # Weight decay factor
    learning_rate = 1e-5  # Custom learning rate

    cnn = model(weight_decay)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print("defined model, optimizer, and compilation done")
    print("------------defined model, optimizer, and compilation done----------------------")
    train_path = 'E:/Deep Learning/TENSORFLOW/rice_image_detection/artifacts/data_preparation/train'
    test_path = 'E:/Deep Learning/TENSORFLOW/rice_image_detection/artifacts/data_preparation/test'
    train_datagen, test_datagen = datagen()

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")
    
    training_set, test_set = train_test_set(train_datagen, test_datagen, train_path, test_path)
    for data in training_set:
        images, labels = data
        print(images.dtype, labels.dtype)
        break
    print("----------------------------training begin---------------------------")
    with mlflow.start_run() as run:
        try:
            mlflow.log_param('weight_decay', weight_decay)
            mlflow.log_param('learning_rate', learning_rate)
            mlflow.log_param('epochs', 1)
            # Fit the model
            history = cnn.fit(x=training_set, validation_data=test_set, epochs=1)

            # Get predictions
            y_pred = np.argmax(cnn.predict(test_set), axis=1)
            y_true = test_set.classes  # Directly use the classes attribute

            # Calculate precision and recall
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # Log metrics
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1 score', f1)
            mlflow.log_metric('kappa', kappa)

            print("accuracy", accuracy)
            print("precision", precision)
            print("recall", recall)
            print("f1 score", f1)
            print("kappa", kappa)

            mlflow.keras.log_model(cnn, "model")
        except Exception as e:
            print(f"Exception during training: {e}")
        finally:
            mlflow.end_run()
        print("----------------------------training end---------------------------")

if __name__ == '__main__':
    main()
