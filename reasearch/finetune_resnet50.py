import os
import mlflow
import mlflow.keras
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
import numpy as np

def datagen():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    return train_datagen, test_datagen

def train_test_set(train_datagen, test_datagen, train_path, test_path):
    training_set = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),  # ResNet50 default input size
        batch_size=32,
        class_mode='categorical')
    test_set = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),  # ResNet50 default input size
        batch_size=32,
        class_mode='categorical',
        shuffle=False)  # Ensure the order of the test set remains the same
    return training_set, test_set

def fine_tune_resnet50(train_path, test_path, epochs=5, learning_rate=1e-4):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze all layers in the base ResNet50 model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers for classification
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(5, activation='softmax')(x)

    # Create the full model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Data generators
    train_datagen, test_datagen = datagen()

    # Ensure directories exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    training_set, test_set = train_test_set(train_datagen, test_datagen, train_path, test_path)

    print("---------------------------- Fine-tuning ResNet50 model ----------------------------")
    with mlflow.start_run() as run:
        try:
            mlflow.log_param('base_model', 'ResNet50')
            mlflow.log_param('learning_rate', learning_rate)
            mlflow.log_param('epochs', epochs)

            # Fine-tune the model
            history = model.fit(x=training_set, validation_data=test_set, epochs=epochs)

            # Evaluate the model
            y_pred = np.argmax(model.predict(test_set), axis=1)
            y_true = test_set.classes

            # Calculate metrics
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

            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("Cohen's Kappa:", kappa)

            mlflow.keras.log_model(model, "ResNet50_model")
        except Exception as e:
            print(f"Exception during fine-tuning: {e}")
        finally:
            mlflow.end_run()
        print("---------------------------- Fine-tuning ended ----------------------------")

if __name__ == '__main__':
    train_path = 'E:/Deep Learning/TENSORFLOW/rice_image_detection/output_dataset/train'
    test_path = 'E:/Deep Learning/TENSORFLOW/rice_image_detection/output_dataset/test'
    fine_tune_resnet50(train_path, test_path)
