# rice_image_detection_cnn
In the dataset there are 5 types(or 5 classes) of rice images namely Arborio, Basmati, Ipsala, Jasmine, Karacadag.
These contains 15000 images each and a total of 60000 images.

## Image Demo
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/karmakaragradwip02/rice_image_detection_cnn/assets/99462819/c2bb5b2e-84b6-499a-b289-00846197417f" alt="Arborio (13)" width="150"/>
  <img src="https://github.com/karmakaragradwip02/rice_image_detection_cnn/assets/99462819/52fc6c52-e437-4962-85c7-ee8a680e4aa5" alt="Basmati (13)" width="150"/>
  <img src="https://github.com/karmakaragradwip02/rice_image_detection_cnn/assets/99462819/b34a6730-4b63-4497-a34f-4cdb3c19d73b" alt="Ipsala (10004)" width="150"/>
  <img src="https://github.com/karmakaragradwip02/rice_image_detection_cnn/assets/99462819/dc66647d-d1c2-4989-b751-03ce7fcb00b3" alt="Jasmine (20)" width="150"/>
  <img src="https://github.com/karmakaragradwip02/rice_image_detection_cnn/assets/99462819/3a0848ae-81ea-451b-8514-239cf4c31a78" alt="Karacadag (92)" width="150"/>
</div>

The dataset is collected from kaggle.
The link of the kaggle dataset iis given below.
https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

This project is based upon single class image recognition.
The model is based upon Convolutional Neural Network.

The model predicts that what type of rice is in the given image.
The model is build using a tool named tensorflow(by google) and using VS Code in Anaconda Navigator.

The code is given in the repository.

Thank you.

MLFLOW_TRACKING_URI=https://dagshub.com/karmakaragradwip02/rice_image_detection_cnn.mlflow 
MLFLOW_TRACKING_USERNAME=karmakaragradwip02
MLFLOW_TRACKING_PASSWORD=9ccb0f28354fcca6469017b32544fa0704b9c343
python script.py

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/karmakaragradwip02/rice_image_detection_cnn.mlflow 
export MLFLOW_TRACKING_USERNAME=karmakaragradwip02
export MLFLOW_TRACKING_PASSWORD=9ccb0f28354fcca6469017b32544fa0704b9c343
```
## Data Pipeline
![data_pipeline](https://github.com/karmakaragradwip02/rice_image_detection_cnn/assets/99462819/3635c662-d3bf-402c-b7bb-2674eb817aea)

## Model Metrics
![graphs](https://github.com/karmakaragradwip02/rice_image_detection_cnn/assets/99462819/a1b7df74-680b-4139-8ad6-e7a4bf240949)
