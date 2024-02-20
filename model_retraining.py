# Import library and modules
import os
import json
from mediapipe_model_maker import object_detector
from mediapipe_model_maker import face_stylizer
import tensorflow as tf

assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Set filepaths to coco datasets
# Link to full dataset: https://universe.roboflow.com/queendev9516-gmail-com/soccer-ball-oa830/dataset/1
training_dataset_path = 'coco_dataset/train'
valid_dataset_path = 'coco_dataset/valid'
test_dataset_path = 'coco_dataset/test'

# Create Dataset objects from filepath and set a cache directory
training_data = object_detector.Dataset.from_coco_folder(training_dataset_path, cache_dir="od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(valid_dataset_path, cache_dir="od_data/valid")
test_data = object_detector.Dataset.from_coco_folder(test_dataset_path, cache_dir="od_data/test")

# Print size of training and validation datasets
print("train_data size: ", training_data.size)
print("validation_data size: ", validation_data.size)

# Set spec to a preset model (MobilenetV2)
spec = object_detector.SupportedModels.MOBILENET_V2
# Set hyperparameters
hparams = object_detector.HParams(export_dir='retrained_model2')
# Set options for model
options = object_detector.ObjectDetectorOptions(supported_model=spec,hparams=hparams)

# Create and retrain model using datasets and options
model = object_detector.ObjectDetector.create(train_data=training_data,
                                                validation_data=validation_data,
                                                options=options)

# Evaluate model and return loss + metrics
loss,coco_metrics = model.evaluate(test_data,batch_size=32)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")

# Export model to filepath set in hyperparameters
model.export_model()
