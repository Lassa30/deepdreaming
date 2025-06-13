import numpy as np

IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TO_MODEL_SHAPE = (224, 224, 3)
