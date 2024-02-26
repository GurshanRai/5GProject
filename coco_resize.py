import json
import numpy as np

# file path to labels.json to resize
file_path = ""

# source resolution
_x = 1920
_y = 1080

# new resolution
targetSize = 256


x_scale = targetSize / _x
y_scale = targetSize / _y

with open(file_path, 'r') as file:
    data = json.load(file)
    
for anno in data['annotations']:
    print("Original: ", end="")
    print(anno['bbox'])
    
    for index in range(len(anno['bbox'])):
        if(index % 2 == 0):
            rescaled_dim = int(np.round(anno['bbox'][index]*x_scale))
            anno['bbox'][index] = rescaled_dim
        else:
            rescaled_dim = int(np.round(anno['bbox'][index]*y_scale))
            anno['bbox'][index] = rescaled_dim
    print("Resize: ", end="")
    print(anno['bbox'])

with open(file_path, 'w') as file:
    data = json.dump(data, file)

print("Rescaled ", end="")
print(file_path)
