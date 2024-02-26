import PIL
import os
from PIL import Image

# image folder path
image_path = ""

for image in os.listdir(image_path):
    f_img = image_path+"/"+image
    img = Image.open(f_img)
    img = img.resize((256,256))
    img.save(f_img)

print('Resized ', end='')
print(image_path)