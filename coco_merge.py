from coco_assistant import COCO_Assistant

# Specify image and annotation directories
img_dir = "/home/codychow/5GProject/5GProject-1/test/images"
ann_dir = "/home/codychow/5GProject/5GProject-1/test/annotations"

# Create COCO_Assistant object
cas = COCO_Assistant(img_dir, ann_dir)

cas.merge()