import cv2
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

# Set up logger
setup_logger()

# Load configuration
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

# Initialize predictor
predictor = DefaultPredictor(cfg)

# Load image
image = cv2.imread("image.jpg")

# Run object detection
outputs = predictor(image)

# Get predictions
boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
classes = outputs["instances"].pred_classes.cpu().numpy()

# Create scene graph
# Create scene graph
scene_graph = {}
for i in range(len(classes)):
    bbox = boxes[i]
    category = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[classes[i]]
    bbox = [int(b) for b in bbox]  # convert bbox to a list of integers
    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # convert bbox from (x1, y1, x2, y2) to (x, y, w, h)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)  # convert bbox to XYXY_ABS format
    scene_graph[category] = bbox

# Print scene graph
print(scene_graph)


# Print scene graph
print(scene_graph)
