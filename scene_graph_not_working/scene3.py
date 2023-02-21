import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# Load image
img = cv2.imread("input.jpg")

# Load configuration and model
cfg = get_cfg()
cfg.merge_from_file("detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Run object detection and relationship extraction
outputs = predictor(img)

# Build scene graph
scene_graph = []
for i, obj in enumerate(outputs["instances"].pred_classes):
    node = {"id": i, "category": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[obj]}
    scene_graph.append(node)

for i, rel in enumerate(outputs["instances"].pred_relations):
    edge = {"id": i, "source": rel[0], "target": rel[1]}
    scene_graph.append(edge)