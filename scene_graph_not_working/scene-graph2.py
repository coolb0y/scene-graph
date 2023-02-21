import cv2
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode

def bbox_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
    return iou

# Set up logger
setup_logger()

# Load configuration
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://VG-Detection/faster_rcnn_R_101_C4_caffemaxpool/model_final.pkl"

# Initialize predictor
predictor = DefaultPredictor(cfg)

# Load image
image = cv2.imread("image.jpg")

# Run object detection
outputs = predictor(image)

# Extract object proposals
proposals = outputs["proposals"].proposal_boxes.tensor.cpu().numpy()

# Get object classes
classes = outputs["instances"].pred_classes.cpu().numpy()

# Create scene graph
scene_graph = {}
for i in range(len(classes)):
    bbox = proposals[i]
    category = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[classes[i]]
    bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    bbox = [int(b) for b in bbox]
    scene_graph[category] = bbox

# Extract object relations
relations = outputs["instances"].pred_boxes.tensor.cpu().numpy()
relation_classes = outputs["instances"].pred_classes.cpu().numpy()

# Create relation graph
relation_graph = {}
for i in range(len(relations)):
    relation_bbox = relations[i]
    relation_category = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[relation_classes[i]]
    for j in range(len(scene_graph)):
        obj_category = list(scene_graph.keys())[j]
        obj_bbox = scene_graph[obj_category]
        if bbox_iou(relation_bbox, obj_bbox) > 0.5:
            if relation_category not in relation_graph:
                relation_graph[relation_category] = {}
            if obj_category not in relation_graph[relation_category]:
                relation_graph[relation_category][obj_category] = []
            relation_graph[relation_category][obj_category].append(bbox_iou(relation_bbox, obj_bbox))

# Print relation graph
for relation in relation_graph:
    print(relation)
    for obj in relation_graph[relation]:
        iou_scores = relation_graph[relation][obj]
        mean_iou = sum(iou_scores) / len(iou_scores)
        print(f"  {relation} - {obj}: {mean_iou}")

