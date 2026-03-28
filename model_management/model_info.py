model_lib = {
    # ── Faster R-CNN (torchvision) ──
    'fasterrcnn_resnet50_fpn': {
        'model_path': 'fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        'family': 'fasterrcnn',
    },
    'fasterrcnn_mobilenet_v3_large_fpn': {
        'model_path': 'fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth',
        'family': 'fasterrcnn',
    },
    'fasterrcnn_mobilenet_v3_large_320_fpn': {
        'model_path': 'fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
        'family': 'fasterrcnn',
    },
    # ── RetinaNet (torchvision) ──
    'retinanet_resnet50_fpn': {
        'model_path': 'retinanet_resnet50_fpn_coco-eeacdbdc.pth',
        'family': 'retinanet',
    },
    # ── SSD (torchvision) ──
    'ssd300_vgg16': {
        'model_path': 'ssd300_vgg16_coco-b556d3b4.pth',
        'family': 'ssd',
    },
    'ssdlite320_mobilenet_v3_large': {
        'model_path': 'ssdlite320_mobilenet_v3_large_coco-a79551df.pth',
        'family': 'ssdlite',
    },
    # ── FCOS (torchvision) ──
    'fcos_resnet50_fpn': {
        'model_path': 'fcos_resnet50_fpn_coco-99b0c9b7.pth',
        'family': 'fcos',
    },
    # ── YOLO (ultralytics — weights auto-downloaded on first use) ──
    # ``model_path`` always refers to the local artifact path relative
    # to ``model_management/models``. Remote download identifiers are
    # resolved by the model zoo when the artifact is missing.
    'yolov5n': {'model_path': 'yolov5nu.pt', 'family': 'yolo'},
    'yolov5s': {'model_path': 'yolov5su.pt', 'family': 'yolo'},
    'yolov5m': {'model_path': 'yolov5mu.pt', 'family': 'yolo'},
    'yolov5l': {'model_path': 'yolov5lu.pt', 'family': 'yolo'},
    'yolov5x': {'model_path': 'yolov5xu.pt', 'family': 'yolo'},
    'yolov8n': {'model_path': 'yolov8n.pt', 'family': 'yolo'},
    'yolov8s': {'model_path': 'yolov8s.pt', 'family': 'yolo'},
    'yolov8m': {'model_path': 'yolov8m.pt', 'family': 'yolo'},
    'yolov8l': {'model_path': 'yolov8l.pt', 'family': 'yolo'},
    'yolov8x': {'model_path': 'yolov8x.pt', 'family': 'yolo'},
    'yolov10n': {'model_path': 'yolov10n.pt', 'family': 'yolo'},
    'yolov10s': {'model_path': 'yolov10s.pt', 'family': 'yolo'},
    'yolov10m': {'model_path': 'yolov10m.pt', 'family': 'yolo'},
    'yolov10l': {'model_path': 'yolov10l.pt', 'family': 'yolo'},
    'yolov10x': {'model_path': 'yolov10x.pt', 'family': 'yolo'},
    'yolo11n': {'model_path': 'yolo11n.pt', 'family': 'yolo'},
    'yolo11s': {'model_path': 'yolo11s.pt', 'family': 'yolo'},
    'yolo11m': {'model_path': 'yolo11m.pt', 'family': 'yolo'},
    'yolo11l': {'model_path': 'yolo11l.pt', 'family': 'yolo'},
    'yolo11x': {'model_path': 'yolo11x.pt', 'family': 'yolo'},
    'yolo12n': {'model_path': 'yolo12n.pt', 'family': 'yolo'},
    'yolo12s': {'model_path': 'yolo12s.pt', 'family': 'yolo'},
    'yolo12m': {'model_path': 'yolo12m.pt', 'family': 'yolo'},
    'yolo12l': {'model_path': 'yolo12l.pt', 'family': 'yolo'},
    'yolo12x': {'model_path': 'yolo12x.pt', 'family': 'yolo'},
    # ── DETR (HuggingFace — auto-downloaded on first use) ──
    'detr_resnet50': {'model_path': 'detr_resnet50', 'family': 'detr'},
    'detr_resnet101': {'model_path': 'detr_resnet101', 'family': 'detr'},
    'conditional_detr_resnet50': {'model_path': 'conditional_detr_resnet50', 'family': 'detr'},
    # ── RT-DETR (ultralytics) ──
    'rtdetr_l': {'model_path': 'rtdetr-l.pt', 'family': 'rtdetr'},
    'rtdetr_x': {'model_path': 'rtdetr-x.pt', 'family': 'rtdetr'},
}


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

classes = {
    "vehicle": ['car', 'bus', 'train', 'truck'],
    "persons": [1, 2, 4],
    "roadside-objects": [10, 11, 13, 14]
}


annotation_cols = ('frame_index', 'target_id', 'bbox_x1', 'bbox_y1',
                   'bbox_x2', 'bbox_y2', 'score', 'object_category',)
