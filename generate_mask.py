import os
import cv2
import numpy as np
from PIL import Image
import torch
import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import List


#SEQUENCE OF COMMANDS TO BE EXECUTED IN. You have to check alternative of cd in windows i.e chdir
# 1: '%cd{HOME_DIR}',
# 2: '!git clone https: // github.com / IDEA - Research / GroundingDINO.git'
# 3: '%cd {HOME_DIR} / GroundingDINO',
# 4:'!pip install - q - e.', '!pip install - q roboflow'
# 5: '%cd {HOME}'
# 6:'!mkdir {HOME_DIR} / weights',
# 7: '%cd {HOME_DIR} / weights',
# 8: '!wget - q https: // github.com / IDEA - Research / GroundingDINO / releases / download / v0.1.0 - alpha / groundingdino_swint_ogc.pth'
# 9: '!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
# 10: '!{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git' you need to run this command to intsall segment anything model
# 11: '!pip uninstall -y supervision'
# 12: '!pip install -q -U supervision==0.6.0'


# Get the current working directory
HOME_DIR=''#os.getcwd() #need to run '!git clone https: // github.com / IDEA - Research / GroundingDINO.git' command to clone GroundingDINO repo in home dir
GD_DIR=''#os.path.join(HOME_DIR, "GroundingDINO") # need to run '!pip install - q - e.', '!pip install - q roboflow' after moving into GroundingDINO
weights_dir=''#os.path.join(HOME_DIR, "weights") #need to download weights here via command '!wget - q https: // github.com / IDEA - Research / GroundingDINO / releases / download / v0.1.0 - alpha / groundingdino_swint_ogc.pth'
# import sys
# !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git' you need to run this command to intsall segment anything model
# !pip uninstall -y supervision
# !pip install -q -U supervision==0.6.0



# Paths to configuration and checkpoint files
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME_DIR, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME_DIR, "weights", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join(HOME_DIR, "weights", "sam_vit_h_4b8939.pth")

# Check if files exist
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

# Set the device to GPU if available, otherwise use CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the GroundingDINO model for object detection
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Encoder version for the SAM model
SAM_ENCODER_VERSION = "vit_h"

# Load the SAM model and create a predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Segment objects in an image based on bounding box coordinates.

    Args:
        sam_predictor (SamPredictor): SAM model predictor.
        image (np.ndarray): Input image.
        xyxy (np.ndarray): Bounding box coordinates [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: Array of segmentation masks.
    """
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def enhance_class_name(class_names: List[str]) -> List[str]:
    """
    Enhance class names by adding a prefix.

    Args:
        class_names (List[str]): List of class names.

    Returns:
        List[str]: List of enhanced class names.
    """
    return [f"all {class_name}s" for class_name in class_names]


def create_mask(image, objec):
    """
    Create segmentation masks based on object detection.

    Args:
        image (np.ndarray): Input image.
        objec (str): Object to detect.

    Returns:
        np.ndarray: Segmentation mask.
    """
    objec = objec.split(',')
    CLASSES = objec
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image = np.array(image)
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    box_annotator = sv.BoxAnnotator()
    labels = [f"{CLASSES[class_id]} " for class_id in list(detections.class_id)]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections.mask = segment(sam_predictor=sam_predictor, image=image_rgb, xyxy=detections.xyxy)

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [f"{CLASSES[class_id]} " for class_id in list(detections.class_id)]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    titles = [CLASSES[class_id] for class_id in detections.class_id]
    images_list = []

    for i, _ in enumerate(detections.class_id):
        if titles[i] in objec:
            pil_im = Image.fromarray(detections.mask[i])
            pil_im.save(f'image{i}.png')
            images_list.append(pil_im)

    desired_mask = np.array(images_list[0])
    for i in range(1, len(images_list)):
        temp_mask = np.array(images_list[i])
        desired_mask = cv2.bitwise_or(temp_mask, desired_mask)

    desired_mask = np.repeat(desired_mask[:, :, np.newaxis], 3, axis=2)
    desired_mask = np.asarray(desired_mask, dtype=np.uint8)
    mask3d = desired_mask
    mask3d[desired_mask > 0] = 255.0
    cv2.imwrite("desired_mask.png", mask3d)
    return mask3d


def prediction(init_image, word_mask):
    """
    Perform object detection, segmentation, and create a final segmentation mask.

    Args:
        init_image (np.ndarray): Input image.
        word_mask (str): Object to detect.

    Returns:
        np.ndarray: Final segmentation mask.
    """
    mask = create_mask(init_image, word_mask)
    return mask
