import cv2
import numpy as np
import torch
from PIL import Image

# def lane_detection(processor, model, image):

# ================================
# ========= Basic Utils ==========
# ================================
def resize_image(image, width):
    """
    Resizes an image while preserving its aspect ratio.

    Args:
        image: OpenCV image.
        width: Desired output width in pixels.

    Returns:
        The resized image.
    """

    scale = width / image.shape[1]
    height = int(image.shape[0] * scale)

    return cv2.resize(image, (width, height))

def map_masks_and_boxes_to_image(
    image,
    masks=None,
    detections=None,
    mask_alpha=0.4,
    draw_class_names=True,
    mask_colors=None
):
    """
    Overlay masks and bounding boxes onto an image.

    Parameters
    ----------
    image : np.ndarray
        Original OpenCV BGR image.

    masks : list[np.ndarray] or np.ndarray
        One mask or a list of masks.
        Each mask can be:
        - single-channel binary mask
        - 3-channel BGR mask

    detections : list[dict]
        List of detections from object_detection().

    mask_alpha : float
        Transparency of mask overlays.

    draw_class_names : bool
        Whether to draw class labels above bounding boxes.

    mask_colors : list[tuple]
        Optional list of BGR colors, one per mask.
        Example:
        [
            (0, 255, 0),    # green
            (0, 0, 255)     # red
        ]

    Returns
    -------
    output_image : np.ndarray
        Image with masks and bounding boxes drawn.
    """

    output_image = image.copy()

    if masks is None:
        masks = []

    if detections is None:
        detections = []

    # Allow a single mask to be passed directly
    if not isinstance(masks, list):
        masks = [masks]

    # Default colors are BGR, not RGB
    if mask_colors is None:
        mask_colors = [
            (0, 255, 0),      # green: road
            (0, 0, 255),      # red: lanes
            (255, 0, 0),      # blue
            (0, 255, 255),    # yellow
            (255, 0, 255),    # magenta
            (255, 255, 0),    # cyan
        ]

    # -----------------------------
    # Draw masks
    # -----------------------------
    for mask_index, mask in enumerate(masks):
        if mask is None:
            continue

        color = mask_colors[mask_index % len(mask_colors)]

        # Convert 3-channel mask to single-channel binary mask
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        # Make sure mask is uint8
        mask_gray = mask_gray.astype(np.uint8)

        # Create binary mask: anything nonzero becomes mask area
        binary_mask = mask_gray > 0

        # Create a color overlay image
        colored_mask = np.zeros_like(output_image)
        colored_mask[binary_mask] = color

        # Blend only where the mask exists
        blended_image = cv2.addWeighted(
            output_image,
            1.0,
            colored_mask,
            mask_alpha,
            0
        )

        output_image[binary_mask] = blended_image[binary_mask]

    # -----------------------------
    # Draw bounding boxes
    # -----------------------------
    for detection in detections:
        box = detection["box"]
        class_name = detection.get("class_name", "object")
        confidence = detection.get("confidence", 0.0)

        x1, y1, x2, y2 = box

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        box_color = (255, 255, 255)

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        if draw_class_names:
            label = f"{class_name}: {confidence:.2f}"

            cv2.putText(
                output_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
                cv2.LINE_AA
            )

    return output_image

# ===================================
# ========== Segmenetation ==========
# ===================================
def segment_roads(processor, model, image):
    """
    Runs SegFormer semantic segmentation and returns only the road mask.

    Args:
        processor: Loaded SegFormer image processor.
        model: Loaded SegFormer semantic segmentation model.
        image: OpenCV BGR image.

    Returns:
        A 3-channel OpenCV BGR mask image where:
            - road pixels are white
            - non-road pixels are black

    Example:
        road_mask = draw_segformer_road_mask(
            processor,
            segformer_model,
            image_resized
        )
    """

    # SegFormer expects RGB images, but OpenCV uses BGR.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert NumPy RGB image to PIL image.
    pil_image = Image.fromarray(image_rgb)

    # Prepare image for SegFormer.
    inputs = processor(
        images=pil_image,
        return_tensors="pt"
    )

    # Run model in inference mode.
    with torch.no_grad():
        outputs = model(**inputs)

    # Model output shape:
    # [batch_size, num_classes, height, width]
    logits = outputs.logits

    # Resize prediction back to original image size.
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(image.shape[0], image.shape[1]),
        mode="bilinear",
        align_corners=False
    )

    # Pick the most likely class for every pixel.
    predicted_class_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # For ADE20K SegFormer, road is commonly class id 6.
    road_class_id = 6

    # Create a single-channel binary mask.
    road_mask = np.zeros_like(predicted_class_map, dtype=np.uint8)
    road_mask[predicted_class_map == road_class_id] = 255

    # Convert single-channel grayscale mask to 3-channel BGR,
    # so it can be displayed or stacked with other OpenCV images.
    road_mask_bgr = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)

    return road_mask_bgr


# TODO: install and test model
def segment_drivable_area_and_lanes_twinlitenet(
    model,
    image,
    device="cpu",
    input_size=(640, 360)
):
    """
    Runs TwinLiteNet+ and returns drivable-area and lane-line masks.

    Args:
        model: Loaded TwinLiteNet+ model.
        image: OpenCV BGR image.
        device: "cpu" or "cuda".
        input_size: Model input size as (width, height).

    Returns:
        A tuple:
            drivable_mask_bgr, lane_mask_bgr

        Each mask is a 3-channel OpenCV BGR image with:
            - white pixels where the class exists
            - black pixels elsewhere

    Notes:
        TwinLiteNet+ usually performs two tasks:
            1. Drivable area segmentation
            2. Lane line segmentation

        Depending on the model implementation, the output order may be:
            outputs[0] = drivable area
            outputs[1] = lane line

        If the masks look swapped, reverse the two outputs.
    """

    original_height = image.shape[0]
    original_width = image.shape[1]

    # -----------------------------
    # Preprocess image
    # -----------------------------

    # OpenCV uses BGR, but PyTorch vision models usually expect RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to the model input size.
    resized_rgb = cv2.resize(image_rgb, input_size)

    # Convert from 0-255 integers to 0.0-1.0 floats.
    image_float = resized_rgb.astype(np.float32) / 255.0

    # Convert from HWC to CHW.
    # OpenCV/NumPy image shape:
    #     height, width, channels
    #
    # PyTorch model image shape:
    #     channels, height, width
    image_chw = image_float.transpose(2, 0, 1)

    # Add batch dimension.
    # Shape becomes:
    #     1, channels, height, width
    input_tensor = torch.from_numpy(image_chw).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # -----------------------------
    # Run inference
    # -----------------------------
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)

    # -----------------------------
    # Extract model outputs
    # -----------------------------
    # This part may need adjustment depending on the exact TwinLiteNet+ code.
    # Common multi-task style:
    #     outputs[0] = drivable area prediction
    #     outputs[1] = lane line prediction
    drivable_output = outputs[0]
    lane_output = outputs[1]

    # Remove batch dimension.
    drivable_output = drivable_output[0]
    lane_output = lane_output[0]

    # -----------------------------
    # Convert predictions into masks
    # -----------------------------
    def output_to_binary_mask(output):
        """
        Converts a TwinLiteNet+ output tensor into a binary OpenCV mask.
        """

        # Case 1:
        # Output shape is [classes, height, width]
        # Use argmax to select the most likely class at each pixel.
        if len(output.shape) == 3 and output.shape[0] > 1:
            class_map = torch.argmax(output, dim=0)
            mask = class_map.cpu().numpy().astype(np.uint8)

            # Assume class 1 means foreground.
            mask[mask > 0] = 255

        # Case 2:
        # Output shape is [1, height, width]
        # Use sigmoid + threshold.
        elif len(output.shape) == 3 and output.shape[0] == 1:
            probability = torch.sigmoid(output[0])
            mask = (probability > 0.5).cpu().numpy().astype(np.uint8) * 255

        # Case 3:
        # Output shape is already [height, width]
        else:
            probability = torch.sigmoid(output)
            mask = (probability > 0.5).cpu().numpy().astype(np.uint8) * 255

        return mask

    drivable_mask = output_to_binary_mask(drivable_output)
    lane_mask = output_to_binary_mask(lane_output)

    # Resize masks back to the original image size.
    drivable_mask = cv2.resize(
        drivable_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    lane_mask = cv2.resize(
        lane_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert single-channel masks to 3-channel BGR masks
    # so they work with your map_masks_and_boxes_to_image() function.
    drivable_mask_bgr = cv2.cvtColor(drivable_mask, cv2.COLOR_GRAY2BGR)
    lane_mask_bgr = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

    return drivable_mask_bgr, lane_mask_bgr

def segment_lanes_ufldv2(
    lane_model,
    image,
    thickness=6,
    return_points=False
):
    """
    Runs Ultra Fast Lane Detection V2 and returns a lane mask.

    Args:
        lane_model: Loaded UFLDv2LaneDetector model.
        image: OpenCV BGR image.
        thickness: Thickness of the lane lines drawn into the mask.
        return_points: If True, also return the raw lane point coordinates.

    Returns:
        If return_points == False:
            lane_mask_bgr

        If return_points == True:
            lane_mask_bgr, lane_points

        lane_mask_bgr is a 3-channel BGR image where:
            - lane pixels are white
            - non-lane pixels are black
    """

    lane_points = lane_model.predict(image)

    lane_mask = np.zeros(
        (image.shape[0], image.shape[1]),
        dtype=np.uint8
    )

    for lane in lane_points:
        if len(lane) < 2:
            continue

        sorted_points = sorted(
            lane,
            key=lambda point: (point[1], point[0])
        )

        points_array = np.array(
            sorted_points,
            dtype=np.int32
        )

        cv2.polylines(
            lane_mask,
            [points_array],
            isClosed=False,
            color=255,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    lane_mask_bgr = cv2.cvtColor(
        lane_mask,
        cv2.COLOR_GRAY2BGR
    )

    if return_points:
        return lane_mask_bgr, lane_points

    return lane_mask_bgr

# ========================================
# ========== Object Detection ============
# ========================================
def object_detection(image, model, confidence_threshold=0.4):
    """
    Runs YOLO object detection on an image and returns bounding box data.

    Args:
        image: OpenCV BGR image.
        model: Loaded YOLO model.
        confidence_threshold: Minimum confidence required to keep detection.

    Returns:
        A list of dictionaries. Each dictionary contains:
            - class_id: YOLO class ID number
            - class_name: YOLO class name
            - confidence: Detection confidence score
            - box: Bounding box location as [x1, y1, x2, y2]

        x1, y1 = top-left corner
        x2, y2 = bottom-right corner

    Example:
        detections = object_detection(image_resized, yolo_model)

        for detection in detections:
            print(detection["class_name"], detection["box"])
    """

    results = model(image, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if confidence < confidence_threshold:
                continue

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            }

            detections.append(detection)

    return detections


# ========== Organize Miscelaneous ==========
def segment_objects(
    model,
    image,
    segmentation_type="instance"
):
    """
    Runs YOLO segmentation and returns segmentation mask(s).

    Args:
        model: Loaded YOLO segmentation model.
        image: OpenCV BGR image.
        segmentation_type:
            "instance" = return one mask per detected object.
            "semantic" = return one combined mask.

    Returns:
        If segmentation_type == "instance":
            List of binary BGR masks.
            Each mask corresponds to one detected object.

        If segmentation_type == "semantic":
            One combined binary BGR mask.

    Example:
        # Instance segmentation
        masks = segment_objects(
            segmentation_model,
            image,
            segmentation_type="instance"
        )

        # Semantic segmentation
        mask = segment_objects(
            segmentation_model,
            image,
            segmentation_type="semantic"
        )
    """

    results = model(image, verbose=False)

    # No objects detected
    if len(results) == 0 or results[0].masks is None:
        if segmentation_type == "instance":
            return []

        return np.zeros_like(image)

    masks = results[0].masks.data.cpu().numpy()

    # ---------------------------------
    # Instance segmentation
    # ---------------------------------
    if segmentation_type == "instance":

        instance_masks = []

        for mask in masks:

            mask = (mask * 255).astype(np.uint8)

            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0])
            )

            # Convert grayscale mask to BGR
            mask = cv2.cvtColor(
                mask,
                cv2.COLOR_GRAY2BGR
            )

            instance_masks.append(mask)

        return instance_masks

    # ---------------------------------
    # Semantic segmentation
    # ---------------------------------
    elif segmentation_type == "semantic":

        semantic_mask = np.zeros(
            (image.shape[0], image.shape[1]),
            dtype=np.uint8
        )

        for mask in masks:

            mask = (mask * 255).astype(np.uint8)

            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0])
            )

            semantic_mask = cv2.bitwise_or(
                semantic_mask,
                mask
            )

        return cv2.cvtColor(
            semantic_mask,
            cv2.COLOR_GRAY2BGR
        )

    else:
        raise ValueError(
            "segmentation_type must be 'instance' or 'semantic'."
        )
