
import random
from PIL import Image
import numpy as np

import base64
from io import BytesIO

def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return f"data:image;base64,{base64.b64encode(buffer.getvalue()).decode()}"

# 从 run_crop.sh 和 crop.py 脚本中提取的硬编码参数
SCALE_FACTOR = 0.15
MODE = "center"
RANDOMIZE_ASPECT_RATIO = True

def _calculate_crop_dims(scale_factor, bbox_width, bbox_height, img_width, img_height, randomize_aspect_ratio):
    """根据策略计算最终的裁剪尺寸。"""
    # 步骤1: 根据scale插值计算基础尺寸
    base_crop_width = int(bbox_width + scale_factor * (img_width - bbox_width))
    base_crop_height = int(bbox_height + scale_factor * (img_height - bbox_height))

    if not randomize_aspect_ratio:
        return base_crop_width, base_crop_height

    # 步骤2: 如果激活随机长宽比，则进行调整
    if base_crop_width > base_crop_height:
        long_side_dim, short_side_dim = base_crop_width, base_crop_height
        bbox_long_side_dim = bbox_width
    else:
        long_side_dim, short_side_dim = base_crop_height, base_crop_width
        bbox_long_side_dim = bbox_height

    # 步骤3: 确定随机化的下限，确保不小于bbox的对应边
    lower_bound = max(short_side_dim, bbox_long_side_dim)
    upper_bound = long_side_dim

    if lower_bound >= upper_bound:
        return base_crop_width, base_crop_height

    # 步骤4: 使用二次方随机函数，在安全范围内生成新长边
    random_scale = random.random() ** 2
    new_long_side = lower_bound + (upper_bound - lower_bound) * random_scale

    if base_crop_width > base_crop_height:
        final_crop_width, final_crop_height = int(new_long_side), int(short_side_dim)
    else:
        final_crop_width, final_crop_height = int(short_side_dim), int(new_long_side)
    
    return final_crop_width, final_crop_height

def _get_crop_coordinates(bbox, crop_width, crop_height, img_width, img_height, mode):
    """计算裁剪区域的左上角坐标 (crop_x, crop_y)。"""
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    min_crop_x = max(0, x2 - crop_width)
    max_crop_x = min(x1, img_width - crop_width)
    
    min_crop_y = max(0, y2 - crop_height)
    max_crop_y = min(y1, img_height - crop_height)

    if max_crop_x < min_crop_x or max_crop_y < min_crop_y:
        crop_x = max(0, x1 - (crop_width - bbox_width) // 2)
        crop_y = max(0, y1 - (crop_height - bbox_height) // 2)
        if crop_x + crop_width > img_width:
            crop_x = img_width - crop_width
        if crop_y + crop_height > img_height:
            crop_y = img_height - crop_height
        return int(crop_x), int(crop_y)

    if mode == "random":
        crop_x = random.randint(min_crop_x, max_crop_x)
        crop_y = random.randint(min_crop_y, max_crop_y)
    elif mode == "center":
        ideal_x = x1 - (crop_width - bbox_width) / 2
        ideal_y = y1 - (crop_height - bbox_height) / 2
        sigma_x = (max_crop_x - min_crop_x) / 6 if max_crop_x > min_crop_x else 1
        sigma_y = (max_crop_y - min_crop_y) / 6 if max_crop_y > min_crop_y else 1
        crop_x = np.random.normal(loc=ideal_x, scale=sigma_x)
        crop_y = np.random.normal(loc=ideal_y, scale=sigma_y)
        crop_x = int(np.clip(crop_x, min_crop_x, max_crop_x))
        crop_y = int(np.clip(crop_y, min_crop_y, max_crop_y))
    else:
        raise ValueError(f"未知的模式: {mode}")

    return crop_x, crop_y

def crop_image_for_training(original_image: Image.Image, original_gt_bbox: list[int]) -> dict:
    """
    根据固定的训练参数裁剪图像。

    Args:
        original_image (Image.Image): 用户的原始图。
        original_gt_bbox (list[int]): 在原图坐标系下的 [x1, y1, x2, y2]。

    Returns:
        dict: 一个包含子图及相关信息的字典，如果无法处理则返回 None。
    """
    if not original_image or not original_gt_bbox:
        return None

    try:
        img_width, img_height = original_image.size
        x1 = int(original_gt_bbox[0] * img_width)
        y1 = int(original_gt_bbox[1] * img_height)
        x2 = int(original_gt_bbox[2] * img_width)
        y2 = int(original_gt_bbox[3] * img_height)
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width <= 0 or bbox_height <= 0:
            return None

        crop_width, crop_height = _calculate_crop_dims(
            SCALE_FACTOR, bbox_width, bbox_height, img_width, img_height, RANDOMIZE_ASPECT_RATIO
        )
        
        crop_width = min(crop_width, img_width)
        crop_height = min(crop_height, img_height)

        crop_x, crop_y = _get_crop_coordinates(
            (x1, y1, x2, y2), crop_width, crop_height, img_width, img_height, MODE
        )
        
        sub_image = original_image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
        
        offset = [crop_x, crop_y]
        
        sub_image_gt_bbox = [
            (x1 - crop_x)/crop_width,
            (y1 - crop_y)/crop_height,
            (x2 - crop_x)/crop_width,
            (y2 - crop_y)/crop_height,
        ]

        return (
            sub_image,
            pil_to_base64(sub_image),
            offset,
            sub_image_gt_bbox,
            original_gt_bbox
        )
    except Exception as e:
        print(f"Error during cropping: {e}")
        return None
