import copy
import json
import math
import os
import random
import re
import ast
from typing import Dict

import torch
import transformers
import yaml
from qwen_vl_utils import smart_resize, process_vision_info
from torch.utils.data import Dataset
from PIL import Image

from gui_actor.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    ACTION_PATTENS_XY,
    ADDITIONAL_SPECIAL_TOKENS,
    assistant_template,
    chat_template,
    grounding_system_message,
)
from gui_actor.trainer import rank0_print
from .crop import crop_image_for_training

def reformat_coordinates(text):
    """
    (1) Find all the coordinates in the text.
    (2) Replace the coordinates with the special tokens.
    (3) Return the new text and the coordinates as a list of (x, y), where x in [0, 1] and y in [0, 1].
    """
    epsilon = 0.001
    def adjust_coord(c):
        """
        Adjust coordinate if it is too close to 0 or 1.
        """
        if abs(c) < epsilon:
            return epsilon
        elif abs(c - 1) < epsilon:
            return 1 - epsilon
        return c

    all_matches = []
    for pattern in ACTION_PATTENS_XY:
        matches = list(re.finditer(pattern, text))
        for match in matches:
            all_matches.append((match.start(), match.groups()))
        if pattern == ACTION_PATTENS_XY[0]:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        else:
            target_text = f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}, {DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
        text = re.sub(
            pattern,
            target_text,
            text
        )
    
    coordinates = []
    all_matches.sort(key=lambda x: x[0])
    # Extract coordinates in order
    for _, groups in all_matches:
        # When two coordinate values are found, parse them as one (x, y) pair.
        if len(groups) == 2:
            x_str, y_str = groups
            x = adjust_coord(ast.literal_eval(x_str))
            y = adjust_coord(ast.literal_eval(y_str))
            coordinates.append((x, y))
        # When four coordinate values are found, parse them as two pairs.
        elif len(groups) == 4:
            x1_str, y1_str, x2_str, y2_str = groups
            x1 = adjust_coord(ast.literal_eval(x1_str))
            y1 = adjust_coord(ast.literal_eval(y1_str))
            x2 = adjust_coord(ast.literal_eval(x2_str))
            y2 = adjust_coord(ast.literal_eval(y2_str))
            coordinates.append((x1, y1))
            coordinates.append((x2, y2))
    
    return text, coordinates

def get_token_index(image_processor, image, point_x, point_y):
    """
    Get the index of the visual token that contains the point (x, y).
    Args:
        image_processor: the image processor
        image: the image in PIL format
        point_x: the x coordinate of the point, in [0, 1].
        point_y: the y coordinate of the point, in [0, 1].
    """
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")
    
    # get the original image size and the resized image size
    image = image[0]
    w, h = image.size
    px, py = w * point_x, h * point_y
    # rank0_print(f"px: {px}, py: {py}")
    # get the token index
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = math.floor(px / merge_patch_size)
    y_index = math.floor(py / merge_patch_size)
    
    visual_token_index = y_index * (w // merge_patch_size) + x_index

    # merge all above print into one line
    return visual_token_index

def get_multi_patch_labels(image_processor, image, bbox_gt):
    """
    Get the multi-patch labels for the bounding box.
    Args:
        image_processor: the image processor
        image: the image in PIL format
        bbox_gt: the bounding box in the format of (x_min, y_min, x_max, y_max) [0,1]
    """
    if not image or not bbox_gt:
        return torch.zeros(0)
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")

    # Get the original image size and the resized image size
    image = image[0]
    w, h = image.size

    bbox_gt_abs = [bbox_gt[0]*w, bbox_gt[1]*h, bbox_gt[2]*w, bbox_gt[3]*h]
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox_gt_abs
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    if w % merge_patch_size != 0 or h % merge_patch_size != 0:
        raise ValueError(f"Image size {w}x{h} is not divisible by merge_patch_size {merge_patch_size}")

    grid_h, grid_w = h // merge_patch_size, w // merge_patch_size

    binary_mask = torch.zeros(grid_h * grid_w)
    # Iterate through all patches, check if they overlap with the bounding box
    for y_idx in range(grid_h):
        for x_idx in range(grid_w):
            # Calculate patch boundaries
            patch_x_min = x_idx * merge_patch_size
            patch_y_min = y_idx * merge_patch_size
            patch_x_max = patch_x_min + merge_patch_size
            patch_y_max = patch_y_min + merge_patch_size
            
            # Check if patch overlaps with the bounding box
            if not (patch_x_max <= x_min or patch_x_min >= x_max or 
                    patch_y_max <= y_min or patch_y_min >= y_max):
                # Calculate patch index in the flattened grid
                patch_idx = y_idx * grid_w + x_idx
                binary_mask[patch_idx] = 1

    return binary_mask

def token_index_to_coordinates(image_processor, visual_token_index, image_width, image_height):
    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    x_index = visual_token_index % (image_width // merge_patch_size)
    y_index = visual_token_index // (image_width // merge_patch_size)
    px = x_index * merge_patch_size + merge_patch_size / 2
    py = y_index * merge_patch_size + merge_patch_size / 2
    return px, py

class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        data_path: str,
        data_args,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.list_data_dict = []
        self.list_image_path = []
        self.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
        self.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        self.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]

        # Data loading logic from backup...
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path) as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path) as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    images_folder = dataset.get("images_folder")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path) as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path) as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%"[0]) * len(cur_data_dict) / 100))
                        else:
                            sampling_number = int(sampling_number)

                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
                    self.list_image_path.extend([images_folder] * len(cur_data_dict))
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path) as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)
                self.list_image_path.extend([""] * len(cur_data_dict))

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = (
                1200 * len(sample["image"]) if isinstance(sample["image"], list) else 1200 if "image" in sample else 0
            )
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"

            img_tokens = (
                1200 * len(sample["image"]) if isinstance(sample["image"], list) else 1200 if "image" in sample else 0
            )

            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len + img_tokens)
            else:
                length_list.append(-cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        for _ in range(3): # Retry up to 3 times
            try:
                sample = self._get_item(i)
                if sample is not None:
                    return sample
            except Exception as e:
                rank0_print(f"Error processing sample {i}, trying next. Error: {e}")
                i = random.randint(0, len(self.list_data_dict) - 1)
        
        rank0_print(f"Failed to process sample {i} after multiple retries, getting a random one.")
        return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        image_path_prefix = os.path.join(self.data_args.image_folder, self.list_image_path[i])

        if "image" in sources:
            image_file = self.list_data_dict[i]["image"]
            image_paths = [os.path.join(image_path_prefix, f) for f in image_file] if isinstance(image_file, list) else [os.path.join(image_path_prefix, image_file)]
            sources = copy.deepcopy(sources["conversations"])
        else:
            image_paths = []
            sources = copy.deepcopy(sources["conversations"])

        item_id = self.list_data_dict[i].get("id", i)

        data_dict = self.preprocess_qwen2vl(sources, self.tokenizer, self.processor, image_paths, id=item_id)
        
        if isinstance(i, int):
            # Flatten the batched outputs from preprocess_qwen2vl
            data_dict = {
                key: val[0] if isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor) else val
                for key, val in data_dict.items()
            }

        data_dict["id"] = item_id

        if data_dict.get("pixel_values") is not None:
            n_image_tokens = sum(
                grid[1] * grid[2]
                for grid in data_dict["image_grid_thw"]
            )
            if (len(data_dict["input_ids"]) + n_image_tokens) > self.tokenizer.model_max_length:
                rank0_print(f"=== Removed data_dict {i} because it is longer than the model_max_length: {len(data_dict['input_ids'])} + {n_image_tokens} > {self.tokenizer.model_max_length}")
                return None

        return data_dict

    def preprocess_qwen2vl(
        self,
        source, # conversations
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        image_paths: list, # Now a list of paths
        system_message: str = grounding_system_message,
        agent_mode: bool = True,
        chat_template: str = chat_template,
        assistant_template: str = assistant_template,
        id: int = None,
    ) -> Dict:
        
        original_image_pil = None
        new_data_to_return = {
            "sub_image_offsets": [],
            "sub_image_gt_bboxes": [],
            "original_gt_bboxes": [],
            "has_sub_image": [],
        }

        roles = {"human": "user", "gpt": "assistant", "system": "system"}
        processor.tokenizer = tokenizer

        pixel_values_list, image_grid_thw_list = [], []
        input_id, target = [], []
        coordinates, visual_token_indices_of_coordinates, multi_patch_labels = [], [], []
        
        image_path_index = 0
        processed_pil_images = []

        if roles.get(source[0]["from"]) == "system":
            system_message = source[0]["value"]
            source = source[1:self.data_args.max_conv_turns]

        system_input_id = tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": [{"type": "text", "text": system_message}]}],
            chat_template=chat_template,
        )
        input_id += system_input_id
        target += [IGNORE_INDEX] * len(system_input_id)

        for conv_original in source:
            conv = copy.deepcopy(conv_original)

            try:
                role = conv["role"]
                content = conv["content"]
            except Exception:
                role = conv["from"]
                content = conv["value"]
            role = roles.get(role, role)

            image_count = content.count(DEFAULT_IMAGE_TOKEN)
            proc_conv = conv # Start with the original conv

            # --- USER TURN --- 
            if image_count > 0 and role == "user":
                image_placeholders = []
                if image_path_index < len(image_paths):
                    try:
                        img_path = image_paths[image_path_index]
                        pil_img = Image.open(img_path).convert("RGB")
                        original_image_pil = pil_img # Store original image
                        image_placeholders.append({
                            "type": "image",
                            "image": img_path,
                            "min_pixels": self.processor.image_processor.min_pixels,
                            "max_pixels": self.processor.image_processor.max_pixels,
                        })
                        image_path_index += 1
                    except Exception as e:
                        rank0_print(f"Could not load image {img_path}: {e}")
                
                text_content = content.replace(DEFAULT_IMAGE_TOKEN, "")
                proc_conv = {"role": role, "content": image_placeholders + [{"type": "text", "text": text_content}]}

            # --- ASSISTANT TURN --- 
            elif role == "assistant":
                bbox_gt_val = conv.get("bbox_gt")
                text_content = content
                sub_image_generated = False

                if original_image_pil and bbox_gt_val:
                    crop_info = crop_image_for_training(original_image_pil, bbox_gt_val)
                    if crop_info:
                        sub_image_pil = crop_info["sub_image"]
                        assistant_prefix = "Based on the cropped sub-image, I will now provide a more precise solution. "
                        text_content = f'{assistant_prefix}{DEFAULT_IMAGE_TOKEN}\n{text_content}'
                        
                        sub_image_placeholder = {
                            "type": "image",
                            "image": sub_image_pil,
                            "min_pixels": self.processor.image_processor.min_pixels,
                            "max_pixels": self.processor.image_processor.max_pixels,
                        }

                        proc_conv = {
                            "role": role, 
                            "content": [sub_image_placeholder, {"type": "text", "text": text_content}],
                            "recipient": conv.get("recipient", "os"),
                        }
                        
                        conv["bbox_gt"] = crop_info["sub_image_gt_bbox"]

                        new_data_to_return["sub_image_offsets"].append(crop_info["offset"])
                        new_data_to_return["sub_image_gt_bboxes"].append(crop_info["sub_image_gt_bbox"])
                        new_data_to_return["original_gt_bboxes"].append(crop_info["original_gt_bbox"])
                        new_data_to_return["has_sub_image"].append(True)
                        sub_image_generated = True
                
                if not sub_image_generated:
                    new_data_to_return["has_sub_image"].append(False)
                    proc_conv = {"role": role, "content": [{"type": "text", "text": text_content}]}

            # --- UNIFIED PROCESSING PER TURN (FROM BACKUP) ---
            image_inputs, _ = process_vision_info([proc_conv])
            processed_pil_images.extend(image_inputs)

            templated_conv = tokenizer.apply_chat_template(
                conversation=[proc_conv],
                chat_template=assistant_template if role == 'assistant' else chat_template,
                tokenize=False,
            )
            inputs = processor(text=[templated_conv], images=image_inputs, return_tensors="pt")

            if image_inputs:
                pixel_values_list.append(inputs["pixel_values"])
                image_grid_thw_list.append(inputs["image_grid_thw"])

            if role == 'assistant' and proc_conv.get("recipient") == "os" and conv.get("bbox_gt"):
                text_for_reformat, coord = reformat_coordinates(templated_conv)
                coordinates.extend(coord)
                if image_inputs:
                    patch_mask = get_multi_patch_labels(
                        processor.image_processor,
                        [image_inputs[-1]], # Use the last processed image (sub-image if present)
                        conv["bbox_gt"]
                    )  
                    multi_patch_labels.append(patch_mask)

            encode_id = inputs.input_ids[0].tolist()
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        # --- FINAL ASSEMBLY ---
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        target = [IGNORE_INDEX if token == self.pointer_end_token_id else token for token in target]

        data_dict = {
            "input_ids": [torch.tensor(input_id, dtype=torch.long)],
            "labels": [torch.tensor(target, dtype=torch.long)],
            "coordinates": [coordinates] if coordinates else [[]],
            "visual_token_indices_of_coordinates": [visual_token_indices_of_coordinates] if visual_token_indices_of_coordinates else [[]],
            "multi_patch_labels": [torch.stack(multi_patch_labels)] if multi_patch_labels else [None],
        }

        if pixel_values_list:
            data_dict["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            data_dict["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
        
        for key, val in new_data_to_return.items():
            if val:
                dtype = torch.bool if key == 'has_sub_image' else torch.float32
                data_dict[key] = [torch.tensor(val, dtype=dtype)]
            else:
                if 'bboxes' in key: empty_tensor = torch.empty(0, 4)
                elif 'offsets' in key: empty_tensor = torch.empty(0, 2)
                else: empty_tensor = torch.empty(0, dtype=torch.bool)
                data_dict[key] = [empty_tensor]
        
        return data_dict