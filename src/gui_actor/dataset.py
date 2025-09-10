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
    if len(image) != 1:
        raise ValueError(f"Expected 1 image, got {len(image)}")

    # Get the original image size and the resized image size
    image = image[0]
    w, h = image.size

    bbox_gt = [bbox_gt[0]*w, bbox_gt[1]*h, bbox_gt[2]*w, bbox_gt[3]*h]
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox_gt
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    merge_patch_size = image_processor.patch_size * image_processor.merge_size
    assert w % merge_patch_size == 0 and h % merge_patch_size == 0, f"Image size {w}x{h} is not divisible by merge_patch_size {merge_patch_size}"
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

        # Handle multiple JSON files specified in the data_path
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
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
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
                        # NOTE: we only use json_path with .json now
                        # Handle the images_folder in yaml
                        with open(json_path) as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
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
                self.list_image_path.extend([""] * len(cur_data_dict))  # NOTE: the image subfolder is empty...

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
        sample = self._get_item(i)
        if sample is None:
            new_index = random.randint(0, len(self.list_data_dict) - 1)
            return self.__getitem__(new_index)
        else:
            return sample
        try:
            sample = self._get_item(i)
            if sample is None:
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                return self.__getitem__(new_index)
        except Exception as e:
            print(f"Failed to fetch sample {i}. Exception:", e)
            new_index = random.randint(0, len(self.list_data_dict) - 1)
            return self.__getitem__(new_index)
        return sample

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        image_path = os.path.join(self.data_args.image_folder, self.list_image_path[i])

        if "image" in sources:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image_list = [os.path.join(image_path, image_file) for image_file in image_file]
            else:
                image_list = [os.path.join(image_path, image_file)]

            sources = copy.deepcopy(sources["conversations"])
        elif "video" in sources:
            raise NotImplementedError("Video is not supported for Qwen2VL")
        else:
            sources = copy.deepcopy(sources["conversations"])

        item_id = self.list_data_dict[i].get("id", i)

        data_dict = self.preprocess_qwen2vl(sources, self.tokenizer, self.processor, image_list, id=item_id)
        if isinstance(i, int):
            data_dict = {
                "input_ids": data_dict["input_ids"][0],
                "labels": data_dict["labels"][0],
                "coordinates": data_dict["coordinates"][0],
                "visual_token_indices_of_coordinates": data_dict["visual_token_indices_of_coordinates"][0],
                "pixel_values": data_dict["pixel_values"],
                "image_grid_thw": data_dict["image_grid_thw"],
                "multi_patch_labels": data_dict["multi_patch_labels"][0],   # add multi_patch_labels                
            }

        data_dict["id"] = item_id

        # return None if the input_ids is longer than the model_max_length
        n_image_tokens = (
            data_dict["image_grid_thw"][0][0] * 
            data_dict["image_grid_thw"][0][1] * 
            data_dict["image_grid_thw"][0][2] / 
            self.processor.image_processor.merge_size / 
            self.processor.image_processor.merge_size
        )
        if (len(data_dict["input_ids"]) + n_image_tokens) > self.tokenizer.model_max_length:
            rank0_print(f"=== Removed data_dict {i} because it is longer than the model_max_length: {len(data_dict['input_ids'])} + {n_image_tokens} > {self.tokenizer.model_max_length}")
            return None

        return data_dict

    def preprocess_qwen2vl(
        self,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        image_list_path: list, # This is now a list of paths
        **kwargs,
    ) -> Dict:
        from .crop import crop_image_for_training
        from PIL import Image

        # Initialization of data collection lists
        all_texts = []
        all_instruction_ids = []
        all_images = []
        all_sub_image_offsets = []
        all_sub_image_gt_bboxes = []
        all_original_gt_bboxes = []
        has_sub_image = []

        original_image = None
        original_gt_bbox = None

        # Process system message first if it exists
        if sources[0]['from'].lower() == 'system':
            system_message = sources[0]['value']
            sources = sources[1:]
        else:
            system_message = grounding_system_message
        
        system_text = f"system\n{system_message}"
        all_texts.append(system_text)
        all_instruction_ids.append(0) # System message is not part of the loss

        # Process conversation turns
        for i, sentence in enumerate(sources):
            sentence_from = sentence["from"].lower()
            sentence_value = sentence["value"]

            # ---------------- Human Turn ----------------
            if sentence_from == "human":
                if DEFAULT_IMAGE_TOKEN in sentence_value and image_list_path:
                    # Assuming the first image in the list corresponds to the human's turn
                    image_path = image_list_path.pop(0)
                    try:
                        image = Image.open(image_path).convert("RGB")
                        original_image = image
                        
                        image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                        all_images.append(image_tensor)
                        
                        # The <image> token is kept in the text for the tokenizer
                    except Exception as e:
                        rank0_print(f"Error loading image {image_path}: {e}")
                        sentence_value = sentence_value.replace(DEFAULT_IMAGE_TOKEN, "").strip()
                
                text = f"user\n{sentence_value}"
                instruction_id = 0

            # ---------------- GPT/Assistant Turn ----------------
            else:
                try:
                    content_data = json.loads(sentence_value)
                    original_gt_bbox = content_data.get("bbox_gt")
                    sentence_value = content_data.get("text", "")
                except (json.JSONDecodeError, TypeError):
                    original_gt_bbox = None

                if original_image and original_gt_bbox:
                    crop_info = crop_image_for_training(original_image, original_gt_bbox)
                    
                    if crop_info:
                        all_sub_image_offsets.append(crop_info["offset"])
                        all_sub_image_gt_bboxes.append(crop_info["sub_image_gt_bbox"])
                        all_original_gt_bboxes.append(crop_info["original_gt_bbox"])
                        has_sub_image.append(True)

                        sub_image_tensor = processor.preprocess(crop_info["sub_image"], return_tensors="pt")["pixel_values"][0]
                        all_images.append(sub_image_tensor)

                        # Generate multi-patch labels for the sub-image to maintain output format consistency
                        patch_mask = get_multi_patch_labels(
                            processor.image_processor,
                            [crop_info["sub_image"]],  # Pass sub-image
                            crop_info["sub_image_gt_bbox"] # Pass sub-image's GT Bbox
                        )
                        all_multi_patch_labels.append(patch_mask)

                        assistant_prefix = "Based on the cropped sub-image, I will now provide a more precise solution. "
                        sentence_value = f"{assistant_prefix}{DEFAULT_IMAGE_TOKEN}\n{sentence_value}"
                    else:
                        has_sub_image.append(False)
                else:
                    has_sub_image.append(False)

                text = f"assistant\n{sentence_value}"
                instruction_id = 1

            all_texts.append(text)
            all_instruction_ids.append(instruction_id)

        # --- Final Tokenization and Label Creation ---
        full_text = "\n".join(all_texts)
        
        # Replace all <image> tokens with the actual image token for the tokenizer
        full_text = full_text.replace(DEFAULT_IMAGE_TOKEN, tokenizer.image_token)
        
        input_ids = tokenizer(full_text, return_tensors="pt", padding=False).input_ids[0]
        labels = input_ids.clone()
        
        # Mask out non-assistant parts for loss calculation
        tokenized_parts = [tokenizer(text.replace(DEFAULT_IMAGE_TOKEN, tokenizer.image_token), return_tensors="pt", padding=False).input_ids[0] for text in all_texts] 
        
        current_pos = 0
        for i, part_ids in enumerate(tokenized_parts):
            if all_instruction_ids[i] == 0:  # Mask system and user inputs
                labels[current_pos : current_pos + len(part_ids)] = IGNORE_INDEX
            current_pos += len(part_ids)

        # --- Final Data Dictionary Construction ---
        pixel_values = torch.stack(all_images) if all_images else None

        # The rest of the data_dict items are handled in __getitem__ after this function returns
        # We return a more complete dict here to be pruned later
        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            sub_image_offsets=torch.tensor(all_sub_image_offsets, dtype=torch.float32) if all_sub_image_offsets else torch.empty(0, 2),
            sub_image_gt_bboxes=torch.tensor(all_sub_image_gt_bboxes, dtype=torch.float32) if all_sub_image_gt_bboxes else torch.empty(0, 4),
            original_gt_bboxes=torch.tensor(all_original_gt_bboxes, dtype=torch.float32) if all_original_gt_bboxes else torch.empty(0, 4),
            has_sub_image=torch.tensor(has_sub_image, dtype=torch.bool) if has_sub_image else torch.empty(0, dtype=torch.bool),
            # These are placeholders as the original function had them, they might not be needed with the new logic
            coordinates=[[]],
            visual_token_indices_of_coordinates=[[]],
            image_grid_thw=None, # This is specific to Qwen1.5VL processor, might not be needed
            multi_patch_labels=[[]],
        )
        
        # Wrap tensors in a list to match the expected output format of the original __getitem__
        for key in ['input_ids', 'labels', 'sub_image_offsets', 'sub_image_gt_bboxes', 'original_gt_bboxes', 'has_sub_image']:
            if data_dict[key] is not None:
                data_dict[key] = [data_dict[key]]

        return data_dict
