# import base64, os
# # import spaces
# import json
# import torch
# import gradio as gr
# from typing import Optional
# from PIL import Image, ImageDraw
# import numpy as np
# import matplotlib.pyplot as plt
# from qwen_vl_utils import process_vision_info
# from datasets import load_dataset
# from transformers import AutoProcessor
# from gui_actor.constants import chat_template
# from gui_actor.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
# from gui_actor.inference import inference

# MAX_PIXELS = 3200 * 1800

# def resize_image(image, resize_to_pixels=MAX_PIXELS):
#     image_width, image_height = image.size
#     if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
#         resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
#         image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
#         image = image.resize((image_width_resized, image_height_resized))
#     return image

# # @spaces.GPU
# @torch.inference_mode()
# def draw_point(image: Image.Image, point: list, radius=8, color=(255, 0, 0, 128)):
#     overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
#     overlay_draw = ImageDraw.Draw(overlay)
#     x, y = point
#     overlay_draw.ellipse(
#         [(x - radius, y - radius), (x + radius, y + radius)],
#         outline=color,
#         width=5  # Adjust thickness as needed
#     )
#     image = image.convert('RGBA')
#     combined = Image.alpha_composite(image, overlay)
#     combined = combined.convert('RGB')
#     return combined

# # @spaces.GPU
# @torch.inference_mode()
# def get_attn_map(image, attn_scores, n_width, n_height):
#     w, h = image.size
#     scores = np.array(attn_scores[0]).reshape(n_height, n_width)

#     scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
#     # Resize score map to match image size
#     score_map = Image.fromarray((scores_norm * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST) # BILINEAR)
#     # Apply colormap
#     colormap = plt.get_cmap('jet')
#     colored_score_map = colormap(np.array(score_map) / 255.0)  # returns RGBA
#     colored_score_map = (colored_score_map[:, :, :3] * 255).astype(np.uint8)
#     colored_overlay = Image.fromarray(colored_score_map)

#     # Blend with original image
#     blended = Image.blend(image, colored_overlay, alpha=0.3)
#     return blended

# # load model
# if torch.cuda.is_available():
#     # os.system('pip install flash-attn --no-build-isolation')
#     model_name_or_path = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
#     print(f'load model: {model_name_or_path}')
#     data_processor = AutoProcessor.from_pretrained(model_name_or_path)
#     tokenizer = data_processor.tokenizer
#     model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
#         model_name_or_path,
#         torch_dtype=torch.bfloat16,
#         device_map="cuda:0",
#         attn_implementation="flash_attention_2"
#     ).eval()
# else:
#     model_name_or_path = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
#     data_processor = AutoProcessor.from_pretrained(model_name_or_path)
#     tokenizer = data_processor.tokenizer
#     model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
#         model_name_or_path,
#         torch_dtype=torch.bfloat16,
#         device_map="cpu"
#     ).eval()

# title = "GUI-Actor"
# header = """
# <div align="center">
#     <h1 style="padding-bottom: 10px; padding-top: 10px;">🎯 <strong>GUI-Actor</strong>: Coordinate-Free Visual Grounding for GUI Agents</h1>
#     <div style="padding-bottom: 10px; padding-top: 10px; font-size: 16px;">
#         Qianhui Wu*, Kanzhi Cheng*, Rui Yang*, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, Si Qin, Lars Liden<br>
#         Qingwei Lin, Huan Zhang, Tong Zhang, Jianbing Zhang, Dongmei Zhang, Jianfeng Gao<br/>
#     </div>
#     <div style="padding-bottom: 10px; padding-top: 10px; font-size: 16px;">
#         <a href="https://microsoft.github.io/GUI-Actor/">🌐 Project Page</a> | <a href="https://arxiv.org/abs/2403.12968">📄 arXiv Paper</a> | <a href="https://github.com/microsoft/GUI-Actor">💻 Github Repo</a><br/>
#     </div>
# </div>
# """

# theme = "soft"
# css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
#             #anno-img .mask.active {opacity: 0.7}"""

# # @spaces.GPU
# @torch.inference_mode()
# def process(image, instruction):
#     # resize image
#     w, h = image.size
#     if w * h > MAX_PIXELS:
#         image = resize_image(image)

#     conversation = [
#         {
#             "role": "system",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).",
#                 }
#             ]
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": image, # PIL.Image.Image or str to path
#                     # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
#                 },
#                 {
#                     "type": "text",
#                     "text": instruction,
#                 },
#             ],
#         },
#     ]

#     try:
#         pred = inference(conversation, model, tokenizer, data_processor, use_placeholder=True, topk=3)
#     except Exception as e:
#         print(e)
#         return image, f"Error: {e}", None
    
#     px, py = pred["topk_points"][0]
#     output_coord = f"({px:.4f}, {py:.4f})"
#     img_with_point = draw_point(image, (px * w, py * h))

#     n_width, n_height = pred["n_width"], pred["n_height"]
#     attn_scores = pred["attn_scores"]
#     att_map = get_attn_map(image, attn_scores, n_width, n_height)
    
#     return img_with_point, output_coord, att_map


# with gr.Blocks(title=title, css=css) as demo:
#     gr.Markdown(header)
#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(
#                 type='pil', label='Upload image')
#             # text box
#             input_instruction = gr.Textbox(label='Instruction', placeholder='Text your (low-level) instruction here')
#             submit_button = gr.Button(
#                 value='Submit', variant='primary')
#         with gr.Column():
#             image_with_point = gr.Image(type='pil', label='Image with Point (red circle)')
#             with gr.Accordion('Detailed prediction'):
#                 pred_xy = gr.Textbox(label='Predicted Coordinates', placeholder='(x, y)')
#                 att_map = gr.Image(type='pil', label='Attention Map')

#     submit_button.click(
#         fn=process,
#         inputs=[
#             input_image,
#             input_instruction
#         ],
#         outputs=[image_with_point, pred_xy, att_map]
#     )

# # demo.launch(debug=False, show_error=True, share=True)
# # demo.launch(share=True, server_port=7861, server_name='0.0.0.0')
# demo.queue().launch(share=False)

import base64, os
import json
import torch
import gradio as gr
from typing import Optional
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from transformers import AutoProcessor
import importlib
import sys

# 导入你的模块
from gui_actor.constants import chat_template
from gui_actor import modeling_qwen25vl  # 注意这里改为导入模块而不是类
from gui_actor.inference import inference

MAX_PIXELS = 3200 * 1800

def resize_image(image, resize_to_pixels=MAX_PIXELS):
    image_width, image_height = image.size
    if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
        resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
        image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
        image = image.resize((image_width_resized, image_height_resized))
    return image

@torch.inference_mode()
def draw_point(image: Image.Image, point: list, radius=8, color=(255, 0, 0, 128)):
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    x, y = point
    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        outline=color,
        width=5
    )
    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)
    combined = combined.convert('RGB')
    return combined

@torch.inference_mode()
def get_attn_map(image, attn_scores, n_width, n_height):
    w, h = image.size
    scores = np.array(attn_scores[0]).reshape(n_height, n_width)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    score_map = Image.fromarray((scores_norm * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
    colormap = plt.get_cmap('jet')
    colored_score_map = colormap(np.array(score_map) / 255.0)
    colored_score_map = (colored_score_map[:, :, :3] * 255).astype(np.uint8)
    colored_overlay = Image.fromarray(colored_score_map)
    blended = Image.blend(image, colored_overlay, alpha=0.3)
    return blended

# 全局变量存储模型，避免重复加载
MODEL_CACHE = {}

def load_model():
    """只在第一次调用或需要重载时加载模型"""
    if 'model' not in MODEL_CACHE or 'reload_requested' in MODEL_CACHE:
        print("Loading/Reloading model...")
        
        if torch.cuda.is_available():
            model_name_or_path = "microsoft/GUI-Actor-7B-Qwen2.5-VL"
            MODEL_CACHE['data_processor'] = AutoProcessor.from_pretrained(model_name_or_path)
            MODEL_CACHE['tokenizer'] = MODEL_CACHE['data_processor'].tokenizer
            MODEL_CACHE['model'] = modeling_qwen25vl.Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                attn_implementation="flash_attention_2"
            ).eval()
        else:
            model_name_or_path = "microsoft/GUI-Actor-3B-Qwen2.5-VL"
            MODEL_CACHE['data_processor'] = AutoProcessor.from_pretrained(model_name_or_path)
            MODEL_CACHE['tokenizer'] = MODEL_CACHE['data_processor'].tokenizer
            MODEL_CACHE['model'] = modeling_qwen25vl.Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            ).eval()
        
        # 清除重载标记
        MODEL_CACHE.pop('reload_requested', None)
        print("Model loaded successfully!")
    
    return MODEL_CACHE['model'], MODEL_CACHE['tokenizer'], MODEL_CACHE['data_processor']

def reload_model():
    """重载modeling模块和模型"""
    print("Reloading modeling_qwen25vl module...")
    importlib.reload(modeling_qwen25vl)
    MODEL_CACHE['reload_requested'] = True
    return "Model will be reloaded on next inference"

@torch.inference_mode()
def process(image, instruction, reload_checkbox=False):
    # 如果勾选了重载选项，先重载模块
    if reload_checkbox:
        importlib.reload(modeling_qwen25vl)
        MODEL_CACHE['reload_requested'] = True
    
    # 获取模型（如果需要会重新加载）
    model, tokenizer, data_processor = load_model()
    
    # resize image
    w, h = image.size
    if w * h > MAX_PIXELS:
        image = resize_image(image)

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).",
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": instruction,
                },
            ],
        },
    ]

    try:
        pred = inference(conversation, model, tokenizer, data_processor, use_placeholder=True, topk=3)
    except Exception as e:
        print(f"Inference error: {e}")
        return image, f"Error: {e}", None, False
    
    px, py = pred["topk_points"][0]
    output_coord = f"({px:.4f}, {py:.4f})"
    img_with_point = draw_point(image, (px * w, py * h))

    n_width, n_height = pred["n_width"], pred["n_height"]
    attn_scores = pred["attn_scores"]
    original_attn_scores = pred["original_attn_scores"]

    original_att_map = get_attn_map(image, original_attn_scores, n_width, n_height)
    att_map = get_attn_map(image, attn_scores, n_width, n_height)

    return img_with_point, original_att_map, att_map, output_coord, False  # 重置复选框

title = "GUI-Actor"
header = """
<div align="center">
    <h1 style="padding-bottom: 10px; padding-top: 10px;">🎯 <strong>GUI-Actor</strong>: Coordinate-Free Visual Grounding for GUI Agents</h1>
    <div style="padding-bottom: 10px; padding-top: 10px; font-size: 16px;">
        Qianhui Wu*, Kanzhi Cheng*, Rui Yang*, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, Si Qin, Lars Liden<br>
        Qingwei Lin, Huan Zhang, Tong Zhang, Jianbing Zhang, Dongmei Zhang, Jianfeng Gao<br/>
    </div>
    <div style="padding-bottom: 10px; padding-top: 10px; font-size: 16px;">
        <a href="https://microsoft.github.io/GUI-Actor/">🌐 Project Page</a> | <a href="https://arxiv.org/abs/2403.12968">📄 arXiv Paper</a> | <a href="https://github.com/microsoft/GUI-Actor">💻 Github Repo</a><br/>
    </div>
</div>
"""

theme = "soft"
css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
            #anno-img .mask.active {opacity: 0.7}"""

with gr.Blocks(title=title, css=css) as demo:
    gr.Markdown(header)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type='pil', label='Upload image')
            input_instruction = gr.Textbox(label='Instruction', placeholder='Text your (low-level) instruction here')
            reload_checkbox = gr.Checkbox(label='🔄 Reload model code (check this if you modified modeling_qwen25vl.py)', value=False)
            submit_button = gr.Button(value='Submit', variant='primary')
            reload_button = gr.Button(value='🔄 Manual Reload Model Code', variant='secondary')
        with gr.Column():
            image_with_point = gr.Image(type='pil', label='Image with Point (red circle)')
            with gr.Accordion('Detailed prediction'):
                original_att_map = gr.Image(type='pil', label='Original Attention Distribution Map')
                att_map = gr.Image(type='pil', label='Attention Map')
                pred_xy = gr.Textbox(label='Predicted Coordinates', placeholder='(x, y)')
            reload_status = gr.Textbox(label='Reload Status', value='Ready', interactive=False)

    submit_button.click(
        fn=process,
        inputs=[input_image, input_instruction, reload_checkbox],
        outputs=[image_with_point, original_att_map, att_map, pred_xy, reload_checkbox]
    )
    
    reload_button.click(
        fn=reload_model,
        outputs=[reload_status]
    )

if __name__ == "__main__":
    # 预先加载模型
    print("Pre-loading model...")
    load_model()
    print("Starting Gradio interface...")
    demo.queue().launch(share=False, server_port=7861, server_name='0.0.0.0')