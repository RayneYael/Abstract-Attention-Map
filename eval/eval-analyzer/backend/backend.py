from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)  # 允许前端调用

@torch.inference_mode()
def get_attn_map(image, attn_scores, n_width, n_height):
    """你的原始注意力图生成函数"""
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

@app.route('/api/attention_heatmap', methods=['POST'])
def generate_attention_heatmap():
    try:
        data = request.json
        
        # 从base64转换为PIL图像
        image_data = base64.b64decode(data['image_data'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # 生成注意力图
        attn_scores = data['attn_scores']
        n_width = data['n_width']
        n_height = data['n_height']
        
        blended_image = get_attn_map(image, attn_scores, n_width, n_height)
        
        # 转换为base64返回
        buffer = io.BytesIO()
        blended_image.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({'heatmap': f'data:image/png;base64,{result_base64}'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)