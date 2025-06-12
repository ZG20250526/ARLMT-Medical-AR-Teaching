import json
import os

# 定义 JSON 文件的路径
json_input_path = "/data0/GYF-projects/LLaVA-main/LLaVA-main/playground/llava_v1_6_mix665k+med.json"
json_output_path = "/data0/GYF-projects/LLaVA-main/LLaVA-main/playground/llava_v1_6_mix665k+med2024090404.json"
output_txt_path = "/data0/GYF-projects/LLaVA-main/LLaVA-main/playground/llava_v1_6_mix665k+med2024090404中没提到image中存在的图片名称2024073002.tx"

# 获取所有子文件夹中的图像文件名
image_files = set()
root_folder = "/data0/GYF-projects/LLaVA-main/LLaVA-main/playground/data"
subfolders = ["med12w","coco/train2017", "gqa/images", "ocr_vqa/images", "textvqa/train_images", "vg/VG_100K", "vg/VG_100K_2"]
for subfolder in subfolders:
    folder_path = os.path.join(root_folder, subfolder)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            image_files.add(os.path.join(subfolder, filename))

# 读取原始 JSON 数据
with open(json_input_path, 'r') as f:
    data = json.load(f)

# 过滤出存在对应图像文件的条目
filtered_data = [item for item in data if 'image' in item and item['image'] in image_files]

# 保存过滤后的数据到新的 JSON 文件
with open(json_output_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"Filtered data has been saved to {json_output_path}")

# 从 JSON 数据中提取所有图片的名称
json_images = set(item['image'] for item in data if 'image' in item)

# 计算差集：在图像文件夹中但不在 JSON 数据中的图片
missing_images = image_files - json_images

# 将缺失的图片名称保存到文本文件中
with open(output_txt_path, 'w') as f:
    for img in missing_images:
        f.write(img + '\n')

print(f"Missing images have been saved to {output_txt_path}")
