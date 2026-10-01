import base64
from app.shared.runtime.logger import logger,PROJECT_ROOT


# ====================== 【你只需要改这里】填写你的图片路径 ======================
IMAGE_PATH = PROJECT_ROOT / "images" / "default.jpeg"  # 改成你的图片：a.jpg / logo.png 等
COPY_IMAGE_PATH = PROJECT_ROOT / "images" / "default_copy.jpeg"  # 改成你的图片：a.jpg / logo.png 等
# ==============================================================================

# -------------------- 1. 图片 → Base64 字符串（编码） --------------------
print("正在把图片转 Base64...")
with open(IMAGE_PATH, "rb") as image_file:
    # 读取图片二进制 → 转 base64 → 转字符串
    base64_string = base64.b64encode(image_file.read()).decode("utf-8")
print("✅ 图片转 Base64 完成！")
print("Base64 字符串前50个字符：", base64_string[:50], "...")


# -------------------- 2. Base64 字符串 → 还原成图片（解码） --------------------
print("\n正在把 Base64 转回图片...")
output_image_path = COPY_IMAGE_PATH
# 解码 Base64 回到二进制
image_binary = base64.b64decode(base64_string)
# 写入文件
with open(output_image_path, "wb") as f:
    f.write(image_binary)