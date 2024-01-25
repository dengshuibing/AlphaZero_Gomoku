from PIL import Image

def convert_png_to_jpg(input_path, output_path):
    try:
        # 打开PNG图像
        image = Image.open(input_path)

        # 将PNG图像转换为JPEG格式（quality参数可选，设置图像质量）
        image = image.convert("RGB")
        image.save(output_path, "JPEG", quality=95)
        
        print(f"成功将 {input_path} 转换为 {output_path}")
    except Exception as e:
        print(f"转换失败: {e}")

# 输入和输出文件路径
input_png = "./res/black.png"
output_jpg = "./res/black.jpg"

# 调用转换函数
convert_png_to_jpg(input_png, output_jpg)
