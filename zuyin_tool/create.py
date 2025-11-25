import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import re

# === 設定 ===
input_file = "decompose_v1.txt"
font_path = "E:/python/lab_project/zuyin_tool/NotoSansCJK-Regular.ttc"
font_size = 80

generated_folder = "generated_images"
augmented_folder = "augmented_images"

tone = ['ˊ','ˇ','ˋ']
num_aug = 50  # 每個注音產生幾張

os.makedirs(generated_folder, exist_ok=True)
os.makedirs(augmented_folder, exist_ok=True)

# 字型載入
font = ImageFont.truetype(font_path, font_size)

# === 工具函式 ===
def sanitize_filename(text):
    """避免檔名錯誤"""
    #----
    # 將注音特殊符號替換為安全字元
    replacements = {
        'ˊ': '_2',  # 二聲
        'ˇ': '_3',  # 三聲
        'ˋ': '_4',  # 四聲
        '˙': '_0',  # 輕聲
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    #-----
    return re.sub(r'[\\/*?:"<>|]', '_', text)

# === 生成原始圖片 ===
def generate_text_image(text, save_path):
    after = ''
    if text[-1] in tone:
        for i in range(len(text)-2):
            after = after + text[i] + '\n'
        after = after + text[-2] + text[-1]
    elif text[0] != '˙':
        for i in range(len(text)-1):
            after = after + text[i] + '\n'
        after = after + text[-1]
    else:
        after = after + ' '
        for i in range(len(text)-1):
            after = after + text[i] + '\n'
        after = after + text[-1]

    img = Image.new("RGB", (150, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), after, fill="black", font=font)
    img.save(save_path)

# === 資料增強（自然手寫 + 灰底） ===
def augment_image(image):
    h, w = image.shape[:2]

    # --- 1. 整體旋轉 + 隨機偏移（模擬歪斜書寫） ---
    angle = random.uniform(-10, 10)
    shift_x = random.randint(-5, 5)
    shift_y = random.randint(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    M[0, 2] += shift_x
    M[1, 2] += shift_y
    warped = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- 2. 仿射局部扭曲（讓筆畫有點偏歪） ---
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    delta = random.uniform(-0.05, 0.05) * w
    pts2 = np.float32([
        [random.uniform(-delta, delta), random.uniform(-delta, delta)],
        [w + random.uniform(-delta, delta), random.uniform(-delta, delta)],
        [random.uniform(-delta, delta), h + random.uniform(-delta, delta)]
    ])
    M_affine = cv2.getAffineTransform(pts1, pts2)
    warped = cv2.warpAffine(warped, M_affine, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- 3. 灰階處理（讓筆劃更淡） ---
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 將黑轉灰（避免太深）
    gray = cv2.convertScaleAbs(gray, alpha=random.uniform(0.6, 0.85), beta=random.randint(10, 40))

    # --- 4. 模擬筆壓：局部腐蝕/膨脹/斷筆 ---
    kernel_size = random.choice([1, 2])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if random.random() < 0.6:
        gray = cv2.erode(gray, kernel, iterations=random.choice([1, 1, 2]))
    if random.random() < 0.5:
        gray = cv2.dilate(gray, kernel, iterations=random.choice([1, 1, 2]))

    # 局部破碎感
    if random.random() < 0.5:
        mask = np.random.randint(0, 2, size=gray.shape, dtype=np.uint8)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    # --- 5. 模糊邊緣（墨暈、掃描柔化） ---
    ksize = random.choice([3, 5])
    gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # --- 6. 紙張灰底 + 紋理噪聲 ---
    paper_tone = random.randint(220, 250)
    paper = np.full_like(gray, paper_tone, dtype=np.uint8)

    # 紋理噪聲（模擬紙張纖維）
    texture = np.random.normal(0, random.randint(3, 8), (h, w)).astype(np.float32)
    paper_texture = np.clip(paper.astype(np.float32) + texture, 0, 255).astype(np.uint8)

    # 字跡疊在紙上（保留柔灰邊）
    final = cv2.addWeighted(gray, 0.9, paper_texture, 0.1, 0)

    # --- 7. 再次整體亮度/對比隨機微調 ---
    alpha = random.uniform(0.85, 1.1)
    beta = random.uniform(-15, 15)
    final = cv2.convertScaleAbs(final, alpha=alpha, beta=beta)

    # 轉回 3 通道
    return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)


# === 主流程 ===
with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
lines = list(set(lines))  # 去重複
print(f"共 {len(lines)} 個注音")

for text in lines:
    clean_name = sanitize_filename(text)
    subfolder = os.path.join(augmented_folder, clean_name)
    os.makedirs(subfolder, exist_ok=True)

    # 生成基底圖片
    base_path = os.path.join(generated_folder, f"{clean_name}.png")
    generate_text_image(text, base_path)
    img = cv2.imread(base_path)
    #---
    # 使用 Unicode 安全的讀取方式
    try:
        # 讀取圖片 - Unicode 安全版本
        with open(base_path, 'rb') as f:
            img_data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"⚠️ 無法讀取 {text} 的圖片，跳過")
            continue
        
        # 生成多張手寫風格
        for i in range(1, num_aug + 1):
            aug_img = augment_image(img)
            aug_path = os.path.join(subfolder, f"{i:03}.png")
            
            # 使用 Unicode 安全的寫入方式
            _, img_encoded = cv2.imencode('.png', aug_img)
            with open(aug_path, 'wb') as f:
                f.write(img_encoded.tobytes())
        
        print(f"✅ {text} -> {num_aug} 張完成")
    
    except Exception as e:
        print(f"❌ 處理 {text} 時發生錯誤: {e}")
        continue
    #---
    # # 生成多張手寫風格
    # for i in range(1, num_aug + 1):
    #     aug_img = augment_image(img)
    #     aug_path = os.path.join(subfolder, f"{i:03}.png")
    #     cv2.imwrite(aug_path, aug_img)
    # print(f"✅ {text} -> {num_aug} 張完成")

print("\n🎉 所有手寫風格注音圖片已生成完成！")
