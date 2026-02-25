import cv2
import os
import numpy as np

def crop_to_4_3(img):
    h, w = img.shape[:2]
    target_ratio = 4 / 3
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x_start = (w - new_w) // 2
        cropped = img[:, x_start:x_start + new_w]
    else:
        new_h = int(w / target_ratio)
        y_start = (h - new_h) // 2
        cropped = img[y_start:y_start + new_h, :]

    return cropped


# ====== 動画を指定（最大6つまで）======
video_paths = [
    "IMG_2518.mov",
    "IMG_2516.mov",
    "IMG_2517.mov",
    "IMG_4242.mov",
    "IMG_4243.mov",
    "IMG_4244.mov",
    # ここに追加してOK（最大6個）
]

# 🔥 6個制限
MAX_VIDEOS = 6
if len(video_paths) > MAX_VIDEOS:
    print(f"⚠ 動画は最大{MAX_VIDEOS}個までです。先頭{MAX_VIDEOS}個のみ処理します。")
    video_paths = video_paths[:MAX_VIDEOS]

output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

for video_path in video_paths:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 動画を開けません:", video_path)
        continue

    print(f"▶ 処理開始: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"{video_path} 終了")
            break

        cropped = crop_to_4_3(frame)
        resized = cv2.resize(cropped, (640, 480))

        frame_filename = os.path.join(
            output_dir, f"frame_{frame_count:06d}.png"
        )
        cv2.imwrite(frame_filename, resized)

        frame_count += 1

    cap.release()

print("✅ 全動画処理完了")