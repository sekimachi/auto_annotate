import cv2
import os
import numpy as np

def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)

    x_offset = (new_size - nw) // 2
    y_offset = (new_size - nh) // 2

    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    return canvas

#ここを変える
video_path = "input_video.mp4"


output_dir = "images"


os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ★ letterboxで640×640
    frame_640 = letterbox(frame, 640)

    cv2.imshow("Video Frame", frame_640)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame_640)
        print(f"保存しました: {frame_filename}")

    if key == ord('q'):
        break

    frame_count += 1

cap.release()

cv2.destroyAllWindows()