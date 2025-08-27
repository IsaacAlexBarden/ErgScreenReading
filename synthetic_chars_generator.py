import random, glob, os, csv
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import tqdm

CHARSET = list(("0123456789:/.-+,ViewDtalJnFbMrApyugSOcNovmsT"))
FONTS = glob.glob("char_data/fonts/*.ttf")  # LCD type fonts
OUT_DIR = "char_data/dataset_synth"
OUT_IMAGES_DIR = "char_data/dataset_synth/images"
N_PER_CLASS = 1000
IMG_SIZE = 48

try:
    os.makedirs(OUT_DIR, exist_ok=True)
except Exception as e:  # TODO handle exception properly
    raise RuntimeError("Could not create directory due to {e}")

try:
    os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
except Exception as e:  # TODO handle exception properly
    raise RuntimeError("Could not create directory due to {e}")

with open(os.path.join(OUT_DIR, "labels.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "label"])
    for ch in tqdm.tqdm(CHARSET):
        for i in range(N_PER_CLASS):
            font_path = random.choice(FONTS)
            font_size = random.randint(18, 48)  # Sample sizes TODO change magic nums
            font = ImageFont.truetype(font_path, font_size)

            canvas = Image.new("L", (96, 96), 255)  # Canvas to allow augmentation
            draw = ImageDraw.Draw(canvas)
            w_text, h_text = draw.textbbox((0, 0), ch, font=font)[2:]
            x = (96 - w_text) // 2 + random.randint(-3, 3)  # Random translation
            y = (96 - h_text) // 2 + random.randint(-3, 3)
            draw.text((x, y), ch, font=font, fill = 0)

            # Perturb thickness TODO make separate function
            arr = np.array(canvas)
            if random.random() < 0.5:
                k = random.choice([1, 1, 2])
                arr = cv2.erode(arr, np.ones((k, k), dtype=np.uint8))
            else:
                k = random.choice([1, 2])
                arr = cv2.dilate(arr, np.ones((k, k), dtype=np.uint8))
            
            # Rotation/Shear TODO make separate function

            img = Image.fromarray(arr)
            angle = random.uniform(-4, 4)  # degrees
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

            if random.random() < 0.5:
                shear = random.uniform(-0.05, 0.05)
                w0, h0 = img.size
                m = (1, shear, 0, 0, 1, 0)
                img = img.transform((w0, h0), Image.AFFINE, m, resample=Image.BILINEAR, fillcolor=255)

            # Small blur TODO separate function
            if random.random() < 0.8:
                img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.0, 1.5)))
            

            arr = np.array(img).astype(np.float32)
            if random.random() < 0.6:
                y_coords = np.arange(arr.shape[0])[:, None]
                band = 6 * np.sin(2*np.pi * y_coords / random.uniform(12, 24))
                arr = np.clip(arr + band, 0, 255)
            if random.random() < 0.7:
                grad = np.linspace(0, random.uniform(-15, 15), arr.shape[1])[None, :]
                arr = np.clip(arr + grad, 0, 255)

            # Noise TODO separate function
            if random.random() < 0.8:
                noise = np.random.normal(0, random.uniform(1, 8), arr.shape)
                arr = np.clip(arr + noise, 0, 255)
            
            # Rescaling for aliasing effects TODO ditto
            if random.random() < 0.7:
                scale = random.uniform(0.5, 0.9)
                small = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                arr = cv2.resize(small, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            def snap_to_dot_grid(img_np, target_height_px=24):
                # img_np: float32 [0..255]
                h, w = img_np.shape
                scale = target_height_px / h
                # Down to dot grid with NEAREST (make square dots)
                small = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                # Up with BILINEAR to soften grid a touch
                back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
                # Optional slight box blur to mimic bleed
                back = cv2.blur(back, (2,2))
                return back

            arr = snap_to_dot_grid(arr)

            # Crop glyph bounding box
            thresh = (arr < 240).astype(np.uint8)
            ys, xs = np.where(thresh > 0)
            if xs.size == 0:
                xs = np.array([IMG_SIZE])
                ys = np.array([IMG_SIZE])
            
            x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, arr.shape[1])
            y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, arr.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            crop = arr[y0:y1, x0:x1]

            h, w2 = crop.shape
            scale = (IMG_SIZE - 4) / max(h, w2)
            resized = cv2.resize(crop, (int(w2 * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            pad = IMG_SIZE - np.array(resized.shape)  # TODO magic number
            top = pad[0] // 2
            left = pad[1] // 2
            canvas_ = np.full((IMG_SIZE, IMG_SIZE), 255, np.uint8)
            canvas_[top:top + resized.shape[0], left:left + resized.shape[1]] = np.clip(resized, 0, 255).astype(np.uint8)

            fname = f"{ord(ch):05d}_{i:05d}.png"
            cv2.imwrite(os.path.join(OUT_IMAGES_DIR, fname), canvas_)
            w.writerow([fname, ch])



