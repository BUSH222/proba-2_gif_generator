import os
import cv2
import numpy as np


def hex_to_bgr_norm(hex_color):
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    bgr = np.array(rgb[::-1], dtype=np.float32) / 255.0
    return bgr


def tint_image(img, hex_color="#edb103", strength=1):
    img_f = img.astype(np.float32) / 255.0

    if img_f.ndim == 2:
        gray = img_f
        img_color = np.stack([gray, gray, gray], axis=2)
    elif img_f.shape[2] == 4:
        img_color = img_f[:, :, :3]
        gray = img_color[:, :, 2]*0.299 + img_color[:, :, 1]*0.587 + img_color[:, :, 0]*0.114
    elif img_f.shape[2] == 3:
        img_color = img_f
        gray = img_color[:, :, 2]*0.299 + img_color[:, :, 1]*0.587 + img_color[:, :, 0]*0.114

    f = 1.0 - 4.0 * (gray - 0.5) ** 2
    f = np.clip(f * float(strength), 0.0, 1.0)

    f3 = f[:, :, None]
    fill = hex_to_bgr_norm(hex_color).reshape((1, 1, 3))

    out = (1.0 - f3) * img_color + f3 * fill
    out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out_u8


def main(in_path='2025-09-20_15-22_proba2_dump_2.235 GHz/SWAP', out_path='results'):

    os.makedirs(out_path, exist_ok=True)
    for fname in os.listdir(in_path):
        if fname.startswith('SWAP_') and fname.endswith('.png'):
            img = cv2.imread(os.path.join(in_path, fname))
            tinted = tint_image(img, "#ff5900", strength=0.4)

            cv2.imwrite(os.path.join(out_path, fname), tinted)


if __name__ == '__main__':
    main()
