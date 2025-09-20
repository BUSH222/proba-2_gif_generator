import os
import cv2
import numpy as np


def main(in_path='2025-09-20_15-22_proba2_dump_2.235 GHz/SWAP', out_path='results'):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = [0, int(i*0.8), i]
    os.makedirs(out_path, exist_ok=True)
    for fname in os.listdir(in_path):
        if fname.startswith('SWAP_') and fname.endswith('.png'):
            img = cv2.imread(os.path.join(in_path, fname))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_yellow = cv2.LUT(img_bgr, lut)
            cv2.imwrite(os.path.join(out_path, fname), img_yellow)


if __name__ == '__main__':
    main()
