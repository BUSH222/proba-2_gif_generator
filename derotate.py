import os
import cv2
import numpy as np


def to_gray_8u(img):
    if img is None:
        return None
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        dst = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return dst.astype(np.uint8)
    return img


def find_disk_center_and_radius(gray8):
    h, w = gray8.shape[:2]
    blur = cv2.GaussianBlur(gray8, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    center, radius = _center_from_mask(mask)
    if center is not None:
        return center, radius

    _, mask = cv2.threshold(gray8, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    center, radius = _center_from_mask(mask)
    if center is not None:
        return center, radius

    nz = cv2.findNonZero(mask)
    if nz is not None:
        x, y, w0, h0 = cv2.boundingRect(nz)
        cx, cy = x + w0 / 2.0, y + h0 / 2.0
        r = 0.5 * max(w0, h0)
        return (cx, cy), r

    return (w / 2.0, h / 2.0), min(w, h) / 2.0


def _center_from_mask(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return (float(cx), float(cy)), float(r)


def recenter_image(img):
    gray8 = to_gray_8u(img)
    if gray8 is None:
        return None
    (cx, cy), _ = find_disk_center_and_radius(gray8)
    print(cx, cy, _)
    h, w = img.shape[:2]
    tx = (w / 2.0) - cx
    ty = (h / 2.0) - cy
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return shifted


def main(in_path='2025-09-20_15-22_proba2_dump_2.235 GHz/SWAP', out_path='results'):
    os.makedirs(out_path, exist_ok=True)
    for fname in sorted(os.listdir(in_path)):
        if fname.startswith('SWAP_') and fname.endswith('.png'):
            src = os.path.join(in_path, fname)
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Skip unreadable: {src}")
                continue
            rec = recenter_image(img)
            if rec is None:
                print(f"Skip (processing failed): {src}")
                continue
            cv2.imwrite(os.path.join(out_path, fname), rec)



if __name__ == '__main__':
    main()
