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


# --- New helpers for rotation alignment ---

def circular_mask(h, w, radius=None, margin=6):
    cx, cy = w / 2.0, h / 2.0
    if radius is None:
        radius = min(h, w) * 0.5 - margin
    radius = max(1.0, min(radius, min(h, w) * 0.5 - 1.0))
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m = (dist2 <= radius ** 2).astype(np.float32)
    return m


def masked_corr(a, b, mask):
    # a, b: float32, same shape; mask: 0/1 float32
    msum = mask.sum()
    if msum < 1:
        return -1.0
    a_mean = (a * mask).sum() / msum
    b_mean = (b * mask).sum() / msum
    a_z = (a - a_mean) * mask
    b_z = (b - b_mean) * mask
    num = (a_z * b_z).sum()
    den = np.sqrt((a_z * a_z).sum() * (b_z * b_z).sum()) + 1e-12
    return float(num / den)


def rotate_k_image(img, k_ccw):
    k = int(k_ccw) % 4
    if k == 0:
        return img
    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def best_rotation_to_match(ref_gray8, img_gray8, mask):
    ref = ref_gray8.astype(np.float32)
    best_k, best_score = 0, -2.0
    for k in (0, 1, 2, 3):
        rot = np.ascontiguousarray(np.rot90(img_gray8, k))  # CCW k*90
        s = masked_corr(ref, rot.astype(np.float32), mask)
        if s > best_score:
            best_score = s
            best_k = k
    return best_k, best_score


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
    fnames = [f for f in sorted(os.listdir(in_path)) if f.startswith('SWAP_') and f.endswith('.png')]
    if not fnames:
        print("No matching input files.")
        return
    ref_path = os.path.join(in_path, fnames[0])
    ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        print(f"Skip unreadable: {ref_path}")
        return
    ref_rec = recenter_image(ref_img)
    ref_gray = to_gray_8u(ref_rec)
    (_, _), r = find_disk_center_and_radius(ref_gray)
    h, w = ref_gray.shape[:2]
    mask = circular_mask(h, w, radius=min(r * 0.95, min(h, w) * 0.5 - 6))

    ref_rec_colored = tint_image(ref_rec, "#ff5900", strength=0.4)
    cv2.imwrite(os.path.join(out_path, fnames[0]), ref_rec_colored)

    for fname in fnames[1:]:
        src = os.path.join(in_path, fname)
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skip unreadable: {src}")
            continue

        rec = recenter_image(img)
        if rec is None:
            print(f"Skip (processing failed): {src}")
            continue

        gray = to_gray_8u(rec)
        k, score = best_rotation_to_match(ref_gray, gray, mask)
        aligned = rotate_k_image(rec, k)

        colored = tint_image(aligned, "#ff5900", strength=0.4)

        cv2.imwrite(os.path.join(out_path, fname), colored)



if __name__ == '__main__':
    main()
