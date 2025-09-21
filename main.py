import os
import cv2
import numpy as np
import imageio
import argparse


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


def tint_image_original(img, strength=0.5):
    """
    Mimic macOS Night Shift by shifting white balance toward warmer tones.
    
    Args:
        img: numpy array (H,W), (H,W,3), or (H,W,4), dtype=uint8.
        strength: float in [0,1], degree of warmth (0=no change, 1=max warm).
    
    Returns:
        BGR uint8 image with warm shift applied.
    """
    img_f = img.astype(np.float32) / 255.0

    # Drop alpha if present
    if img_f.ndim == 3 and img_f.shape[2] == 4:
        img_f = img_f[:, :, :3]

    # If grayscale, expand to 3 channels
    if img_f.ndim == 2:
        img_f = np.stack([img_f, img_f, img_f], axis=2)

    # White balance multipliers
    # At strength=1.0: red ~1.0, green ~0.9, blue ~0.6 (example warm balance)
    r_gain = 1.0
    g_gain = 1.0 - 0.1 * strength
    b_gain = 1.0 - 0.4 * strength

    gains = np.array([b_gain, g_gain, r_gain], dtype=np.float32)

    # Apply gains per channel
    out = img_f * gains[None, None, :]

    # Normalize so we don't wash out whites
    out = np.clip(out, 0, 1)

    return (out * 255).astype(np.uint8)


def contrast_stretch(img, black_point=0.3, white_point=0):
    if white_point is None:
        white_point = black_point
    
    img_f = img.astype(np.float32)

    # Handle grayscale or color channels independently
    if img_f.ndim == 2:
        channels = [img_f]
    else:
        channels = [img_f[..., c] for c in range(img_f.shape[2])]

    stretched_channels = []
    for ch in channels:
        # Flatten and compute percentiles
        low = np.percentile(ch, black_point)
        high = np.percentile(ch, 100 - white_point)

        if high <= low:
            # avoid div/0 – just return original channel
            stretched = ch
        else:
            stretched = (ch - low) * 255.0 / (high - low)
            stretched = np.clip(stretched, 0, 255)

        stretched_channels.append(stretched)

    if img_f.ndim == 2:
        out = stretched_channels[0]
    else:
        out = cv2.merge([c.astype(np.float32) for c in stretched_channels])

    return out.astype(np.uint8)


def main(in_path,
         out_path,
         tint_color="#ff5900",
         tint_strength=0.4,
         gif_speed=0.1,
         extra_rotation=0):
    os.makedirs(out_path, exist_ok=True)
    fnames = [f for f in sorted(os.listdir(in_path), key=lambda x: (len(x), x)) if f.startswith('SWAP_') and f.endswith('.png')]
    if not fnames:
        print("No matching input files.")
        return

    # Reference
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

    # Save reference (apply extra rotation then tint)
    ref_out = rotate_k_image(ref_rec, extra_rotation)
    ref_out_stretched = contrast_stretch(ref_out, black_point=5, white_point=0.2)
    ref_rec_colored = tint_image(ref_out_stretched, tint_color, strength=tint_strength)
    ref_rec_colored_warm = tint_image_original(ref_rec_colored, strength=2)
    cv2.imwrite(os.path.join(out_path, fnames[0]), ref_rec_colored_warm)

    # Align and save the rest
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
        k, _ = best_rotation_to_match(ref_gray, gray, mask)
        aligned = rotate_k_image(rec, k)
        aligned = rotate_k_image(aligned, extra_rotation)  # apply extra rotation to frames too

        colored = tint_image(aligned, tint_color, strength=tint_strength)
        colored_warm = tint_image_original(colored, strength=2)
        cv2.imwrite(os.path.join(out_path, fname), colored_warm)

    # Build GIF from saved frames (already rotated and tinted)
    out_files = [f for f in sorted(os.listdir(out_path)) if f.endswith('.png')]
    images = []
    for f in out_files:
        img = cv2.imread(os.path.join(out_path, f))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)

    if images:
        gif_path = os.path.join(out_path, "out.gif")
        imageio.mimsave(gif_path, images, duration=gif_speed, loop=0)


def _hex_color(s):
    s = s.strip()
    if not s:
        raise argparse.ArgumentTypeError("tint_color cannot be empty")
    if s[0] != '#':
        s = '#' + s
    if len(s) != 7:
        raise argparse.ArgumentTypeError("tint_color must be #RRGGBB")
    try:
        int(s[1:], 16)
    except ValueError:
        raise argparse.ArgumentTypeError("tint_color must be hex #RRGGBB")
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recenter, rotation-align (by 90° steps), tint, and GIF frames.')
    parser.add_argument('--in_path', '-i', required=True,
                        help='Input folder with PNG frames')
    parser.add_argument('--out_path', '-o', required=True,
                        help='Output folder for processed frames and GIF')
    parser.add_argument('--extra_rotation', '-r', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Extra CCW rotation in 90° steps (0..3, default: 0)')
    parser.add_argument('--tint_color', default="#e57200", type=_hex_color,
                        help='Tint color as #RRGGBB (default: #ff5900)')
    parser.add_argument('--tint_strength', type=float, default=0.4,
                        help='Tint strength in [0,1] (default: 0.4)')
    parser.add_argument('--gif_speed', type=float, default=0.1,
                        help='Frame duration in seconds (default: 0.1)')
    args = parser.parse_args()

    main(
        in_path=args.in_path,
        out_path=args.out_path,
        tint_color=args.tint_color,
        tint_strength=args.tint_strength,
        gif_speed=args.gif_speed,
        extra_rotation=args.extra_rotation
    )
