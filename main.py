import os
import cv2
import numpy as np
import argparse
import imageio
import subprocess


def find_circle(img):
    '''Detect circle in the image using Hough Transform.'''
    blurred = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=img.shape[0] // 8,
        param1=50,
        param2=30,
        minRadius=294,
        maxRadius=310
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int").tolist()
        return circles[0]
    return None


def shift_image(img, current_x, current_y, target_x=1024/2, target_y=1024/2):
    '''Shift image so that (current_x, current_y) moves to (target_x, target_y).'''
    shift_x = target_x - current_x
    shift_y = target_y - current_y
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    return shifted_img


def match_rotation(img_base, img_to_rotate):
    """Find best 90° step rotation of img_to_rotate to match img_base."""
    correlations = []
    for k in range(4):
        rotated_img = np.rot90(img_to_rotate, k)
        res = cv2.matchTemplate(rotated_img, img_base, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        correlations.append(score)
    best_k = int(np.argmax(correlations))
    best_score = correlations[best_k]
    aligned_img = np.rot90(img_to_rotate, best_k)
    return aligned_img, best_score


def feature_align(img_base, img_to_align, max_features=500, good_match_percent=0.15,
                  max_translation=20, max_rotation_deg=15, min_scale=0.9, max_scale=1.1):
    """Align img_to_align to img_base using feature matching + affine transform."""
    orb = cv2.ORB_create(max_features)

    kp1, des1 = orb.detectAndCompute(img_base, None)
    kp2, des2 = orb.detectAndCompute(img_to_align, None)

    if des1 is None or des2 is None:
        return img_to_align, np.eye(2, 3, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 4:
        return img_to_align, np.eye(2, 3, dtype=np.float32)

    matches = sorted(matches, key=lambda x: x.distance)
    num_good = int(len(matches) * good_match_percent)
    matches = matches[:max(4, num_good)]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    M, inliers = cv2.estimateAffinePartial2D(pts2.reshape(-1, 1, 2), pts1.reshape(-1, 1, 2))

    if M is None:
        return img_to_align, np.eye(2, 3, dtype=np.float32)

    tx, ty = M[0, 2], M[1, 2]
    a, b = M[0, 0], M[1, 0]

    translation_magnitude = np.sqrt(tx**2 + ty**2)

    rotation_rad = np.arctan2(b, a)
    rotation_deg = np.degrees(rotation_rad)

    scale = np.sqrt(a**2 + b**2)

    if translation_magnitude > max_translation:
        print(f"Warning: Translation too large ({translation_magnitude:.1f} > {max_translation} px), skipping.")
        return img_to_align, np.eye(2, 3, dtype=np.float32)

    if abs(rotation_deg) > max_rotation_deg:
        print(f"Warning: Rotation too large ({rotation_deg:.1f}° > ±{max_rotation_deg}°), skipping.")
        return img_to_align, np.eye(2, 3, dtype=np.float32)

    if scale < min_scale or scale > max_scale:
        print(f"Warning: Scale out of range ({scale:.3f} not in [{min_scale}, {max_scale}]), skipping.")
        return img_to_align, np.eye(2, 3, dtype=np.float32)
    h, w = img_base.shape[:2]
    aligned = cv2.warpAffine(img_to_align, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return aligned, M


def run_imagemagick_tint(out_path):
    cmd = ['magick', 'mogrify', '-fill', '#edb103', '-tint', '100', '-contrast-stretch', '0.3%', 'SWAP_*']
    try:
        subprocess.run(cmd, cwd=out_path, check=True)
    except Exception:
        cmd = ['mogrify', '-fill', '#edb103', '-tint', '100', '-contrast-stretch', '0.3%', 'SWAP_*']
        subprocess.run(cmd, cwd=out_path, check=True)


def main(in_path, out_path, extra_rotation=0, no_magick=False, fps=5):
    sample_path = os.path.join(
        in_path,
        sorted(os.listdir(in_path), key=lambda x: (x[:21], len(x), x) if (x.endswith('.png') or x.endswith('.jpg'))
               else (float('inf'), 0))[0]
    )
    sample_image = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    processed_count = 0

    for file_name in sorted(os.listdir(in_path), key=lambda x: (x[:21], len(x), x)):
        if not (file_name.endswith('.png') or file_name.endswith('.jpg')):
            continue

        img = cv2.imread(os.path.join(in_path, file_name), cv2.IMREAD_UNCHANGED)

        circles = find_circle(img)
        if circles is None:
            print(f"Warning: No circle found in {file_name}, skipping.")
            continue
        centered_img = shift_image(img, circles[0], circles[1])
        rotated_img, best_rotation = match_rotation(sample_image, centered_img)
        rotated_circles = find_circle(rotated_img)
        if rotated_circles is None:
            print(f"Warning: No circle found after rotation in {file_name}, skipping.")
            continue
        rotated_recentered_img = shift_image(rotated_img, rotated_circles[0], rotated_circles[1])

        fine_aligned, M = feature_align(sample_image, rotated_recentered_img)

        cv2.imwrite(os.path.join(out_path, file_name), fine_aligned)

        processed_count += 1
        if processed_count % 2 == 0:
            sample_image = fine_aligned.copy()

    if not no_magick:
        try:
            run_imagemagick_tint(out_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ImageMagick mogrify failed: {e}. You can use --no_magick to disable it.")
    else:
        print("Skipping ImageMagick tinting.")

    # Gif creation
    out_files = [f for f in sorted(os.listdir(out_path), key=lambda x: (x[:21], len(x), x))
                 if (f.endswith('.png') or f.endswith('.jpg'))]
    images = []
    tmp_dir = os.path.join(out_path, "_tmp_rot")
    if extra_rotation:
        if no_magick:
            print("Skipping extra rotation as ImageMagick is disabled.")
            gif_files = [os.path.join(out_path, f) for f in out_files]
        else:
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            angle = 90 * extra_rotation
            rotated_files = []
            for f in out_files:
                src = os.path.join(out_path, f)
                dst = os.path.join(tmp_dir, f)
                cmd = ["magick", "convert", src, "-rotate", str(angle), dst]
                try:
                    subprocess.run(cmd, check=True)
                    rotated_files.append(dst)
                except Exception as e:
                    cmd = ["convert", src, "-rotate", str(angle), dst]
                    try:
                        subprocess.run(cmd, check=True)
                        rotated_files.append(dst)
                    except Exception:
                        print(f"Failed to rotate image {f} with both magick and convert commands.")
                    print(f"ImageMagick rotate failed for {f}: {e}")
            gif_files = rotated_files
    else:
        gif_files = [os.path.join(out_path, f) for f in out_files]

    for f in gif_files:
        img = cv2.imread(f)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)

    if images:
        gif_path = os.path.join(out_path, "out.gif")
        imageio.mimsave(gif_path, images, 'GIF', fps=fps, loop=0)

    if extra_rotation and os.path.exists(tmp_dir):
        for f in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, f))
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recenter, rotation-align (by 90° steps), tint, and GIF frames.')
    parser.add_argument('--in_path', '-i', required=True,
                        help='Input folder with PNG frames')
    parser.add_argument('--out_path', '-o', default=None,
                        help='Output folder for processed frames and GIF (default: <in_path>_pro)')
    parser.add_argument('--extra_rotation', '-r', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Extra CCW rotation in 90° steps (0..3, default: 0)')
    parser.add_argument('--no_magick', action='store_true',
                        help='Use this flag if magick is not installed as a global command. \
                        Some features will be disabled but the program will run.')
    parser.add_argument('--fps', '-f', type=int, default=10,
                        help='GIF frames per second (default: 10)')
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = args.in_path.rstrip('/\\') + '_pro'

    main(
        in_path=args.in_path,
        out_path=args.out_path,
        extra_rotation=args.extra_rotation,
        no_magick=args.no_magick,
        fps=args.fps
    )
