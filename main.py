import os
import cv2
import numpy as np
import argparse


def find_circle(img):
    blurred = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=img.shape[0] // 8,
        param1=50,
        param2=30,
        minRadius=300,
        maxRadius=308
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int").tolist()
        return circles[0]
    return None


def shift_image(img, current_x, current_y, target_x=1024/2, target_y=1024/2):
    shift_x = target_x - current_x
    shift_y = target_y - current_y
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    return shifted_img
    


def main(in_path, out_path, extra_rotation=0):
    for file_name in sorted(os.listdir(in_path), key=lambda x: (len(x), x)):
        if file_name.endswith('.png'):
            img = cv2.imread(os.path.join(in_path, file_name), cv2.IMREAD_UNCHANGED)
            circles = find_circle(img)
            processed_img = shift_image(img, circles[0], circles[1])
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            cv2.imwrite(os.path.join(out_path, file_name), processed_img)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recenter, rotation-align (by 90° steps), tint, and GIF frames.')
    parser.add_argument('--in_path', '-i', default='2025-09-20_15-22_proba2_dump_2.235 GHz/SWAP',
                        help='Input folder with PNG frames')
    parser.add_argument('--out_path', '-o', default='results',
                        help='Output folder for processed frames and GIF')
    parser.add_argument('--extra_rotation', '-r', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Extra CCW rotation in 90° steps (0..3, default: 0)')
    args = parser.parse_args()

    main(
        in_path=args.in_path,
        out_path=args.out_path,
        extra_rotation=args.extra_rotation
    )
