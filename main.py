import os


def main(in_path='2025-09-20_15-22_proba2_dump_2.235 GHz/SWAP', out_path='results'):
    # Colorize output
    os.makedirs(out_path, exist_ok=True)
    for fname in os.listdir(in_path):
        if fname.startswith('SWAP_') and fname.endswith('.png'):
            pass


if __name__ == '__main__':
    main()
