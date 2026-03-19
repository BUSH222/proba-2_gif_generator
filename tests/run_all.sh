rm -rf tests/proba_qkris/1/SWAP_pro
rm -rf tests/proba_qkris/2/SWAP_pro
rm -rf tests/proba_qkris/3/SWAP_pro
rm -rf tests/proba_qkris/4/SWAP_pro
rm -rf tests/proba_qkris/5/SWAP_pro
rm -rf tests/proba_qkris/6/SWAP_pro
echo "old stuff removed"
python3 main.py -i tests/proba_qkris/1/SWAP
python3 main.py -i tests/proba_qkris/2/SWAP
python3 main.py -i tests/proba_qkris/3/SWAP
python3 main.py -i tests/proba_qkris/4/SWAP
python3 main.py -i tests/proba_qkris/5/SWAP
python3 main.py -i tests/proba_qkris/6/SWAP
echo "processing multiple batches (all folder, break here if not needed)"
rm -rf tests/proba_qkris/all/SWAP_pro
python3 main.py -i tests/proba_qkris/all/SWAP
echo "processing multiple batches (jpegs)"
rm -rf tests/aweeri/SWAP_pro
python3 main.py -i tests/aweeri/SWAP -r 1