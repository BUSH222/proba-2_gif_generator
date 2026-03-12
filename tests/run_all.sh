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