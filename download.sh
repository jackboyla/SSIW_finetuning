wget https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip
wget https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip
mkdir data
mkdir data/base
mkdir data/extended
unzip CMP_facade_DB_base.zip -d data/base
unzip CMP_facade_DB_extended.zip -d data/extended
rm CMP_facade_DB_base.zip
rm CMP_facade_DB_extended.zip
wget https://cloudstor.aarnet.edu.au/plus/s/gXaGsZyvoUwu97t/download -O universal_cat2vec.npy
wget https://cloudstor.aarnet.edu.au/plus/s/AtYYaVSVVAlEwve/download -O segformer_7data.pth
