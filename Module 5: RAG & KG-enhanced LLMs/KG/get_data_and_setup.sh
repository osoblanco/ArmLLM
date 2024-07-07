# pip install -U git+https://github.com/facebookresearch/higher.git
# pip install -U git+https://github.com/EdinburghNLP/torch-adaptive-imle.git
pip install -U torchlens --quiet
pip install -U livelossplot --quiet

sudo apt-get install graphviz

# Download the Nations dataset
#!wget -c https://github.com/TimDettmers/ConvE/raw/master/nations.tar.gz
#!tar xvfz nations.tar.gz


if [ ! -d "data" ]; then
  mkdir data
fi

if [ ! -d "data/umls" ]; then
  mkdir data/umls
fi

if [ ! -d "data/FB15k-237" ]; then
  mkdir data/FB15k-237
fi

if [ ! -d "data/FB15k" ]; then
  mkdir data/FB15k
fi

if [ ! -d "data/WN18RR" ]; then
  mkdir data/WN18RR
fi

# Download the UMLS dataset
wget -P "data/umls" "https://github.com/TimDettmers/ConvE/raw/master/umls.tar.gz"
tar xvfz data/umls/umls.tar.gz -C data/umls

wget -P data/FB15k-237 https://github.com/TimDettmers/ConvE/raw/master/FB15k-237.tar.gz 
tar xvfz data/FB15k-237/FB15k-237.tar.gz  -C data/FB15k-237

wget -P data/WN18RR https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz 
tar xvfz data/WN18RR/WN18RR.tar.gz -C data/WN18RR
