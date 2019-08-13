mkdir dataset
mkdir data

cd dataset/
wget http://pjreddie.com/media/files/cifar.tgz -O cifar.tgz
tar xzvf cifar.tgz
rm -rf cifar.tgz
mv cifar/* .
cd ../

echo "[INFO] Data preparation: success" 

python create_tfrecord.py 
echo "[INFO] Tfrecords file preparation: success"