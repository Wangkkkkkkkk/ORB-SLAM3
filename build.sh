echo "Configuring and building Thirdparty/DBoW2 ..."

# cd Thirdparty/DBoW2
# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j4

# cd ../../g2o

# echo "Configuring and building Thirdparty/g2o ..."4

# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j4

# cd ../../../

# echo "Uncompress vocabulary ..."

# cd Vocabulary
# tar -xf ORBvoc.txt.tar.gz
# cd ..

echo "Configuring and building ORB_SLAM3 ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
