# Please check out the Slice Licence agreement before downloading the following files
# https://github.com/SlicerDMRI/TractCloud/blob/main/LICENSE

curl -L "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0/TrainData_800clu800ol.tar.gz" > "TrainData_800clu800ol.tar.gz"
tar -xzvf TrainData_800clu800ol.tar.gz

curl -L "https://github.com/SlicerDMRI/TractCloud/raw/2c2869cde12b8d617d21be8e19fd33ae73b5f3bc/datasets/FiberClusterAnnotation_Updated20230110.xlsx" > "TrainData_800clu800ol/FiberClusterAnnotation_Updated20230110.xlsx"