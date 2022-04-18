#!/bin/bash
pathDatasetEuroc='/home/kai/file/VO_SpeedUp/Dataset/EuRoc' #Example, it is necesary to change it by the dataset path

pathDatasetTUM_VI='/home/kai/file/VO_SpeedUp/Dataset/TUM_VI'

# Single Session Example (Pure visual)
# echo "Launching MH01 with Stereo sensor"
# ./Examples/Stereo/stereo_euroc ./Vocabulary/ORBvoc.txt ./Examples/Stereo/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt dataset-MH01_stereo
# echo "------------------------------------"
# echo "Evaluation of MH01 trajectory with Stereo sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt f_dataset-MH01_stereo.txt --plot MH01_stereo.pdf --verbose2

# Single Session Example (Pure visual)
# echo "Launching MH01 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Examples/Monocular/EuRoC_TimeStamps/MH01.txt dataset-MH01_mono 
# echo "------------------------------------"
# echo "Evaluation of MH01 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt f_dataset-MH01_mono.txt --plot MH01_mono.pdf --verbose2

# echo "Launching MH02 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/MH02 ./Examples/Monocular/EuRoC_TimeStamps/MH02.txt dataset-MH02_mono 
# echo "------------------------------------"
# echo "Evaluation of MH02 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH02_GT.txt f_dataset-MH02_mono.txt --plot MH02_mono.pdf --verbose2

# echo "Launching MH03 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/MH03 ./Examples/Monocular/EuRoC_TimeStamps/MH03.txt dataset-MH03_mono 
# echo "------------------------------------"
# echo "Evaluation of MH03 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH03_GT.txt f_dataset-MH03_mono.txt --plot MH03_mono.pdf --verbose2

# echo "Launching MH04 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/MH04 ./Examples/Monocular/EuRoC_TimeStamps/MH04.txt dataset-MH04_mono 
# echo "------------------------------------"
# echo "Evaluation of MH04 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH04_GT.txt f_dataset-MH04_mono.txt --plot MH04_mono.pdf --verbose2

# echo "Launching MH05 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/MH05 ./Examples/Monocular/EuRoC_TimeStamps/MH05.txt dataset-MH05_mono 
# echo "------------------------------------"
# echo "Evaluation of MH05 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH05_GT.txt f_dataset-MH05_mono.txt --plot MH05_mono.pdf --verbose2

# echo "Launching V101 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/V101 ./Examples/Monocular/EuRoC_TimeStamps/V101.txt dataset-V101_mono 
# echo "------------------------------------"
# echo "Evaluation of V101 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/V101_GT.txt f_dataset-V101_mono.txt --plot V101_mono.pdf --verbose2

# echo "Launching V102 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/V102 ./Examples/Monocular/EuRoC_TimeStamps/V102.txt dataset-V102_mono 
# echo "------------------------------------"
# echo "Evaluation of V102 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/V102_GT.txt f_dataset-V102_mono.txt --plot V102_mono.pdf --verbose2

echo "Launching V103 with mono sensor"
./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/V103 ./Examples/Monocular/EuRoC_TimeStamps/V103.txt dataset-V103_mono 
echo "------------------------------------"
echo "Evaluation of V103 trajectory with mono sensor"
python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/V103_GT.txt f_dataset-V103_mono.txt --plot V103_mono.pdf --verbose2

# echo "Launching V201 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/V201 ./Examples/Monocular/EuRoC_TimeStamps/V201.txt dataset-V201_mono 
# echo "------------------------------------"
# echo "Evaluation of V201 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/V201_GT.txt f_dataset-V201_mono.txt --plot V201_mono.pdf --verbose2

# echo "Launching V202 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/V202 ./Examples/Monocular/EuRoC_TimeStamps/V202.txt dataset-V202_mono 
# echo "------------------------------------"
# echo "Evaluation of V202 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/V202_GT.txt f_dataset-V202_mono.txt --plot V202_mono.pdf --verbose2

# echo "Launching V203 with mono sensor"
# ./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/V203 ./Examples/Monocular/EuRoC_TimeStamps/V203.txt dataset-V203_mono 
# echo "------------------------------------"
# echo "Evaluation of V203 trajectory with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/V203_GT.txt f_dataset-V203_mono.txt --plot V203_mono.pdf --verbose2

# mono evaluation
# echo "Evaluation of MH01 trajectoty with mono sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt f_dataset-MH01_mono.txt --plot MH01_mono.pdf

# # MultiSession Example (Pure visual)
# echo "Launching Machine Hall with Stereo sensor"
# ./Examples/Stereo/stereo_euroc ./Vocabulary/ORBvoc.txt ./Examples/Stereo/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt "$pathDatasetEuroc"/MH02 ./Examples/Stereo/EuRoC_TimeStamps/MH02.txt "$pathDatasetEuroc"/MH03 ./Examples/Stereo/EuRoC_TimeStamps/MH03.txt "$pathDatasetEuroc"/MH04 ./Examples/Stereo/EuRoC_TimeStamps/MH04.txt "$pathDatasetEuroc"/MH05 ./Examples/Stereo/EuRoC_TimeStamps/MH05.txt dataset-MH01_to_MH05_stereo
# echo "------------------------------------"
# echo "Evaluation of MAchine Hall trajectory with Stereo sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_left_cam/MH_GT.txt f_dataset-MH01_to_MH05_stereo.txt --plot MH01_to_MH05_stereo.pdf


# # Single Session Example (Visual-Inertial)

# echo "Launching MH01 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt dataset-MH01_monoi
# echo "------------------------------------"
# echo "Evaluation of MH01 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/MH01/mav0/state_groundtruth_estimate0/data.csv f_dataset-MH01_monoi.txt --plot MH01_monoi.pdf --verbose2

# echo "Launching MH02 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH02 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/MH02.txt dataset-MH02_monoi
# echo "------------------------------------"
# echo "Evaluation of MH02 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/MH02/mav0/state_groundtruth_estimate0/data.csv f_dataset-MH02_monoi.txt --plot MH02_monoi.pdf --verbose2

# echo "Launching MH03 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH03 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/MH03.txt dataset-MH03_monoi
# echo "------------------------------------"
# echo "Evaluation of MH03 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/MH03/mav0/state_groundtruth_estimate0/data.csv f_dataset-MH03_monoi.txt --plot MH03_monoi.pdf --verbose2

# echo "Launching MH04 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH04 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/MH04.txt dataset-MH04_monoi
# echo "------------------------------------"
# echo "Evaluation of MH04 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/MH04/mav0/state_groundtruth_estimate0/data.csv f_dataset-MH04_monoi.txt --plot MH04_monoi.pdf --verbose2

# echo "Launching MH05 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH05 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/MH05.txt dataset-MH05_monoi
# echo "------------------------------------"
# echo "Evaluation of MH05 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/MH05/mav0/state_groundtruth_estimate0/data.csv f_dataset-MH05_monoi.txt --plot MH05_monoi.pdf --verbose2

# echo "Launching V101 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V101 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V101.txt dataset-V101_monoi
# echo "------------------------------------"
# echo "Evaluation of V101 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/V101/mav0/state_groundtruth_estimate0/data.csv f_dataset-V101_monoi.txt --plot V101_monoi.pdf --verbose2

# echo "Launching V102 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V102 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V102.txt dataset-V102_monoi
# echo "------------------------------------"
# echo "Evaluation of V102 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/V102/mav0/state_groundtruth_estimate0/data.csv f_dataset-V102_monoi.txt --plot V102_monoi.pdf --verbose2

# echo "Launching V103 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V103 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V103.txt dataset-V103_monoi
# echo "------------------------------------"
# echo "Evaluation of V103 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/V103/mav0/state_groundtruth_estimate0/data.csv f_dataset-V103_monoi.txt --plot V103_monoi.pdf --verbose2

# echo "Launching V201 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V201 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V201.txt dataset-V201_monoi
# echo "------------------------------------"
# echo "Evaluation of V201 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/V201/mav0/state_groundtruth_estimate0/data.csv f_dataset-V201_monoi.txt --plot V201_monoi.pdf --verbose2

# echo "Launching V202 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V202 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V202.txt dataset-V202_monoi
# echo "------------------------------------"
# echo "Evaluation of V202 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/V202/mav0/state_groundtruth_estimate0/data.csv f_dataset-V202_monoi.txt --plot V202_monoi.pdf --verbose2

# echo "Launching V203 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V203 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V203.txt dataset-V203_monoi
# echo "------------------------------------"
# echo "Evaluation of V203 trajectory with Monocular-Inertial sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/V203/mav0/state_groundtruth_estimate0/data.csv f_dataset-V203_monoi.txt --plot V203_monoi.pdf --verbose2

# # MultiSession Monocular Examples

# echo "Launching Vicon Room 2 with Monocular-Inertial sensor"
# ./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/V201 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V201.txt "$pathDatasetEuroc"/V202 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V202.txt "$pathDatasetEuroc"/V203 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/V203.txt dataset-V201_to_V203_monoi
# echo "------------------------------------"
# echo "Evaluation of Vicon Room 2 trajectory with Stereo sensor"
# python evaluation/evaluate_ate_scale.py evaluation/Ground_truth/EuRoC_imu/V2_GT.txt f_dataset-V201_to_V203_monoi.txt --plot V201_to_V203_monoi.pdf

# TUM_VI
# Monocular Examples
# echo "Launching Room 1 with Monocular sensor"
# ./Examples/Monocular/mono_tum_vi ./Vocabulary/ORBvoc.txt ./Examples/Monocular/TUM_1024.yaml "$pathDatasetTUM_VI"/dataset-room1_1024_16/mav0/cam0/data ./Examples/Monocular/TUM_TimeStamps/dataset-room1_512.txt dataset-room1_1024_mono
# echo "Evaluation of Room 1 trajectory with Mono sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetTUM_VI"/dataset-room1_1024_16/mav0/mocap0/data.csv f_dataset-room1_1024_mono.txt --plot dataset-room1_1024_mono.pdf --verbose2

# echo "Launching Room 1 with Monocular sensor"
# ./Examples/Monocular/mono_tum_vi ./Vocabulary/ORBvoc.txt ./Examples/Monocular/TUM_512.yaml "$pathDatasetTUM_VI"/dataset-room1_512_16/mav0/cam0/data ./Examples/Monocular/TUM_TimeStamps/dataset-room1_512.txt dataset-room1_512_mono
# echo "Evaluation of Room 1 trajectory with Mono sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetTUM_VI"/dataset-room1_512_16/mav0/mocap0/data.csv f_dataset-room1_512_mono.txt --plot dataset-room1_512_mono.pdf --verbose2

# echo "Launching Room 2 with Monocular sensor"
# ./Examples/Monocular/mono_tum_vi ./Vocabulary/ORBvoc.txt ./Examples/Monocular/TUM_512.yaml "$pathDatasetTUM_VI"/dataset-room2_512_16/mav0/cam0/data ./Examples/Monocular/TUM_TimeStamps/dataset-room2_512.txt dataset-room2_512_mono
# echo "Evaluation of Room 2 trajectory with Mono sensor"
# python evaluation/evaluate_ate_scale.py "$pathDatasetTUM_VI"/dataset-room2_512_16/mav0/mocap0/data.csv f_dataset-room2_512_mono.txt --plot dataset-room2_512_mono.pdf --verbose2