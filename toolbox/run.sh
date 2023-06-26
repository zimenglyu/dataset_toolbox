regression=linear
# regression=poly
# regression=gaussian
# regression=dnn

kernal=RBF
kernal=Matern

input_X_path="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
input_y_path="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv"
norm=standard
pca_level=0.9
dataset_name="combined_2021-2023"
k_fold=5
num_train=53
gaussian_num_trails=100
fold=0
# for norm in minmax 
# do
#     for kernal in RBF Matern
#     do
    for fold in 0 
    do
        python regression_main.py   --regression_method $regression \
                                    --input_X_path $input_X_path \
                                    --input_y_path $input_y_path \
                                    --dataset_name $dataset_name \
                                    --norm_method $norm \
                                    --pca_level $pca_level \
                                    --num_train $num_train \
                                    --fold $fold \
                                    --kernal_function $kernal \
                                    --num_trails $gaussian_num_trails \
                                    --num_k_fold $k_fold 

done
# done
# done