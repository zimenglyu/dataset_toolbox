# regression=linear
# regression=poly
regression=dnn

kernal=RBF
kernal=Matern

# input_X_path="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_10/Cyclone_10_202303_202105_202209_lab_results_spectra.csv"
# input_y_path="/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_10/Cyclone_10_202303_202105_202209_lab_results.csv"
# norm=minmax
input_X_path="/Users/zimenglyu/Documents/datasets/regression/energydata_200_train.csv"
input_y_path="/Users/zimenglyu/Documents/datasets/regression/energydata_200_label.csv"

pca_level=0.8
dataset_name="combined"
k_fold=5
num_train=160
gaussian_num_trails=100

for pca_level in 1
do
    folder_name="results_energy_200/$pca_level/$regression"  # Specify the folder path here
    if [ ! -d "$folder_name" ]; then
    echo "Creating folder: $folder_name"
    mkdir -p "$folder_name"
    else
    echo "Folder already exists: $folder_name"
    fi

    for norm in standard
    do
        # for kernal in RBF Matern
        # do
        for fold in 0 1 2 3 4 5 6 7 8 9
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
                                        --output_path $folder_name \
                                        --num_k_fold $k_fold 

    done
    done
    done