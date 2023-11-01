

# regression=linear
# regression=poly 
# regression=gaussian
regression=dnn

kernal=RBF
kernal=Matern

file_length=200

input_path="/Users/zimenglyu/Documents/datasets/regression"

norm=standard
pca_level=0.9
dataset_name="combined"
k_fold=5
num_train=$(echo "0.8 * $file_length" | bc | cut -d'.' -f1)
echo 'number of train datapoints are: '$num_train
gaussian_num_trails=100
fold=0


for pca_level in 1
do
    for file in 1 2 3 4 5 6 7 8 9
    do
        folder_name="results_energy_${file_length}/pca_$pca_level/$regression/$file"  # Specify the folder path here
        if [ ! -d "$folder_name" ]; then
        echo "Creating folder: $folder_name"
        mkdir -p "$folder_name"
        else
        echo "Folder already exists: $folder_name"
        fi

        for norm in minmax standard robust 
        do
            for kernal in RBF
            do
                for fold in 0
                do
                    train_file="$input_path/$file_length/energydata_${file}.csv"
                    train_label="$input_path/$file_length/energydata_${file}_label.csv"
                    python regression_main.py   --regression_method $regression \
                                                --input_X_path $train_file \
                                                --input_y_path $train_label \
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
    done
done