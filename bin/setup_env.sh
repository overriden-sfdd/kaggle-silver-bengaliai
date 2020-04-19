#!/bin/bash
setup(){
    model_name=$1
    input_folders=$2

    mkdir input
    for sub_folder in $input_folders
    do
        mkdir input/$sub_folder
    done

    mkdir model    
    mkdir model/$model_name
}
setup xresnet50 $"bengaliai-cv19 folded_data train_images"
