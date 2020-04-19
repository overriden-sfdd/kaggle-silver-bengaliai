setup(){
    model_name=$1
    declare -a input_folders=($2)
    
    mkdir input
    for sub_folder in "input_folders[@]" 
    do
        mkdir input/$sub_folder
    done
    
    mkdir model/$model_name
}

setup xresnet50 "begnaliai-cv19" "folded_data" "train_images"