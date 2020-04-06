get_files(){
	file_dir=$1
	input_dir=$2

	data_dir=./input/${input_dir}
	mv ${file_dir} input_dir/${data_dir}
}

get_files ~/Downloads/bengaliai-cv19 bengaliai-cv19
get_files ~/Downloads/train_with_fold.csv folded_data
get_files ~/Downloads/train_images train_images
