train() {
    model=$1
    fold=$2
    epochs=$3 
    gpu=$4   

    conf=./conf/${model}.py
    python3 -m src.cnn.main train ${conf} --fold ${fold} --epochs ${epochs} --gpu ${gpu}
}
train xresnet50 0 1 0
# train xresnet50 1 1
# train xresnet50 2 1
# train xresnet50 3 1
# train xresnet50 4 1
