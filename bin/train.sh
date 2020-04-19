train() {
    model=$1
    fold=$2
    epochs=$3 
    gpu=$4   

    conf=./conf/${model}.py
    python3 -m src.cnn.main train ${conf} --fold ${fold} --epochs ${epochs} --gpu ${gpu}
}
train xresnet50 0 45 1
train xresnet50 1 45 1
train xresnet50 2 45 1
train xresnet50 3 45 1
train xresnet50 4 45 1
