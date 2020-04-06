train() {
    model=$1
    fold=$2
    gpu=$3    

    conf=./conf/${model}.py
    python3 -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
}
train xresnet50 0 1
train xresnet50 1 1
train xresnet50 2 1
train xresnet50 3 1
train xresnet50 4 1
