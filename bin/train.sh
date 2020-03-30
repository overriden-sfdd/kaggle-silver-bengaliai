train() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
}

train model100 0 0
