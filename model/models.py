from model.cnn import CNN4, CNN4_MNIST, CNN8
from model.srn import srn_replace_modules


def get_model(model, dataset):
    if "cifar100" in dataset:
        if "cnn8" in model:
            net = CNN8(100)
    elif "mnist" in dataset:
        if "cnn4" in model:
            net = CNN4_MNIST()
    else:
        if "cnn4" in model:
            net = CNN4()
        if "cnn8" in model:
            net = CNN8()

    if "srn" in model:
        srn_replace_modules(net)
    return net


if __name__ == "__main__":
    print("done")
