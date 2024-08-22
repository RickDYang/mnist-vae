import argparse
from dotenv import load_dotenv

from data import load_data
from model import MnistVaeModel

def train():
    batch_size = 512
    # The traning epochs, for my tesing, 80 is a good epochs
    epochs = 80
    # user all train data to train, test to val
    train_loader, _, val_loader = load_data(batch_size, ratio=1)
    model = MnistVaeModel()
    # model.load("mnist_vae_model.pt")
    # model.load()
    model.train(train_loader, val_loader, epochs)


def infer():
    model = MnistVaeModel()
    # model.load("mnist_vae_model_best.pt")
    model.load()
    model.infer(list(range(10)))

def upload():
    # Init and create the model
    model = MnistVaeModel()
    model.upload()


def from_pretrain():
    # Init and create the model
    model = MnistVaeModel()
    model.from_pretrain()
    model.infer(list(range(10)))


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run MNIST generation model with different modes: train/infer/upload/from_pretrain"
    )
    parser.add_argument("mode", choices=["train", "infer", "upload", "from_pretrain"], help="The mode to run")

    args = parser.parse_args()

    if args.mode == "train":
        train()
        infer()
    elif args.mode == "infer":
        infer()
    elif args.mode == "upload":
        upload()
    elif args.mode == "from_pretrain":
        from_pretrain()
    else:
        print(f"The mode={args.mode} not supported")
