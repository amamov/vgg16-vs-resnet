import torchvision


class ResNetMachine(Machine):

    """
    [Machine based in ResNet Model]
    """

    def __init__(self, batch_size=64, epoch_size=1, learning_rate=0.01, momentum=0.5):
        super().__init__(batch_size, epoch_size)
        model = torchvision.models.resnet.ResNet(pretrained=True, progress=True)
        self.setup_model(model)


if __name__ == "__main__":
    machine = Machine(batch_size=64, epoch_size=1)
    machine.learning()
