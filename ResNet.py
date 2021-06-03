# * 해당 파일은 VGG16 파일과 함께 Colab에서 실행해야 합니다.

import torchvision
from ResNet import Machine


class ResNetMachine(Machine):

    """
    [Machine based in ResNet Model]
    VGG Machine Inheritance
    """

    def __init__(self, batch_size=64, epoch_size=1, learning_rate=0.01, momentum=0.5):
        super().__init__(batch_size, epoch_size, learning_rate, momentum)
        model = torchvision.models.resnet.ResNet(pretrained=True, progress=True)
        self.setup_model(model)


if __name__ == "__main__":
    machine = Machine(batch_size=64, epoch_size=1)
    machine.learning()
