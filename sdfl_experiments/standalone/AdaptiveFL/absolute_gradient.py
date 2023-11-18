import torch
import torch.nn as nn
from sdfl_api.models.MLP_mnist import MLPMnist
from sdfl_api.models.resnet_18 import SlimmableResNet18
from sdfl_api.models.mobilenet_v1 import SlimmableMobileNetV1
from sdfl_api.models.vgg16 import SlimmableVGG16
from sdfl_api.models.vgg11 import SlimmableVGG11
from sdfl_api.models.resnet_9 import SlimmableResNet9


class TestAbGradient:
    def __init__(self, args, dataset, device):
        self.args = args
        self.train_loaders, self.test_loader = dataset
        self.device = device

    def create_model(self, model_name):
        model = None
        if model_name == "MLP_mnist":
            model = MLPMnist()
        elif model_name == "resnet18":
            model = SlimmableResNet18()
        elif model_name == 'mobilenet_v1':
            model = SlimmableMobileNetV1()
        elif model_name == 'vgg16':
            model = SlimmableVGG16()
        elif model_name == 'vgg11':
            model = SlimmableVGG11()
        elif model_name == 'resnet9':
            model = SlimmableResNet9()
        return model

    def test_ab_gradient(self):
        width_list = [1, 0.75, 0.5, 0.25]
        for i, width_mult in enumerate(width_list):
            # 在测试集上进行预测
            model = self.create_model(self.args.model)
            # weight_path = '../../../weights/resnet9-global-2.pth'
            weight_path = '/home/lgz/paper/federated_learning_202305/codes/weights/resnet9-global-1.pth'
            model.load_state_dict(torch.load(weight_path))
            model.set_width_mult(width_mult)
            criterion = nn.CrossEntropyLoss().to(self.device)
            # # 在测试集上进行预测
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print("accuracy: ",accuracy)
            for images, labels in self.test_loader:
                images.requires_grad_(True)
                output = model(images)
                loss = criterion(output, labels) # 你可以根据你的任务定义合适的损失函数

                # 执行反向传播
                loss.backward()

                # 获取模型中所有参数的绝对梯度
                abs_gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # print(len(param))
                        abs_gradients[name] = param.grad.abs().mean().item()

                print("=====")

                print("width:", width_mult, "absolute gradient: ", abs_gradients)
                break
