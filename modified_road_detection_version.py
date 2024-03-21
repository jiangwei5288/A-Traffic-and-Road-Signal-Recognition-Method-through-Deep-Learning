import torch
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
# from scipy.optimize import minimize
import torch.nn.functional as F

savepth = "D:/pycharm/pythonproject/test/Road_detection_modified/"
os.makedirs(savepth, exist_ok=True)

flag_train = True

task = 0  # 目前task = 0
print(task)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.Resize((200, 200)),  # 或者 transforms.Resize(200) 或 transforms.Resize((None, 200)) 等
    transforms.CenterCrop((200, 200)),  # 或者其他的保持宽高比的变换操作
    transforms.ToTensor()
])
# 如何在放大同时保留比例

test_data = datasets.ImageFolder(
    root="D:\My_project/test2",
    transform=transform
)

training_data = datasets.ImageFolder(
    root="D:\My_project/test1",
    transform=transform
)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 样本容量
    num_batches = len(dataloader)  # 批次数量
    model.eval()  # 切换为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad():  # 在评估阶段禁用梯度计算以加速过程
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # 使用模型时进行预测并保存到该标签
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    test_accuracy = correct
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, test_accuracy


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    loss_sum = []
    train_correct_predictions = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()  # 对模型进行反向传播
        optimizer.step()  # 根据优化器更新模型参数
        optimizer.zero_grad()  # 将梯度清零
        loss = loss.item() / len(X)

        loss_sum.append(loss)
    loss_avr = np.mean(loss_sum)

    _, predicted = torch.max(pred, 1)  # 获取每个样本的预测类别
    train_correct_predictions += (predicted == y).sum().item()
    train_accuracy = train_correct_predictions / size

    print("loss_avr = ", loss_avr)
    return loss_avr, train_accuracy


# Define model

class NeuralNetwork(nn.Module):  # CNN_model
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               dilation=1)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 47 * 47, 140)
        self.fc2 = nn.BatchNorm1d(140)
        self.fc3 = nn.Linear(140, 58)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ！！！！调整参数（尤其是pool），提高训练准确率！！！！


if flag_train:

    batch_size = 64

    learning_rate = 1e-3

    epochs = 200

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork().to(device)  # 创建一个NeuralNetwork类的实例并将模型移动到指定cuda或cpu
    print(model)

    loss_fn = nn.CrossEntropyLoss()  # 创建交叉熵，计算预测结果与真实值的差距
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 创建优化器，此处为SGD随机梯度下降优化器

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    min_loss = +np.inf  # 初始化最小值为正无穷
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        trainL, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        testL, test_acc = test(test_dataloader, model, loss_fn)
        train_loss_list.append(trainL)
        test_loss_list.append(testL)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        if testL < min_loss:
            min_loss = testL  # 最小误差列表
            torch.save(model.state_dict(), savepth + "/best_model.pth")  # 保存最佳模型
            print("最优模型已保存，路径为：", savepth)

    np.savetxt(savepth + 'train_loss_list.csv', train_loss_list)
    np.savetxt(savepth + 'test_loss_list.csv', test_loss_list)
    np.savetxt(savepth + 'train_acc_list.csv', train_acc_list)
    np.savetxt(savepth + 'test_acc_list.csv', test_acc_list)
    print("Done!")
    torch.save(model.state_dict(), savepth + "/model_25_95%.pth")
    print("Saved PyTorch Model State to model.pth")

else:
    correct_num = 0
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(savepth + '/model_25_95%.pth'))
    classes = [
        "limit_speed_to_5km_per_hour",
        "limit_speed_to_15km_per_hour",
        "limit_speed_to_30km_per_hour",
        "limit_speed_to_40km_per_hour",
        "limit_speed_to_50km_per_hour",
        "limit_speed_to_60km_per_hour",
        "limit_speed_to_70km_per_hour",
        "limit_speed_to_80km_per_hour",
        "no_left_or_straight",
        "no_right_or_straight",
        "no_straight",
        "no_left",
        "no_left_or_right",
        "no_right",
        "no_overtaking",
        "no_turning_back",
        "no_car_driving",
        "no_horn_honking",
        "highest_speed_is_40km_per_hour",
        "highest_speed_is_50km_per_hour",
        "straight_or_right",
        "straight",
        "left",
        "left_or_right",
        "right",
        "left_to_exit",
        "right_to_exit",
        "loop",
        "car_road",
        "horn_allowed",
        "bike_road",
        "turn_around_allowed",
        "intersection_with_a_fork",
        "pay_attention_to_traffic_lights",
        "watch_out_for_danger",
        "pay_attention_to_pavements",
        "pay_attention_to_bikes",
        "pay_attention_to_schools",
        "pay_attention_to_left_sharp_turns",
        "pay_attention_to_right_sharp_turns",
        "Caution_Downhill",
        "Caution_Uphill",
        "Slow_driving",
        "Right_intersection_sign",
        "Left_intersection_sign",
        "watch_out_for_villages",
        "two_rough_section_ahead",
        "Watch_out_for_trains",
        "Construction_zone_ahead",
        "three_rough_section_ahead",
        "Caution_Guardrail",
        "Caution_Trailer",
        "Stop",
        "Driving_forbidden",
        "Parking_forbidden",
        "No_entry",
        "Yield",
        "Stop_for_checking"
    ]

    model.eval()
    for i in range(len(test_data)):

        x, y = test_data[i][0], test_data[i][1]
        # 获得第一个样本的数据x和第一个样本的标签y
        with torch.no_grad():
            x = x.to(device)
            pred = model(x.unsqueeze(0))
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            # reshaped_image = np.transpose(x, (1, 2, 0))
            # plt.imshow(reshaped_image)
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
            # plt.show()

            if (predicted == actual):
                correct_num += 1
            print(correct_num)
            print(correct_num / (i + 1))
y1_values = np.loadtxt('D:\pycharm\pythonproject/test\Road_detection_modifiedtrain_loss_list.csv')
y2_values = np.loadtxt('D:\pycharm\pythonproject/test\Road_detection_modifiedtest_loss_list.csv')
y3_values = np.loadtxt('D:\pycharm\pythonproject/test\Road_detection_modifiedtrain_acc_list.csv')
y4_values = np.loadtxt('D:\pycharm\pythonproject/test\Road_detection_modifiedtest_acc_list.csv')
x = np.arange(1, len(y1_values) + 1)  # 用作横轴，表示1-100的epoch坐标
fig, ax = plt.subplots()
plt.figure(figsize=(8, 8))  # 打印图表大小
ax.plot(x, y1_values, label='train_loss_average')  # 横轴为1-100，纵轴为train_loss_list
ax.plot(x, y2_values, label='test_loss_average')
ax.plot(x, y3_values, label='train_acc_average')
ax.plot(x, y4_values, label='test_acc_average')
ax.legend()  # 添加图例，区分两条曲线
ax.set_title("train_loss and test_loss_&_accuracy_label")  # 总标题
ax.set_xlabel("epoch")  # x轴标题
ax.set_ylabel("loss_&_accuracy")  # y轴标题
plt.show()
