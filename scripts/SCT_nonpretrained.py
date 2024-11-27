from joblib import load
import torch.utils.data as Data
import seaborn as sns
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
# 加载数据集
def dataloader(batch_size, workers=2):
    # 训练集
    train_xdata = load('./data5/trainX_1024_10c')
    train_ylabel = load('./data5/trainY_1024_10c')
    # 验证集
    val_xdata = load('./data5/valX_1024_10c')
    val_ylabel = load('./data5/valY_1024_10c')
    # 测试集
    # test_xdata = load('./data5/testX_1024_10c')
    # test_ylabel = load('./data4/testY_1024_10c')

    # 加载数据
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_xdata, train_ylabel),
                                   batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_xdata, val_ylabel),
                                 batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    #test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_xdata, test_ylabel),
    #                             batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    return train_loader, val_loader








class FCNNTransformer(nn.Module):
    def __init__(self, batch_size, timeconv_arch, spaceconv_arch, timeinput_dim, spaceinput_dim, output_dim, hidden_dim,
                 num_layers, num_heads, dropout_rate=0.5, input_channels=1):
        """
        分类任务  params:
        batch_size          : 批次量大小
        timeconv_arch       : 一维时域信号 cnn 网络结构
        spaceconv_arch      : 一维频域信号 cnn 网络结构
        timeinput_dim       : 时域卷积输入维度
        spaceinput_dim      : 频域卷积输入维度
        output_dim          : 输出的维度,类别数
        hidden_dim          : 注意力维度
        num_layers          : Transformer编码器层数
        num_heads           : 多头注意力头数
        dropout_rate              : 随机丢弃神经元的概率
        input_channels            : CNN输入维度(通道数)
        """
        super().__init__()
        # 批次量大小
        self.batch_size = batch_size
        # time-cnn参数
        self.timeconv_arch = timeconv_arch  # 网络结构
        self.timeinput_channels = input_channels  # 输入通道数
        self.timefeatures = self.timecnnmake_layers()
        # space-cnn参数
        self.spaceconv_arch = spaceconv_arch  # 网络结构
        self.spaceinput_channels = input_channels  # 输入通道数
        self.spacefeatures = self.spacecnnmake_layers()
        #添加注意力机制

        # Transformer编码器
        self.hidden_dim = hidden_dim
        # Time Transformer layers
        self.timetransformer = TransformerEncoder(
            TransformerEncoderLayer(timeinput_dim, num_heads, hidden_dim, dropout=0.5, batch_first=True),
            num_layers
        )
        # Space Transformer layers
        self.spacetransformer = TransformerEncoder(
            TransformerEncoderLayer(spaceinput_dim, num_heads, hidden_dim, dropout=0.5, batch_first=True),
            num_layers
        )
        # 序列平均池化操作
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 定义全连接层
        self.classifier = nn.Linear(timeinput_dim + spaceinput_dim, output_dim)

    # VGG卷积池化结构
    def timecnnmake_layers(self):
        layers = []
        for (num_convs, out_channels,kerel_size) in self.timeconv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.timeinput_channels, out_channels, kernel_size=kerel_size, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.timeinput_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(out_channels))  # 添加批量归一化层
        return nn.Sequential(*layers)

    def spacecnnmake_layers(self):
        layers = []
        for (num_convs, out_channels,kernel_size) in self.spaceconv_arch:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.spaceinput_channels, out_channels, kernel_size=kernel_size, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.spaceinput_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(out_channels))  # 添加批量归一化层
        return nn.Sequential(*layers)

    def forward(self, input_seq):

        # 时域特征 卷积池化处理
        time_input = input_seq.view(self.batch_size, 1, 512)
        # CNN 1d卷积 网络输入 [batch,H_in, seq_length]
        time_features = self.timefeatures(time_input)
        # print(time_features.size())   # torch.Size([32, 128, 129])
        # 调换维度[B, D, L] --> [B, L, D]
        time_features = time_features.permute(0, 2, 1)
        # Time-Transformer 处理
        # 在PyTorch中，transformer模型的性能与batch_first参数的设置相关。
        # 当batch_first为True时，输入的形状应为(batch, sequence, feature)，这种设置在某些情况下可以提高推理性能。
        time_transformer_output = self.timetransformer(time_features)  # torch.Size([32, 129, 128])
        # 序列平均池化操作
        time_transformer_output_avgpool = self.avgpool(
            time_transformer_output.transpose(1, 2))  # torch.Size([32, 1, 128])
        time_features = time_transformer_output_avgpool.reshape(self.batch_size, -1)  # torch.Size([32, 128])

        # 频域特征 卷积池化处理
        # 快速傅里叶变换
        # fft_result = torch.fft.rfft(input_seq, dim=1)
        # magnitude_spectrum = torch.abs(fft_result[:, :256]) #有改动 # [256, 512]
        # # 归一化操作
        # # normalized_spectrum = F.normalize(magnitude_spectrum, p=2, dim=1)
        # space_input = magnitude_spectrum.view(self.batch_size, 1, -1)  # torch.Size([32, 1, 512])
        # CNN 1d卷积 网络输入 [batch,H_in, seq_length]
        space_features = self.spacefeatures(time_input)  # torch.Size([32, 64, 65])
        # print(space_features.size())  # torch.Size([32, 64, 65])
        # 调换维度[B, D, L] --> [B, L, D]
        space_features = space_features.permute(0, 2, 1)
        # Space-Transformer 处理
        space__transformer_output = self.spacetransformer(space_features)  # torch.Size([32, 65, 64])
        # 序列平均池化操作
        space__transformer_output_avgpool = self.avgpool(
            space__transformer_output.transpose(1, 2))  # torch.Size([32, 1, 64])
        space_features = space__transformer_output_avgpool.reshape(self.batch_size, -1)  # torch.Size([32, 64])

        # 并行融合特征
        combined_features = torch.cat((time_features, space_features), dim=1)  # torch.Size([32, 128+64])

        outputs = self.classifier(combined_features)  # torch.Size([32, 10]  # 仅使用最后一个时间步的输出
        return outputs







# 看下这个网络结构总共有多少个参数
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')


loss_function = nn.CrossEntropyLoss(reduction='sum')  # loss
batch_size = 8
train_loader, val_loader = dataloader(batch_size)


def main():
    # 参数与配置

    #torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有GPU先用GPU训练
    print(device)
    # 加载数据
    print(len(train_loader))
    print(len(val_loader))
    #print(len(test_loader))
    # 定义 FCNNBiGRUAttModel 模型参数

    # 时域 卷积参数
    timeconv_arch = ((2, 32,9), (2, 64,6), (2, 128,3))  # CNN 层卷积池化结构  类似VGG
    # 频域 卷积参数
    spaceconv_arch = ((2, 16,3), (2, 32,3), (2, 64,3))  # CNN 层卷积池化结构  类似VGG
    # Transformer参数
    timeinput_dim = 128  # 时域卷积输入维度
    spaceinput_dim = 64  # 频域卷积输入维度
    hidden_dim = 128  # 注意力维度
    output_dim = 9  # 输出维度 9分类
    num_layers = 2  # 编码器层数
    num_heads = 2  # 多头注意力头数
    model = FCNNTransformer(batch_size, timeconv_arch, spaceconv_arch, timeinput_dim, spaceinput_dim, output_dim,
                            hidden_dim, num_layers, num_heads)
    # model_path=r'./model3/best_model_cnn_transformer_loss0.03012287.pt'
    # model = torch.load(model_path)
    # 定义损失函数和优化函数
    model = model.to(device)

    learn_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), learn_rate,weight_decay=1)  # 优化器#防止过你和可以添加weight_decay=0.1
    count_parameters(model)
    print(model)
    # 样本长度

    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size

    # 最高准确率  最佳模型
    best_accuracy = 0.0
    best_class_labels=[]
    best_predicted_labels_list=[]
    best_model = model

    train_loss = []  # 记录在训练集上每个epoch的loss的变化情况
    train_acc = []  # 记录在训练集上每个epoch的准确率的变化情况
    validate_acc = []
    validate_loss = []

    # 计算模型运行时间
    start_time = time.time()
    epochs =200
    # 训练模型


    for epoch in range(epochs):
        # 训练
        model.train()
        loss_epoch = 0.  # 保存当前epoch的loss和
        correct_epoch = 0  # 保存当前epoch的正确个数和
        correct_epochxxx = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(seq)  # torch.Size([16, 10])
            # 对模型输出进行softmax操作，得到概率分布
            probabilities = F.softmax(y_pred, dim=1)
            # 得到预测的类别
            predicted_labels = torch.argmax(probabilities, dim=1)
            # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
            correct_epoch += (predicted_labels == labels).sum().item()
            # 损失计算
            loss = loss_function(y_pred, labels)
            loss_epoch += loss.item()
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

        # 计算准确率
        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)
        print(f'Epoch: {epoch + 1:2} train_Loss: {loss_epoch / train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        class_labels = []  # 存储类别标签
        predicted_labels_list = []  # 存储预测的标签
        with torch.no_grad():
            loss_validate = 0.
            correct_validate = 0
            for data, label in val_loader:
                model.eval()   #eval模式会使bn层与dropout层失效
                data, label = data.to(device), label.to(device)
                pre = model(data)
                # 对模型输出进行softmax操作，得到概率分布
                probabilities = F.softmax(pre, dim=1)
                # 得到预测的类别
                predicted_labels = torch.argmax(probabilities, dim=1)
                # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
                correct_validate += (predicted_labels == label).sum().item()
                loss = loss_function(pre, label)
                loss_validate += loss.item()
                class_labels.extend(label.tolist())
                predicted_labels_list.extend(predicted_labels.tolist())
            # print(f'validate_sum:{loss_validate},  validate_Acc:{correct_validate}')
            val_accuracy = correct_validate / val_size
            print(f'Epoch: {epoch + 1:2} val_Loss:{loss_validate / val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')
            validate_loss.append(loss_validate / val_size)
            validate_acc.append(val_accuracy)
            # 如果当前模型的准确率优于之前的最佳准确率，则更新最佳模型
            # 保存当前最优模型参数
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_class_labels=class_labels
                best_predicted_labels_list=predicted_labels_list
                best_model = model  # 更新最佳模型的参数

    # 保存最后的参数
    # torch.save(model, 'final_model_cnn_transformer.pt')
    # 保存最好的参数
    torch.save(best_model, f'./model5/best_model_cnn_transformer_{round(best_accuracy, 8)}.pt')
    confusion_mat = confusion_matrix(best_class_labels, best_predicted_labels_list)

    # 计算每一类的分类准确率
    report = classification_report(best_class_labels, best_predicted_labels_list, digits=4)
    print(report)

    # 原始标签和自定义标签的映射
    label_mapping = {
   0: "#0", 1: "#1 ", 2: "#2", 3: "#3", 4: "#4",
        5: "#5", 6: "#6", 7: "#7", 8: "#8"
    }

    # 绘制混淆矩阵
    plt.figure(figsize=(15, 12), dpi=300)
    sns.set(font_scale=2)
    sns.heatmap(confusion_mat, xticklabels=label_mapping.values(), yticklabels=label_mapping.values(), annot=True,
                fmt='d',
                cmap='Blues')
    plt.xlabel('Predicted Labels', fontsize=23)
    plt.ylabel('True Labels', fontsize=23)
    plt.xticks(fontsize=15, rotation=45)
    plt.yticks(fontsize=15, rotation=45)
    plt.savefig(f'./fig5/gif_{round(best_accuracy, 6)}3.svg', dpi=600)
    plt.show()
    plt.show()

    matplotlib.rc("font", family='Microsoft YaHei')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    fig1,ax1=plt.subplots()
    plt.yticks(np.arange(0.0,1.01,0.1))
    ax1.grid(ls='--',lw=0.5,color='#4E616C')
    ax1.plot(range(epochs), validate_acc, color='#038355', label='validate_acc')
    ax1.plot(range(epochs), train_acc, color='#ffc34e', label='train_acc')
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax1.set_xlabel('epoch', fontsize=15)
    ax1.set_ylabel('Accuracy', fontsize=15)

    plt.savefig(f'./fig5/gif_{round(best_accuracy, 8)}1.svg',dpi=600)
    plt.show()  # 显示 lable


    fig2,ax2=plt.subplots()
    plt.yticks(np.arange(0, 2, 0.2))
    ax2.grid(ls='--',lw=0.5,color='#4E616C')
    ax2.plot(range(epochs), validate_loss, color='#ae5e52', label='validate_loss')
    ax2.plot(range(epochs), train_loss, color='#6b9ac8', label='train_loss')
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax2.set_xlabel('epoch', fontsize=15)
    ax2.set_ylabel('Loss', fontsize=15)

    plt.savefig(f'./fig5/gif_{round(best_accuracy, 8)}2.svg',dpi=600)
    plt.show()  # 显示 lable
    print("best_accuracy :", best_accuracy)
    # 将训练和验证的损失和准确率数据组织成DataFrame
    results_df = pd.DataFrame({
    'Train Loss': train_loss,
    'Validation Loss': validate_loss,
    'Train Accuracy': train_acc,
    'Validation Accuracy': validate_acc
    })

# 保存到CSV文件
    results_df.to_csv('./figdata/SCT_results/SCT_training_validation_results3.csv', index=False)
    # val_acc_data = pd.DataFrame(data = validate_acc,index = None,columns = ["validate_acc"])
    # train_acc_data = pd.DataFrame(data = train_acc,index = None,columns = ["train_acc"])
    # val_loss_data = pd.DataFrame(data = validate_loss,index = None,columns = ["validate_loss"])
    # train_loss_data = pd.DataFrame(data = train_loss,index = None,columns = ["train_loss"])
    
    # all_data=pd.concat([val_acc_data,train_acc_data,val_loss_data,train_loss_data],ignore_index=True,axis=1)
    # all_data.to_csv(f"./fig5/gif_{round(best_accuracy, 8)}2.csv")
    #绘制最好模型的混淆矩阵
    model = best_model
    # 使用测试集数据进行推断
    with torch.no_grad():
        correct_test = 0
        test_loss = 0
        for test_data, test_label in val_loader:
            # 将模型设置为评估模式
            model.eval()
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_test += (predicted_labels == test_label).sum().item()
            loss = loss_function(test_output, test_label)
            test_loss += loss.item()

    test_accuracy = correct_test / len(val_loader.dataset)
    test_loss = test_loss / len(val_loader.dataset)
    print(f'Test Accuracy: {test_accuracy:4.4f}  Test Loss: {test_loss:10.8f}')

    # 得出每一类的分类准确率， 绘制混淆矩阵---0db
    # 使用测试集数据进行推断并计算每一类的分类准确率
    class_labels = []  # 存储类别标签
    predicted_labels = []  # 存储预测的标签

    with torch.no_grad():
        for test_data, test_label in val_loader:
            # 将模型设置为评估模式
            model.eval()
            test_data = test_data.to(device)
            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

            class_labels.extend(test_label.tolist())
            predicted_labels.extend(predicted.tolist())

    # 混淆矩阵

if __name__ == '__main__':
    main()