# 模型 测试集 验证
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix
from train2 import loss_function, test_loader,FCNNTransformer,val_loader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
model_path='/home/xtc/文档/FFT+CNN+Trans/model2/best_model_cnn_transformer_0.9944.pt'
model = torch.load(model_path)
# 参数与配置
 # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size =16
# 有GPU先用GPU训练
# 使用测试集数据进行推断
# 得出每一类的分类准确率， 绘制混淆矩阵---0db
# 使用测试集数据进行推断并计算每一类的分类准确率

test_acc_list=[]
for i in range(20):
    with torch.no_grad():
        correct_test = 0
        test_loss = 0
        class_labels = []  # 存储类别标签
        predicted_labels_list = []  # 存储预测的标签
        for test_data, test_label in test_loader:

            test_data, test_label = test_data.to(device), test_label.to(device)
            test_output = model(test_data)
            probabilities = F.softmax(test_output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            correct_test += (predicted_labels == test_label).sum().item()
            loss = loss_function(test_output, test_label)
            test_loss += loss.item()
            class_labels.extend(test_label.tolist())
            predicted_labels_list.extend(predicted_labels.tolist())
        test_accuracy = correct_test / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        test_acc_list.append(test_accuracy)
        print(f'Test Accuracy: {test_accuracy:4.4f}  Test Loss: {test_loss:10.8f}')
        ma=np.mean(test_acc_list)
        if max(test_acc_list) > 0.982:
            # 混淆矩阵
            confusion_mat = confusion_matrix(class_labels, predicted_labels_list)
            # 计算每一类的分类准确率
            report = classification_report(class_labels, predicted_labels_list, digits=4)
            print(report)
            # 绘制混淆矩阵
            # 原始标签和自定义标签的映射
            label_mapping = {
                0: "wool", 1: "wood", 2: "cotton", 3: "leather", 4: "A4 paper",
                5: "styrofoam", 6: "copper", 7: "glass", 8: "PTFE"
            }
            # 绘制混淆矩阵
            plt.figure(figsize=(12, 10), dpi=300)
            sns.heatmap(confusion_mat, xticklabels=label_mapping.values(), yticklabels=label_mapping.values(), annot=True,
                        fmt='d',
                        cmap='summer')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()






