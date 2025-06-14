import pandas as pd
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageDraw
from collections import Counter
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, precision_score, accuracy_score, classification_report, confusion_matrix
pd.set_option('display.max_columns', None)    # Dataframe显示所有列

model_dir_path = '2-7-2-2/models'
model_dir_namelist = os.listdir(model_dir_path)
result_save_path = '3-4-4'
# 获取图片尺寸信息
pict_test_data_dir = 'D:/ANCAI/science/paper/max/data after mask/data after mask'
# 可视化图像放大倍数
amplify = 16

model_accuracy_pd = pd.DataFrame(columns=['model name', 'point test accuracy', 'result test accuracy without weight', 'result test accuracy without weight model', 'result test accuracy with weight model1', 'result test accuracy with weight model2'])
model_accuracy_pd_model_name_list = []
model_accuracy_pd_point_test_accuracy_list = []
model_accuracy_pd_result_test_accuracy_without_weight_list = []
model_accuracy_pd_result_test_accuracy_without_weight_model_list = []
model_accuracy_pd_result_test_accuracy_with_weight_model1_list = []
model_accuracy_pd_result_test_accuracy_with_weight_model2_list = []

for model_dir_name in model_dir_namelist:
    os.mkdir(result_save_path + '/' + str(model_dir_name))
    os.mkdir(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化')
    train_pd_name = model_dir_path + '/' + model_dir_name + '/训练集数据.pkl'
    train_array_normalized_name = model_dir_path + '/' + model_dir_name + '/训练集数据特征归一化数组.npy'
    train_array_target_name = model_dir_path + '/' + model_dir_name + '/训练集标签数组.npy'
    val_pd_name = model_dir_path + '/' + model_dir_name + '/验证集数据.pkl'
    val_array_normalized_name = model_dir_path + '/' + model_dir_name + '/验证集数据特征归一化数组.npy'
    val_array_target_name = model_dir_path + '/' + model_dir_name + '/验证集标签数组.npy'
    test_pd_name = model_dir_path + '/' + model_dir_name + '/测试集数据.pkl'
    test_array_normalized_name = model_dir_path + '/' + model_dir_name + '/测试集数据特征归一化数组.npy'
    test_array_target_name = model_dir_path + '/' + model_dir_name + '/测试集标签数组.npy'
    model_name = model_dir_path + '/' + model_dir_name + '/DNN.h5'

    train_pd = pd.read_pickle(train_pd_name)
    train_array_normalized = np.load(train_array_normalized_name)
    train_array_target = np.load(train_array_target_name)
    val_pd = pd.read_pickle(val_pd_name)
    val_array_normalized = np.load(val_array_normalized_name)
    val_array_target = np.load(val_array_target_name)
    test_pd = pd.read_pickle(test_pd_name)
    test_array_normalized = np.load(test_array_normalized_name)
    test_array_target = np.load(test_array_target_name)
    model = tf.keras.models.load_model(model_name)
    test_pd.reset_index(drop=True, inplace=True)

    # 训练weights模型，用来计算随输入变化的weights，作为每个像素点的weights。
    # 输入为每个像素点的6个特征，输出为一个0-1之间的概率，表示使用的像素点分类模型对每个像素点分类结果的可信度，作为该像素点在该像素点分类模型下的投票权重。
    # 第一种训练方式：像素模型分类正确的像素点，训练weights模型时标签为1，分类错误标签为0。训练神经网络二分类模型
    # 第二种训练方式：像素模型在正确的分类上预测的概率作为训练weights模型时的标签，训练神经网络回归模型
    # 第三种训练方式：像素模型分类正确的像素点，训练weights模型时标签为1，分类错误标签为0。训练SVM二分类模型
    # 三种训练方式会获得三个weights模型

    # weights模型所需数据
    X_train = train_array_normalized
    Y_train_predict = model.predict(train_array_normalized)
    Y_train_pred = np.argmax(Y_train_predict, axis=1)
    weights_Y_train_1 = np.equal(train_array_target, Y_train_pred).astype(int)
    weights_Y_train_2 = Y_train_predict[np.arange(Y_train_predict.shape[0]), train_array_target]
    X_val = val_array_normalized
    Y_val_predict = model.predict(val_array_normalized)
    Y_val_pred = np.argmax(Y_val_predict, axis=1)
    weights_Y_val_1 = np.equal(val_array_target, Y_val_pred).astype(int)
    weights_Y_val_2 = Y_val_predict[np.arange(Y_val_predict.shape[0]), val_array_target]
    X_test = test_array_normalized
    Y_test_predict = model.predict(test_array_normalized)
    Y_test_pred = np.argmax(Y_test_predict, axis=1)
    weights_Y_test_1 = np.equal(test_array_target, Y_test_pred).astype(int)
    weights_Y_test_2 = Y_test_predict[np.arange(Y_test_predict.shape[0]), test_array_target]
    print(weights_Y_train_1)
    print(weights_Y_train_2)

    # 第一个weights模型
    weight_model1 = tf.keras.models.Sequential()
    weight_model1.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    weight_model1.add(BatchNormalization())
    weight_model1.add(tf.keras.layers.Dropout(0.6))

    weight_model1.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    weight_model1.add(BatchNormalization())
    weight_model1.add(tf.keras.layers.Dropout(0.6))
    weight_model1.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
    weight_model1.add(BatchNormalization())
    weight_model1.add(tf.keras.layers.Dropout(0.6))

    weight_model1.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))  # kernel_regularizer=tf.keras.regularizers.l2(l2=10)
    weight_model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath=result_save_path + '/' + str(model_dir_name) + '/best_weight_model1.hdf5', verbose=1, save_best_only=True)
    # monitor:监视参数，min_delta:小于此数认为不变化，mode:loss小好，acc大好，patience:n周期无提升则退出，restore_best_weights:取最优权重
    earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, mode='max', verbose=1, restore_best_weights=True)
    # 增加validation_data参数作为验证集，添加早停止机制，训练时打乱序列顺序
    history1 = weight_model1.fit(X_train, weights_Y_train_1, callbacks=[checkpointer, earlyStop], epochs=100, batch_size=10, verbose=1,
                        validation_data=(X_val, weights_Y_val_1), shuffle=True)
    acc1 = history1.history['accuracy']
    val_acc1 = history1.history['val_accuracy']
    loss1 = history1.history['loss']
    val_loss1 = history1.history['val_loss']
    epochs1 = range(len(acc1))
    best_weight_model1 = tf.keras.models.load_model(result_save_path + '/' + str(model_dir_name) + '/best_weight_model1.hdf5')
    y_pred1 = best_weight_model1.predict(X_test)

    # 画accuracy曲线
    plt.plot(epochs1, acc1, 'r')
    plt.plot(epochs1, val_acc1, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])
    plt.savefig(result_save_path + '/' + str(model_dir_name) + '/weight模型1 准确率学习曲线.jpg')
    plt.cla()
    # 画loss曲线
    plt.plot(epochs1, loss1, 'r')
    plt.plot(epochs1, val_loss1, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    plt.savefig(result_save_path + '/' + str(model_dir_name) + '/weight模型1 损失学习曲线.jpg')
    plt.cla()

    y_predict1 = np.around(y_pred1, 0)
    weight_model1.summary()

    accuracyscore = accuracy_score(weights_Y_test_1, y_predict1)
    print("accuracy is ", accuracyscore)
    accuracy_pd = pd.DataFrame({'accuracy score': [accuracyscore]})
    accuracy_pd.to_excel(result_save_path + '/' + str(model_dir_name) + '/weight模型1 准确率.xlsx', index=False)
    accuracy_pd.to_pickle(result_save_path + '/' + str(model_dir_name) + '/weight模型1 准确率.pkl')

    # 第二个weights模型
    weight_model2 = tf.keras.models.Sequential()
    weight_model2.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    weight_model2.add(BatchNormalization())

    weight_model2.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    weight_model2.add(BatchNormalization())
    weight_model2.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
    weight_model2.add(BatchNormalization())

    weight_model2.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))  # kernel_regularizer=tf.keras.regularizers.l2(l2=10)
    weight_model2.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath=result_save_path + '/' + str(model_dir_name) + '/best_weight_model2.hdf5',
                                   verbose=1, save_best_only=True)
    # monitor:监视参数，min_delta:小于此数认为不变化，mode:loss小好，acc大好，patience:n周期无提升则退出，restore_best_weights:取最优权重
    earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, mode='max', verbose=1,
                              restore_best_weights=True)
    # 增加validation_data参数作为验证集，添加早停止机制，训练时打乱序列顺序
    history2 = weight_model2.fit(X_train, weights_Y_train_2, callbacks=[checkpointer, earlyStop], epochs=100, batch_size=10, verbose=1,
                                 validation_data=(X_val, weights_Y_val_1), shuffle=True)
    acc2 = history2.history['accuracy']
    val_acc2 = history2.history['val_accuracy']
    loss2 = history2.history['loss']
    val_loss2 = history2.history['val_loss']
    epochs2 = range(len(acc2))
    best_weight_model2 = tf.keras.models.load_model(result_save_path + '/' + str(model_dir_name) + '/best_weight_model2.hdf5')
    y_pred2 = best_weight_model2.predict(X_test)

    # 画accuracy曲线
    plt.plot(epochs2, acc2, 'r')
    plt.plot(epochs2, val_acc2, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])
    plt.savefig(result_save_path + '/' + str(model_dir_name) + '/weight模型2 准确率学习曲线.jpg')
    plt.cla()
    # 画loss曲线
    plt.plot(epochs2, loss2, 'r')
    plt.plot(epochs2, val_loss2, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    plt.savefig(result_save_path + '/' + str(model_dir_name) + '/weight模型2 损失学习曲线.jpg')
    plt.cla()

    y_predict2 = y_pred2
    weight_model2.summary()

    # 模型预测MSE,RMES,R2
    mse = mean_squared_error(weights_Y_test_2, y_predict2)
    print("MSE= ", mse)
    rmse = np.sqrt(mse)
    print('RMSE=', rmse)
    r2 = r2_score(weights_Y_test_2, y_predict2)
    print('R2=', r2)
    model2score_pd = pd.DataFrame({'MSE': [mse], 'RMSE': [rmse], 'R2': [r2]})
    model2score_pd.to_excel(result_save_path + '/' + str(model_dir_name) + '/weight模型2 score.xlsx', index=False)
    model2score_pd.to_pickle(result_save_path + '/' + str(model_dir_name) + '/weight模型2 score.pkl')
    '''
    # 第三个weights模型
    weight_model3 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    weight_model3.fit(X_train, weights_Y_train_1)
    y_pred = weight_model3.predict(X_test)
    print("Accuracy:", accuracy_score(weights_Y_test_1, y_pred))
    print("Classification Report:\n", classification_report(weights_Y_test_1, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(weights_Y_test_1, y_pred))
    '''


    # 测试集result名称
    test_result_name = []
    for i in test_pd['result_name']:
        if i not in test_result_name:
            test_result_name.append(i)


    test_pd['M0 possibility'] = Y_test_predict[:, 0]
    test_pd['M1 possibility'] = Y_test_predict[:, 1]
    test_pd['M2 possibility'] = Y_test_predict[:, 2]
    test_pd['predict category by point'] = Y_test_pred
    test_pd['weight model 1 prediction'] = y_predict1
    test_pd['weight model 2 prediction'] = y_predict2

    # 根据每个点的预测结果，投票得到整个细胞的分类
    test_pd['predict category by result without weight'] = [-1 for i in range(test_pd.shape[0])]
    test_pd['weighted vote without weight'] = [[] for i in range(test_pd.shape[0])]
    test_pd['weighted vote dic without weight'] = [{} for i in range(test_pd.shape[0])]
    test_pd['predict category by result without weight model'] = [-1 for i in range(test_pd.shape[0])]
    test_pd['weighted vote without weight model'] = [[] for i in range(test_pd.shape[0])]
    test_pd['weighted vote dic without weight model'] = [{} for i in range(test_pd.shape[0])]
    test_pd['predict category by result with weight model1'] = [-1 for i in range(test_pd.shape[0])]
    test_pd['weighted vote with weight model1'] = [[] for i in range(test_pd.shape[0])]
    test_pd['weighted vote dic with weight model1'] = [{} for i in range(test_pd.shape[0])]
    test_pd['predict category by result with weight model2'] = [-1 for i in range(test_pd.shape[0])]
    test_pd['weighted vote with weight model2'] = [[] for i in range(test_pd.shape[0])]
    test_pd['weighted vote dic with weight model2'] = [{} for i in range(test_pd.shape[0])]

    test_pd['predict category by result without weight'] = test_pd['predict category by result without weight'].astype('object')
    test_pd['weighted vote without weight'] = test_pd['weighted vote without weight'].astype('object')
    test_pd['weighted vote dic without weight'] = test_pd['weighted vote dic without weight'].astype('object')
    test_pd['predict category by result without weight model'] = test_pd['predict category by result without weight model'].astype('object')
    test_pd['weighted vote without weight model'] = test_pd['weighted vote without weight model'].astype('object')
    test_pd['weighted vote dic without weight model'] = test_pd['weighted vote dic without weight model'].astype('object')
    test_pd['predict category by result with weight model1'] = test_pd['predict category by result with weight model1'].astype('object')
    test_pd['weighted vote with weight model1'] = test_pd['weighted vote with weight model1'].astype('object')
    test_pd['weighted vote dic with weight model1'] = test_pd['weighted vote dic with weight model1'].astype('object')
    test_pd['predict category by result with weight model2'] = test_pd['predict category by result with weight model2'].astype('object')
    test_pd['weighted vote with weight model2'] = test_pd['weighted vote with weight model2'].astype('object')
    test_pd['weighted vote dic with weight model2'] = test_pd['weighted vote dic with weight model2'].astype('object')
    for result_name in test_result_name:
        a = []
        point_index_list = test_pd[test_pd['result_name'] == result_name].index.tolist()
        weighted_vote_dic_without_weight = {0: 0, 1: 0, 2: 0}  # 用来保存一个细胞内每个像素点的投票结果加权后的结果
        weighted_vote_dic_without_weight_model = {0: 0, 1: 0, 2: 0}
        weighted_vote_dic_with_weight_model1 = {0: 0, 1: 0, 2: 0}
        weighted_vote_dic_with_weight_model2 = {0: 0, 1: 0, 2: 0}
        for i in point_index_list:
            weighted_vote_dic_without_weight[test_pd['predict category by point'][i]] += 1
            weighted_vote_dic_without_weight_model[test_pd['predict category by point'][i]] += Y_test_predict[i, test_pd['predict category by point'][i]]  # 权重为模型预测类别的具体预测概率值
            weighted_vote_dic_with_weight_model1[test_pd['predict category by point'][i]] += float(y_predict1[i]) * Y_test_predict[i, test_pd['predict category by point'][i]]  # 权重为权重模型1对该点预测置信度的估计*模型预测类别的具体预测概率值
            weighted_vote_dic_with_weight_model2[test_pd['predict category by point'][i]] += float(y_predict2[i]) * Y_test_predict[i, test_pd['predict category by point'][i]]  # 权重为权重模型2对该点预测置信度的估计*模型预测类别的具体预测概率值
        weighted_vote_list_without_weight = sorted((list(weighted_vote_dic_without_weight.items())), key=lambda x: x[1], reverse=True)
        weighted_vote_list_without_weight_model = sorted((list(weighted_vote_dic_without_weight_model.items())), key=lambda x: x[1], reverse=True)
        weighted_vote_list_with_weight_model1 = sorted((list(weighted_vote_dic_with_weight_model1.items())), key=lambda x: x[1], reverse=True)
        weighted_vote_list_with_weight_model2 = sorted((list(weighted_vote_dic_with_weight_model2.items())), key=lambda x: x[1], reverse=True)
        for i in point_index_list:
            # 每个样本的像素点投票加权计算的结果
            test_pd.at[i, 'weighted vote without weight'] = weighted_vote_list_without_weight
            test_pd.at[i, 'weighted vote dic without weight'] = weighted_vote_dic_without_weight
            test_pd.at[i, 'weighted vote without weight model'] = weighted_vote_list_without_weight_model
            test_pd.at[i, 'weighted vote dic without weight model'] = weighted_vote_dic_without_weight_model
            test_pd.at[i, 'weighted vote with weight model1'] = weighted_vote_list_with_weight_model1
            test_pd.at[i, 'weighted vote dic with weight model1'] = weighted_vote_dic_with_weight_model1
            test_pd.at[i, 'weighted vote with weight model2'] = weighted_vote_list_with_weight_model2
            test_pd.at[i, 'weighted vote dic with weight model2'] = weighted_vote_dic_with_weight_model2
            # 像素点投票加权计算的结果作为样本的预测结果
            test_pd.at[i, 'predict category by result without weight'] = weighted_vote_list_without_weight[0][0]
            test_pd.at[i, 'predict category by result without weight model'] = weighted_vote_list_without_weight_model[0][0]
            test_pd.at[i, 'predict category by result with weight model1'] = weighted_vote_list_with_weight_model1[0][0]
            test_pd.at[i, 'predict category by result with weight model2'] = weighted_vote_list_with_weight_model2[0][0]
    test_pd.to_excel(result_save_path + '/' + str(model_dir_name) + '/测试集样品逐点数据及预测结果.xlsx', index=False)
    test_pd.to_pickle(result_save_path + '/' + str(model_dir_name) + '/测试集样品逐点数据及预测结果.pkl')

    # 寻找并保存预测错的点和样本的数据
    wrong_predict_point_pd = test_pd[test_pd['predict category by point'] != test_pd['target']]
    wrong_predict_result_pd_without_weight = test_pd[test_pd['predict category by result without weight'] != test_pd['target']]
    wrong_predict_result_pd_without_weight_model = test_pd[test_pd['predict category by result without weight model'] != test_pd['target']]
    wrong_predict_result_pd_with_weight_model1 = test_pd[test_pd['predict category by result with weight model1'] != test_pd['target']]
    wrong_predict_result_pd_with_weight_model2 = test_pd[test_pd['predict category by result with weight model2'] != test_pd['target']]
    wrong_predict_point_pd.to_excel(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的点.xlsx', index=False)
    wrong_predict_point_pd.to_pickle(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的点.pkl')
    wrong_predict_result_pd_without_weight.to_excel(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的样本 without weight.xlsx', index=False)
    wrong_predict_result_pd_without_weight.to_pickle(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的样本 without weight.pkl')
    wrong_predict_result_pd_without_weight_model.to_excel(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的样本 without weight model.xlsx', index=False)
    wrong_predict_result_pd_without_weight_model.to_pickle(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的样本 without weight model.pkl')
    wrong_predict_result_pd_with_weight_model1.to_excel(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的样本 with weight model1.xlsx', index=False)
    wrong_predict_result_pd_with_weight_model1.to_pickle(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的样本 with weight model1.pkl')
    wrong_predict_result_pd_with_weight_model2.to_excel(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的样本 with weight model2.xlsx', index=False)
    wrong_predict_result_pd_with_weight_model2.to_pickle(result_save_path + '/' + str(model_dir_name) + '/测试集预测错误的样本 with weight model2.pkl')
    # 预测错误数量统计
    wrong_predict_result_name_without_weight = []
    for i in wrong_predict_result_pd_without_weight['result_name']:
        if i not in wrong_predict_result_name_without_weight:
            wrong_predict_result_name_without_weight.append(i)
    wrong_predict_result_name_without_weight_model = []
    for i in wrong_predict_result_pd_without_weight_model['result_name']:
        if i not in wrong_predict_result_name_without_weight_model:
            wrong_predict_result_name_without_weight_model.append(i)
    wrong_predict_result_name_with_weight_model1 = []
    for i in wrong_predict_result_pd_with_weight_model1['result_name']:
        if i not in wrong_predict_result_name_with_weight_model1:
            wrong_predict_result_name_with_weight_model1.append(i)
    wrong_predict_result_name_with_weight_model2 = []
    for i in wrong_predict_result_pd_with_weight_model2['result_name']:
        if i not in wrong_predict_result_name_with_weight_model2:
            wrong_predict_result_name_with_weight_model2.append(i)
    wrong_predict_point_result_number_pd = pd.DataFrame(
        {'total point number': [test_pd.shape[0]],
         'wrong point number': [wrong_predict_point_pd.shape[0]],
         'wrong point proportion': [wrong_predict_point_pd.shape[0] / test_pd.shape[0]],
         'total result number': [len(test_result_name)],
         'wrong result number without weight': [len(wrong_predict_result_name_without_weight)],
         'wrong result proportion without weight': [len(wrong_predict_result_name_without_weight) / len(test_result_name)],
         'wrong result number without weight model': [len(wrong_predict_result_name_without_weight_model)],
         'wrong result proportion without weight model': [len(wrong_predict_result_name_without_weight_model) / len(test_result_name)],
         'wrong result number with weight model1': [len(wrong_predict_result_name_with_weight_model1)],
         'wrong result proportion with weight model1': [len(wrong_predict_result_name_with_weight_model1) / len(test_result_name)],
         'wrong result number with weight model2': [len(wrong_predict_result_name_with_weight_model2)],
         'wrong result proportion with weight model2': [len(wrong_predict_result_name_with_weight_model2) / len(test_result_name)]})
    wrong_predict_point_result_number_pd.to_excel(result_save_path + '/' + str(model_dir_name) + '/预测错误的点和样本的数量和比例统计.xlsx', index=False)
    wrong_predict_point_result_number_pd.to_pickle(result_save_path + '/' + str(model_dir_name) + '/预测错误的点和样本的数量和比例统计.pkl')
    model_accuracy_pd_model_name_list.append(model_dir_name)
    model_accuracy_pd_point_test_accuracy_list.append(1 - (wrong_predict_point_pd.shape[0] / test_pd.shape[0]))
    model_accuracy_pd_result_test_accuracy_without_weight_list.append(1 - (len(wrong_predict_result_name_without_weight) / len(test_result_name)))
    model_accuracy_pd_result_test_accuracy_without_weight_model_list.append(1 - (len(wrong_predict_result_name_without_weight_model).shape[0] / len(test_result_name)))
    model_accuracy_pd_result_test_accuracy_with_weight_model1_list.append(1 - (len(wrong_predict_result_name_with_weight_model1) / len(test_result_name)))
    model_accuracy_pd_result_test_accuracy_with_weight_model2_list.append(1 - (len(wrong_predict_result_name_with_weight_model2).shape[0] / len(test_result_name)))


    # 将每个细胞的逐点分类结果作图
    # 根据每个点三个分类预测概率计算该点三分类图的像素图以及投票柱状图，并作图保存
    for result_name in test_result_name:
        # 获取图片尺寸
        f = open(pict_test_data_dir + '/CorrMorpho/' + result_name[:2] + '/' + result_name.replace('MechD', 'CorrMorpho'))
        line = f.readline()
        r_list = []
        while line:
            num = list(map(float, line.split()))
            r_list.append(num)
            line = f.readline()
        f.close()
        r_array = np.array(r_list)

        vote_to_M0_num_point = 0
        vote_to_M1_num_point = 0
        vote_to_M2_num_point = 0
        # 根据原始数据的尺寸创建作图尺寸
        M0_map = np.zeros(r_array.shape, dtype=int)
        M1_map = np.zeros(r_array.shape, dtype=int)
        M2_map = np.zeros(r_array.shape, dtype=int)
        point_index_list = test_pd[test_pd['result_name'] == result_name].index.tolist()
        for i in point_index_list:
            M0_map[test_pd.iloc[i]['site'][0], test_pd.iloc[i]['site'][1]] = round(test_pd.iloc[i]['M0 possibility'] * 255)
            M1_map[test_pd.iloc[i]['site'][0], test_pd.iloc[i]['site'][1]] = round(test_pd.iloc[i]['M1 possibility'] * 255)
            M2_map[test_pd.iloc[i]['site'][0], test_pd.iloc[i]['site'][1]] = round(test_pd.iloc[i]['M2 possibility'] * 255)
            if test_pd.iloc[i]['predict category by point'] == 0:
                vote_to_M0_num_point += 1
            if test_pd.iloc[i]['predict category by point'] == 1:
                vote_to_M1_num_point += 1
            if test_pd.iloc[i]['predict category by point'] == 2:
                vote_to_M2_num_point += 1
        pixel_num_result = vote_to_M0_num_point + vote_to_M1_num_point + vote_to_M2_num_point
        plt.bar(['M0', 'M1', 'M2'], [100 * vote_to_M0_num_point / pixel_num_result, 100 * vote_to_M1_num_point / pixel_num_result, 100 * vote_to_M2_num_point / pixel_num_result])
        plt.title('Pixel voting results without weight')
        plt.xlabel('category')
        plt.ylabel('percentage')
        plt.savefig(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化/' + result_name[: -9] + ' without weight.jpg')
        plt.cla()    # 清空图片

        wv_dic = test_pd.iloc[point_index_list[0]]['weighted vote dic without weight model']  # 加权投票结果列记录的是样本的结果，一个样本内所有像素点的这一列是相同的，选一个做代表
        wv_sum = 0
        for v in wv_dic:
            wv_sum += wv_dic[v]
        plt.bar(['M0', 'M1', 'M2'], [100 * wv_dic[0] / wv_sum, 100 * wv_dic[1] / wv_sum, 100 * wv_dic[2] / wv_sum])
        plt.title('Pixel voting results with weight')
        plt.xlabel('category')
        plt.ylabel('percentage')
        plt.savefig(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化/' + result_name[: -9] + ' with weight model.jpg')
        plt.cla()  # 清空图片

        wv_dic = test_pd.iloc[point_index_list[0]]['weighted vote dic with weight model1']
        wv_sum = 0
        for v in wv_dic:
            wv_sum += wv_dic[v]
        plt.bar(['M0', 'M1', 'M2'], [100 * wv_dic[0] / wv_sum, 100 * wv_dic[1] / wv_sum, 100 * wv_dic[2] / wv_sum])
        plt.title('Pixel voting results with weight')
        plt.xlabel('category')
        plt.ylabel('percentage')
        plt.savefig(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化/' + result_name[: -9] + ' with weight model1.jpg')
        plt.cla()  # 清空图片

        wv_dic = test_pd.iloc[point_index_list[0]]['weighted vote dic with weight model2']
        wv_sum = 0
        for v in wv_dic:
            wv_sum += wv_dic[v]
        plt.bar(['M0', 'M1', 'M2'], [100 * wv_dic[0] / wv_sum, 100 * wv_dic[1] / wv_sum, 100 * wv_dic[2] / wv_sum])
        plt.title('Pixel voting results with weight')
        plt.xlabel('category')
        plt.ylabel('percentage')
        plt.savefig(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化/' + result_name[: -9] + ' with weight model2.jpg')
        plt.cla()  # 清空图片

        img_M0 = Image.new('RGB', (r_array.shape[0] * amplify, r_array.shape[1] * amplify), (0, 0, 0))
        draw = ImageDraw.Draw(img_M0)
        for m in range(r_array.shape[0]):
            for n in range(r_array.shape[1]):
                for a in range(amplify):
                    for b in range(amplify):
                        draw.point((m * amplify + a, n * amplify + b), fill=(M0_map[m, n], 0, 0))
        img_M0.save(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化/' + result_name[: -9] + '-M0.jpg')

        img_M1 = Image.new('RGB', (r_array.shape[0] * amplify, r_array.shape[1] * amplify), (0, 0, 0))
        draw = ImageDraw.Draw(img_M1)
        for m in range(r_array.shape[0]):
            for n in range(r_array.shape[1]):
                for a in range(amplify):
                    for b in range(amplify):
                        draw.point((m * amplify + a, n * amplify + b), fill=(0, M1_map[m, n], 0))
        img_M1.save(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化/' + result_name[: -9] + '-M1.jpg')

        img_M2 = Image.new('RGB', (r_array.shape[0] * amplify, r_array.shape[1] * amplify), (0, 0, 0))
        draw = ImageDraw.Draw(img_M2)
        for m in range(r_array.shape[0]):
            for n in range(r_array.shape[1]):
                for a in range(amplify):
                    for b in range(amplify):
                        draw.point((m * amplify + a, n * amplify + b), fill=(0, 0, M2_map[m, n]))
        img_M2.save(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化/' + result_name[: -9] + '-M2.jpg')

        img_all_category = Image.new('RGB', (r_array.shape[0] * amplify, r_array.shape[1] * amplify), (0, 0, 0))
        draw = ImageDraw.Draw(img_all_category)
        for m in range(r_array.shape[0]):
            for n in range(r_array.shape[1]):
                for a in range(amplify):
                    for b in range(amplify):
                        draw.point((m * amplify + a, n * amplify + b), fill=(M0_map[m, n], M1_map[m, n], M2_map[m, n]))
        img_all_category.save(result_save_path + '/' + str(model_dir_name) + '/逐点分类可视化/' + result_name[: -9] + '-all category.jpg')

model_accuracy_pd['model name'] = model_accuracy_pd_model_name_list
model_accuracy_pd['point test accuracy'] = model_accuracy_pd_point_test_accuracy_list
model_accuracy_pd['result test accuracy without weight'] = model_accuracy_pd_result_test_accuracy_without_weight_list
model_accuracy_pd['result test accuracy without weight model'] = model_accuracy_pd_result_test_accuracy_without_weight_model_list
model_accuracy_pd['result test accuracy with weight model1'] = model_accuracy_pd_result_test_accuracy_with_weight_model1_list
model_accuracy_pd['result test accuracy with weight model2'] = model_accuracy_pd_result_test_accuracy_with_weight_model2_list
model_accuracy_pd.to_excel(result_save_path + '/模型准确率.xlsx', index=False)
model_accuracy_pd.to_pickle(result_save_path + '/模型准确率.pkl')

