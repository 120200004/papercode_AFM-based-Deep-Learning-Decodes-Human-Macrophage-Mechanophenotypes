import random
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.experimental import CosineDecayRestarts
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import confusion_matrix


def train_val_test_split(data_pd, category_name_list, category_position_in_name, save_dir, result_test_size=0.2, result_val_size=0.2, result_train_size=0.6):
    result_name = []
    result_name_by_category_dic = {}    # 分类别保存样本名称
    for i in category_name_list:
        result_name_by_category_dic[i] = []
    for i in data_pd['result_name']:
        if i not in result_name:
            result_name.append(i)
            result_name_by_category_dic[i[category_position_in_name[0]:category_position_in_name[1]]].append(i)
    result_num_by_category_dic = {}
    for i in category_name_list:
        result_num_by_category_dic[i] = [len(result_name_by_category_dic[i])]
        point_num_by_category = 0
        for j in result_name_by_category_dic[i]:
            point_index = data_pd[data_pd['result_name'] == j].index.tolist()
            point_num_by_category += len(point_index)
        result_num_by_category_dic[i].append(point_num_by_category)
    result_num_by_category_dic['total'] = [len(result_name)]    # 可用样本总数
    result_num_by_category_dic['total'].append(data_pd.shape[0])    # 可用点总数
    result_num_by_category_pd = pd.DataFrame(result_num_by_category_dic, index=['result', 'point'])
    result_num_by_category_pd.to_excel(save_dir + '/分类别样本和点的数量.xlsx', index=True)
    result_num_by_category_pd.to_pickle(save_dir + '/分类别样本和点的数量.pkl')
    with open(save_dir + '/分类别样本名称.pkl', 'wb') as f:
        pickle.dump(result_name_by_category_dic, f)

    result_test_name_by_category_dic = {}
    result_test_name_by_category_dic['total'] = []
    result_test_num_by_category_dic = {}
    a = 0
    for i in category_name_list:
        result_test_name_by_category_dic[i] = random.sample(result_name_by_category_dic[i],
                                                           round(result_num_by_category_pd[i][0] * result_test_size))
        result_test_num_by_category_dic[i] = ([float(len(result_test_name_by_category_dic[i])), 0])
        a += len(result_test_name_by_category_dic[i])
        result_test_name_by_category_dic['total'].extend(result_test_name_by_category_dic[i])
    result_test_num_by_category_dic['total'] = [float(a), 0]
    random.shuffle(result_test_name_by_category_dic['total'])
    with open(save_dir + '/测试集分类别样本名称.pkl', 'wb') as f:
        pickle.dump(result_test_name_by_category_dic, f)
    test_pd = pd.DataFrame(columns=data_pd.columns)
    index_list = []
    for i in result_test_name_by_category_dic['total']:
        for j in category_name_list:
            if i in result_test_name_by_category_dic[j]:
                point_index = data_pd[data_pd['result_name'] == i].index.tolist()
                index_list.extend(point_index)
                cnt = len(point_index)
                result_test_num_by_category_dic['total'][1] += cnt
                result_test_num_by_category_dic[j][1] += cnt
    if index_list:
        test_pd = data_pd.iloc[index_list].reset_index(drop=True)
    for i in category_name_list:
        result_test_num_by_category_dic[i].extend([result_test_num_by_category_dic[i][0] /
                                                   result_test_num_by_category_dic['total'][0],
                                                   result_test_num_by_category_dic[i][1] /
                                                   result_test_num_by_category_dic['total'][1]])
    result_test_num_by_category_dic['total'].extend([result_test_num_by_category_dic['total'][0] /
                                                     result_num_by_category_pd['total'][0],
                                                     result_test_num_by_category_dic['total'][1] /
                                                     result_num_by_category_pd['total'][1]])
    result_test_num_by_category_pd = pd.DataFrame(result_test_num_by_category_dic, index=['result', 'point',
                                                                                          'result_proportion',
                                                                                          'point_proportion'])
    result_test_num_by_category_pd.to_excel(save_dir + '/测试集分类别样本和点的数量.xlsx', index=True)
    result_test_num_by_category_pd.to_pickle(save_dir + '/测试集分类别样本和点的数量.pkl')
    test_pd.to_excel(save_dir + '/测试集数据.xlsx', index=False)
    test_pd.to_pickle(save_dir + '/测试集数据.pkl')
    result_name = [i for i in result_name if i not in result_test_name_by_category_dic['total']]
    for i in category_name_list:
        result_name_by_category_dic[i] = [j for j in result_name_by_category_dic[i]
                                          if j not in result_test_name_by_category_dic[i]]
    result_val_name_by_category_dic = {}
    result_val_name_by_category_dic['total'] = []
    result_val_num_by_category_dic = {}
    a = 0
    for i in category_name_list:
        result_val_name_by_category_dic[i] = random.sample(result_name_by_category_dic[i],
                                                            round(result_num_by_category_pd[i][0] * result_val_size))
        result_val_num_by_category_dic[i] = [float(len(result_val_name_by_category_dic[i])), 0]
        a += len(result_val_name_by_category_dic[i])
        result_val_name_by_category_dic['total'].extend(result_val_name_by_category_dic[i])
    result_val_num_by_category_dic['total'] = [float(a), 0]
    random.shuffle(result_val_name_by_category_dic['total'])
    with open(save_dir + '/验证集分类别样本名称.pkl', 'wb') as f:
        pickle.dump(result_val_name_by_category_dic, f)
    val_pd = pd.DataFrame(columns=data_pd.columns)
    index_list = []
    for i in result_val_name_by_category_dic['total']:
        for j in category_name_list:
            if i in result_val_name_by_category_dic[j]:
                point_index = data_pd[data_pd['result_name'] == i].index.tolist()
                index_list.extend(point_index)
                cnt = len(point_index)
                result_val_num_by_category_dic['total'][1] += cnt
                result_val_num_by_category_dic[j][1] += cnt
    if index_list:
        val_pd = data_pd.iloc[index_list].reset_index(drop=True)
    for i in category_name_list:
        result_val_num_by_category_dic[i].extend([result_val_num_by_category_dic[i][0] /
                                                  result_val_num_by_category_dic['total'][0],
                                                  result_val_num_by_category_dic[i][1] /
                                                  result_val_num_by_category_dic['total'][1]])
    result_val_num_by_category_dic['total'].extend([result_val_num_by_category_dic['total'][0] /
                                                     result_num_by_category_pd['total'][0],
                                                     result_val_num_by_category_dic['total'][1] /
                                                     result_num_by_category_pd['total'][1]])
    result_val_num_by_category_pd = pd.DataFrame(result_val_num_by_category_dic, index=['result', 'point',
                                                                                          'result_proportion',
                                                                                          'point_proportion'])
    result_val_num_by_category_pd.to_excel(save_dir + '/验证集分类别样本和点的数量.xlsx', index=True)
    result_val_num_by_category_pd.to_pickle(save_dir + '/验证集分类别样本和点的数量.pkl')
    val_pd.to_excel(save_dir + '/验证集数据.xlsx', index=False)
    val_pd.to_pickle(save_dir + '/验证集数据.pkl')

    result_name = [i for i in result_name if i not in result_val_name_by_category_dic['total']]
    for i in category_name_list:
        result_name_by_category_dic[i] = [j for j in result_name_by_category_dic[i]
                                          if j not in result_val_name_by_category_dic[i]]
    result_train_name_by_category_dic = {}
    result_train_name_by_category_dic['total'] = []
    result_train_num_by_category_dic = {}
    a = 0
    for i in category_name_list:
        result_train_name_by_category_dic[i] = result_name_by_category_dic[i]
        result_train_num_by_category_dic[i] = [float(len(result_train_name_by_category_dic[i])), 0]
        a += len(result_train_name_by_category_dic[i])
        result_train_name_by_category_dic['total'].extend(result_train_name_by_category_dic[i])
    result_train_num_by_category_dic['total'] = [float(a), 0]
    random.shuffle(result_train_name_by_category_dic['total'])
    with open(save_dir + '/训练集分类别样本名称.pkl', 'wb') as f:
        pickle.dump(result_train_name_by_category_dic, f)
    train_pd = pd.DataFrame(columns=data_pd.columns)
    index_list = []
    for i in result_train_name_by_category_dic['total']:
        for j in category_name_list:
            if i in result_train_name_by_category_dic[j]:
                point_index = data_pd[data_pd['result_name'] == i].index.tolist()
                index_list.extend(point_index)
                cnt = len(point_index)
                result_train_num_by_category_dic['total'][1] += cnt
                result_train_num_by_category_dic[j][1] += cnt
    if index_list:
        train_pd = data_pd.iloc[index_list].reset_index(drop=True)
    for i in category_name_list:
        result_train_num_by_category_dic[i].extend([result_train_num_by_category_dic[i][0] /
                                                  result_train_num_by_category_dic['total'][0],
                                                  result_train_num_by_category_dic[i][1] /
                                                  result_train_num_by_category_dic['total'][1]])
    result_train_num_by_category_dic['total'].extend([result_train_num_by_category_dic['total'][0] /
                                                     result_num_by_category_pd['total'][0],
                                                     result_train_num_by_category_dic['total'][1] /
                                                     result_num_by_category_pd['total'][1]])
    result_train_num_by_category_pd = pd.DataFrame(result_train_num_by_category_dic, index=['result', 'point',
                                                                                          'result_proportion',
                                                                                          'point_proportion'])
    result_train_num_by_category_pd.to_excel(save_dir + '/训练集分类别样本和点的数量.xlsx', index=True)
    result_train_num_by_category_pd.to_pickle(save_dir + '/训练集分类别样本和点的数量.pkl')
    train_pd.to_excel(save_dir + '/训练集数据.xlsx', index=False)
    train_pd.to_pickle(save_dir + '/训练集数据.pkl')
    print(result_train_num_by_category_pd)
    print()
    return train_pd, val_pd, test_pd


def normalization(data_pd, pd_save_name, standardized_normalized_save_name, feature_array_save_name, target_array_save_name, save_dir):
    feature_point_pd_normalized = pd.DataFrame()
    feature_point_pd_normalized['Adh_point'] = 2 * ((data_pd['Adh_point'] - data_pd['Adh_point'].min()) / (data_pd['Adh_point'].max() - data_pd['Adh_point'].min())) - 1
    feature_point_pd_normalized['MechD_point'] = 2 * ((data_pd['MechD_point'] - data_pd['MechD_point'].min()) / (data_pd['MechD_point'].max() - data_pd['MechD_point'].min())) - 1
    feature_point_pd_normalized['MechS_point'] = 2 * ((data_pd['MechS_point'] - data_pd['MechS_point'].min()) / (data_pd['MechS_point'].max() - data_pd['MechS_point'].min())) - 1
    feature_point_pd_normalized['Morpho_point'] = 2 * ((data_pd['Morpho_point'] - data_pd['Morpho_point'].min()) / (data_pd['Morpho_point'].max() - data_pd['Morpho_point'].min())) - 1
    feature_point_pd_normalized['Distance_to_center_divide_max_point'] = 2 * ((data_pd['Distance_to_center_divide_max_point'] - data_pd['Distance_to_center_divide_max_point'].min()) / (data_pd['Distance_to_center_divide_max_point'].max() - data_pd['Distance_to_center_divide_max_point'].min())) - 1
    feature_point_pd_normalized['Distance_to_center_order_percentage_point'] = 2 * ((data_pd['Distance_to_center_order_percentage_point'] - data_pd['Distance_to_center_order_percentage_point'].min()) / (data_pd['Distance_to_center_order_percentage_point'].max() - data_pd['Distance_to_center_order_percentage_point'].min())) - 1
    feature_point_pd_normalized.to_excel(save_dir + '/' + pd_save_name + '.xlsx', index=False)
    feature_point_pd_normalized.to_pickle(save_dir + '/' + pd_save_name + '.pkl')
    standardized_normalized_pd = pd.DataFrame()
    standardized_normalized_pd['Adh_point_mean'] = [data_pd['Adh_point'].mean()]
    standardized_normalized_pd['Adh_point_std'] = [data_pd['Adh_point'].std()]
    standardized_normalized_pd['Adh_point_max'] = [data_pd['Adh_point'].max()]
    standardized_normalized_pd['Adh_point_min'] = [data_pd['Adh_point'].min()]
    standardized_normalized_pd['MechD_point_mean'] = [data_pd['MechD_point'].mean()]
    standardized_normalized_pd['MechD_point_std'] = [data_pd['MechD_point'].std()]
    standardized_normalized_pd['MechD_point_max'] = [data_pd['MechD_point'].max()]
    standardized_normalized_pd['MechD_point_min'] = [data_pd['MechD_point'].min()]
    standardized_normalized_pd['MechS_point_mean'] = [data_pd['MechS_point'].mean()]
    standardized_normalized_pd['MechS_point_std'] = [data_pd['MechS_point'].std()]
    standardized_normalized_pd['MechS_point_max'] = [data_pd['MechS_point'].max()]
    standardized_normalized_pd['MechS_point_min'] = [data_pd['MechS_point'].min()]
    standardized_normalized_pd['Morpho_point_mean'] = [data_pd['Morpho_point'].mean()]
    standardized_normalized_pd['Morpho_point_std'] = [data_pd['Morpho_point'].std()]
    standardized_normalized_pd['Morpho_point_max'] = [data_pd['Morpho_point'].max()]
    standardized_normalized_pd['Morpho_point_min'] = [data_pd['Morpho_point'].min()]
    standardized_normalized_pd['Distance_to_center_divide_max_point_mean'] = [data_pd['Distance_to_center_divide_max_point'].mean()]
    standardized_normalized_pd['Distance_to_center_divide_max_point_std'] = [data_pd['Distance_to_center_divide_max_point'].std()]
    standardized_normalized_pd['Distance_to_center_divide_max_point_max'] = [data_pd['Distance_to_center_divide_max_point'].max()]
    standardized_normalized_pd['Distance_to_center_divide_max_point_min'] = [data_pd['Distance_to_center_divide_max_point'].min()]
    standardized_normalized_pd['Distance_to_center_order_percentage_point_mean'] = [data_pd['Distance_to_center_order_percentage_point'].mean()]
    standardized_normalized_pd['Distance_to_center_order_percentage_point_std'] = [data_pd['Distance_to_center_order_percentage_point'].std()]
    standardized_normalized_pd['Distance_to_center_order_percentage_point_max'] = [data_pd['Distance_to_center_order_percentage_point'].max()]
    standardized_normalized_pd['Distance_to_center_order_percentage_point_min'] = [data_pd['Distance_to_center_order_percentage_point'].min()]
    standardized_normalized_pd.to_excel(save_dir + '/' + standardized_normalized_save_name + '.xlsx', index=False)
    standardized_normalized_pd.to_pickle(save_dir + '/' + standardized_normalized_save_name + '.pkl')
    feature_point_list_standardized_normalized = [list(feature_point_pd_normalized.iloc[i]) for i in range(feature_point_pd_normalized.shape[0])]
    feature_point_array_standardized_normalized = np.array(feature_point_list_standardized_normalized)
    target_point_list = list(data_pd['target'])
    target_point_array = np.array(target_point_list)
    np.save(save_dir + '/' + feature_array_save_name + '.npy', feature_point_array_standardized_normalized)
    np.save(save_dir + '/' + target_array_save_name + '.npy', target_point_array)
    return feature_point_array_standardized_normalized, target_point_array


def model_training(X_train, Y_train, X_val, Y_val, X_test, Y_test, save_dir, base_save_dir):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.GaussianNoise(0.1))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.GaussianNoise(0.1))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.GaussianNoise(0.1))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

    steps_per_epoch = len(X_train) // 4
    cosine_scheduler = CosineDecayRestarts(initial_learning_rate=0.01, first_decay_steps=5 * steps_per_epoch, t_mul=2, m_mul=0.5, alpha=0.5)
    optimizer = tf.keras.optimizers.SGD(learning_rate=cosine_scheduler, momentum=0.95, nesterov=True)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    callbacks = [ModelCheckpoint(filepath=save_dir + '/best_model.hdf5', monitor='val_accuracy', mode='max', save_best_only=True, save_format='h5'),
                 EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)]


    history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_val, Y_val), callbacks=callbacks, shuffle=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    print(save_dir + '/best_model.hdf5')
    best_model = tf.keras.models.load_model(save_dir + '/best_model.hdf5')
    y_pred = best_model.predict(X_test)

    history_pd = pd.DataFrame()
    history_pd['epoch'] = epochs
    history_pd['train acc'] = acc
    history_pd['val acc'] = val_acc
    history_pd['train loss'] = loss
    history_pd['val loss'] = val_loss
    history_pd.to_excel(save_dir + '/模型学习率曲线数据.xlsx', index=False)
    history_pd.to_pickle(save_dir + '/模型学习率曲线数据.pkl')

    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])
    plt.savefig(save_dir + '/准确率学习曲线.jpg')
    plt.close('all')
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    plt.savefig(save_dir + '/损失学习曲线.jpg')
    plt.close('all')

    y_predict = []
    for s in range(0, y_pred.shape[0]):
        if y_pred[s, 0] == max(y_pred[s, 0], y_pred[s, 1], y_pred[s, 2]):
            y_predict.append(0)
        elif y_pred[s, 1] == max(y_pred[s, 0], y_pred[s, 1], y_pred[s, 2]):
            y_predict.append(1)
        elif y_pred[s, 2] == max(y_pred[s, 0], y_pred[s, 1], y_pred[s, 2]):
            y_predict.append(2)
        else:
            continue
    y_predict = np.array(y_predict)
    best_model.summary()

    acc = precision_score(Y_test, y_predict, average=None)
    print("accurate for M0, M1, and M2= ", acc)
    accuracyscore = accuracy_score(Y_test, y_predict)
    print("accuracy is ", accuracyscore)
    accuracy_pd = pd.DataFrame(columns=['accurate for M0', 'accurate for M1', 'accurate for M2', 'total accuracy'])
    new_row = pd.DataFrame({'accurate for M0': acc[0], 'accurate for M1': acc[1], 'accurate for M2': acc[2], 'total accuracy': accuracyscore}, index=[0])
    accuracy_pd = pd.concat([accuracy_pd, new_row], ignore_index=True)
    accuracy_pd.to_excel(save_dir + '/模型预测准确率.xlsx', index=False)
    accuracy_pd.to_pickle(save_dir + '/模型预测准确率.pkl')

    confusion_matrix = confusion_matrix(Y_test, y_predict, labels=[0, 1, 2], sample_weight=None)
    cm_normalized_p = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    cm_normalized_n = confusion_matrix.astype('int')
    plt.matshow(cm_normalized_p, cmap=plt.get_cmap('Reds'))
    plt.colorbar()
    for i in range(len(cm_normalized_p)):
        for j in range(len(cm_normalized_p)):
            plt.annotate(round(cm_normalized_p[j, i], 3), xy=(i, j), color='g', fontsize=12,
                         horizontalalignment='center', verticalalignment='center')
    plt.title("Confusion Matrix", fontsize=12)
    plt.xlabel('Predict Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xticks(range(len(cm_normalized_p)), ['M0', 'M1', 'M2'], fontsize=12)
    plt.yticks(range(len(cm_normalized_p)), ['M0', 'M1', 'M2'], fontsize=12)
    plt.savefig(save_dir + '/Confusion Matrix.jpg')
    plt.savefig(base_save_dir + '/.jpg')
    plt.close('all')
    plt.matshow(cm_normalized_n, cmap=plt.get_cmap('Reds'))
    plt.colorbar()
    for i in range(len(cm_normalized_n)):
        for j in range(len(cm_normalized_n)):
            plt.annotate(round(cm_normalized_n[j, i], 3), xy=(i, j), color='g', fontsize=12,
                         horizontalalignment='center', verticalalignment='center')
    plt.title("Confusion Matrix", fontsize=12)
    plt.xlabel('Predict Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xticks(range(len(cm_normalized_n)), ['M0', 'M1', 'M2'], fontsize=12)
    plt.yticks(range(len(cm_normalized_n)), ['M0', 'M1', 'M2'], fontsize=12)
    plt.savefig(save_dir + '/Confusion Matrix_nunber.jpg')
    plt.close('all')

data_pd = pd.read_pickle('1-4/data_point_pd.pkl')
base_save_dir = '2-7-2-2'
category_name_list = ['M0', 'M1', 'M2']
category_position_in_name = [0, 2]


os.makedirs(base_save_dir)
save_dir = base_save_dir
train_pd, val_pd, test_pd = train_val_test_split(data_pd, category_name_list, category_position_in_name, save_dir, result_test_size=0.2, result_val_size=0.2, result_train_size=0.6)
X_train, Y_train = normalization(train_pd, '训练集数据归一化', '训练集数据归一化参数', '训练集数据特征归一化数组', '训练集标签数组', save_dir)
X_val, Y_val = normalization(val_pd, '验证集数据归一化', '验证集数据归一化参数', '验证集数据特征归一化数组', '验证集标签数组', save_dir)
X_test, Y_test = normalization(test_pd, '测试集数据归一化', '测试集数据归一化参数', '测试集数据特征归一化数组', '测试集标签数组', save_dir)
model_training(X_train, Y_train, X_val, Y_val, X_test, Y_test, save_dir, base_save_dir)

