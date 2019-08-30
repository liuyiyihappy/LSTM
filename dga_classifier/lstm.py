"""Train and test LSTM classifier--Keras"""
import dga_classifier.data as data
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#from sklearn.cross_validation import train_test_split
import os
os.environ['KERAS_BACKEND']='tensorflow'

##################################定义模型######################################
def build_model(max_features, maxlen):
    """Build LSTM model"""
    model = Sequential()    #定义一个基本的神经网络模型
    
   # 添加一个嵌入层，将每个字符转换为128个浮点数的向量（128不是幻数）
   # 一旦这个层被训练（输入字符和输出128个浮点数），每个字符基本上都经过一次查找。
   # max_features定义有效字符数。input_length是我们将要传递给神经网络的最大长度字符串
    model.add(Embedding(max_features, 128, input_length=maxlen))  
   
    #添加了一个LSTM层，128表示我们内部状态的维度，维度越大对模型的描述也就更具体，在这里128刚好适合我们的需求
    model.add(LSTM(128))   
    
    model.add(Dropout(0.5))  #防止模型过拟合
    
    model.add(Dense(1))    #Dense层（全连接层）
    
    model.add(Activation('sigmoid'))  #激活函数sigmoid，把输入的连续实值“压缩”到0和1之间。如果是非常大的负数，那么输出就是0；如果是非常大的正数，输出就是1

    model.compile(loss='binary_crossentropy',    
                  optimizer='rmsprop')    #使用优化器对交叉熵损失函数进行优化。RMSProp是随机梯度下降的变体，并且往往对循环神经网络非常有效。

    return model

################################预处理代码#############################################
def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    indata = data.get_data()

    # Extract data and labels
    X = [x[1] for x in indata]

    labels = [x[0] for x in indata]   #labels全是corebot

    # Generate a dictionary of valid characters
    # 将每个字符串转换为表示每个可能字符的int数组，这种编码是任意的，但是应该从1开始（我们为结束序列token保留0）并且是连续的。
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])

    # Convert characters to int and pad
    # 将每个int数组填充至相同的长度。填充能让我们的toolbox更好地优化计算（理论上，LSTM不需要填充）
    # maxlen表示每个数组的长度。当阵列太长时，此函数将填充0和crop
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)

    # Convert labels to 0-1
    y = [0 if x == 'benign' else 1 for x in labels]    #y全是1

    final_data = []

    #使用ROC曲线分割我们的测试和训练集，以及评估我们的表现
    for fold in range(nfolds):
        print ("lstm:fold %u/%u" % (fold+1, nfolds))
        
        """创造训练集和测试集"""
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, 
                                                                           test_size=0.2)
        
        """创建模型"""
        print ('Build LSTM model...')
        model = build_model(max_features, maxlen)

        print ("Train LSTM model...")
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)  #0.05
        #print(y_holdout)


        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            """训练模型"""
            model.fit(X_train, y_train, batch_size=batch_size, epochs=1)

            """评估模型"""
            t_probs = model.predict_proba(X_holdout)   # model.predict返回的是类别，model.predict_proba返回的是分类概率
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)  # AUC即ROC曲线下面积，oc_auc_score直接根据真实值和预测值计算auc值，省略计算roc的过程，AUC值越大，性能越好

#报错！！！！y_holdout只有一个取值"""

            print ('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict_proba(X_test)

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print (sklearn.metrics.confusion_matrix(y_test, probs > .5))
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 2:
                    break

        final_data.append(out_data)

    return final_data
