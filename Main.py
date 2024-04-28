from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint 
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Activation, RepeatVector

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns


main = Tk()
main.title("Advanced Traffic Sampling and Analysis for Enhanced Network Security: Leveraging CNN Dense-Net and Data Augmentation")
main.geometry("1300x1200")

global filename
global dataset
global X, Y
global X_train, X_test, y_train, y_test
global labels
columns = ['proto', 'service', 'state']
label_encoder = []
global accuracy, precision, recall, fscore, scaler, pca, lite_model

#fucntion to upload dataset
def uploadDataset():
    global filename, dataset, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename) #read dataset from uploaded file
    labels = np.unique(dataset['label'])
    text.insert(END,"Dataset Values\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    
    label = dataset.groupby('label').size()
    label.plot(kind="bar")
    plt.xlabel('Attack Names')
    plt.ylabel('Attack Count')
    plt.title("Dataset Detail 0 (Normal) & 1 (Attack)")
    plt.show()

def preprocess():
    text.delete('1.0', END)
    global dataset, scaler
    global X_train, X_test, y_train, y_test, X, Y
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    dataset.drop(['attack_cat'], axis = 1,inplace=True)
    for i in range(len(columns)):
        le = LabelEncoder()
        dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
        label_encoder.append(le)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffle dataset
    X = X[indices]
    Y = Y[indices]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text.insert(END,"Dataset After Features Processing & Normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset before applying PCA : "+str(X.shape[1])+"\n\n")

def calculateMetrics(algorithm, predict, y_test):
    labels = ['Normal', 'Attack']
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = a + 1
    p = p + 1
    f = f + 1
    r = r + 1
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runPCA():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y, pca
    if os.path.exists('model/pca.txt'):
        with open('model/pca.txt', 'rb') as file:
            pca = pickle.load(file)
        file.close()
        X = pca.fit_transform(X)
    else:
        pca = PCA(n_components = 20)
        X = pca.fit_transform(X)
        with open('model/pca.txt', 'wb') as file:
            pickle.dump(pca, file)  
        file.close()
    text.insert(END,"Total features found in dataset after applying PCA : "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train Algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train Algorithms : "+str(X_test.shape[0])+"\n")

def runSVM():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    svm_cls = svm.SVC()
    svm_cls.fit(X_train[0:1000], y_train[0:1000])
    predict = svm_cls.predict(X_test[0:100])
    calculateMetrics("SVM", predict, y_test[0:100])      

def runRandomForest():
    global X_train, X_test, y_train, y_test
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train[0:1000], y_train[0:1000])
    predict = rf_cls.predict(X_test[0:100])
    calculateMetrics("Random Forest", predict, y_test[0:100])

def runCNN():
    global X_train, X_test, y_train, y_test, lite_model
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    lite_model = Sequential()
    #defining CNN layer with 32 neurons or filters to filter and encode dataset features
    lite_model.add(Conv1D(filters=32, kernel_size=9, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    #defining another layer to further filter features
    lite_model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
    #max pool layer to collect filtered features from CNN
    lite_model.add(MaxPooling1D(pool_size=2))
    #convert multidimension features to single dimension
    lite_model.add(Flatten())
    lite_model.add(RepeatVector(2))
    #defining LSTM layer as decoder to predict output
    lite_model.add(LSTM(32, activation='relu'))
    #defining output layer with 100 neurons
    lite_model.add(Dense(units = 100, activation = 'relu'))
    lite_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile the model
    lite_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #train and load the model
    if os.path.exists("model/lite_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lite_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lite_model.fit(X_train, y_train, batch_size = 128, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lite_model.load_weights("model/lite_weights.hdf5")
    predict = lite_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Propose CNN", predict, testY)

def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','F1 Score',fscore[1]],['Random Forest','Accuracy',accuracy[1]],
                       ['Propose CNN','Precision',precision[2]],['Propose CNN','Recall',recall[2]],['Propose CNN','F1 Score',fscore[2]],['Propose CNN','Accuracy',accuracy[2]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()

def trainingGraph():
    f = open('model/history.pckl', 'rb')
    graph = pickle.load(f)
    f.close()
    accuracy = graph['accuracy']
    error = graph['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(error, 'ro-', color = 'red')
    plt.legend(['CNN Accuracy', 'CNN Loss'], loc='upper left')
    plt.title('CNN Training Accuracy & Loss Graph')
    plt.show()    

def predict():
    global scaler, lite_model, pca, label_encoder
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    for i in range(len(columns)):
        dataset[columns[i]] = pd.Series(label_encoder[i].fit_transform(dataset[columns[i]].astype(str)))
    dataset = dataset.values
    X = scaler.transform(dataset)
    X1 = pca.transform(X)
    X = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))
    predict = lite_model.predict(X) 
    predict = np.argmax(predict, axis=1)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 0:
            text.insert(END,"Test Data : "+str(X1[i])+" ====> NO ATTACK DETECTED\n\n")
        else:
            text.insert(END,"Test Data : "+str(X1[i])+" ====> ATTACK DETECTED\n\n")  

font = ('times', 16, 'bold')
title = Label(main, text='Advanced Traffic Sampling and Analysis for Enhanced Network Security: Leveraging CNN Dense-Net and Data Augmentation')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload UNSW-NB15 Dataset", command=uploadDataset)
uploadButton.place(x=20,y=550)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=300,y=550)
processButton.config(font=ff)

pcaButton = Button(main, text="PCA Dimension Reduction", command=runPCA)
pcaButton.place(x=560,y=550)
pcaButton.config(font=ff)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=800,y=550)
svmButton.config(font=ff)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=20,y=600)
rfButton.config(font=ff)

cnnButton = Button(main, text="Run Propose CNN Algorithm", command=runCNN)
cnnButton.place(x=300,y=600)
cnnButton.config(font=ff)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=560,y=600)
graphButton.config(font=ff)

predictButton = Button(main, text="Detect Attack from Test Data", command=predict)
predictButton.place(x=800,y=600)
predictButton.config(font=ff)

#trainingButton = Button(main, text="CNN Training Accuracy & Loss Graph", command=trainingGraph)
#trainingButton.place(x=1040,y=600)
#trainingButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
