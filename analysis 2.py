from subject import *
subs=[subject(2),subject(3),subject(4),subject(5),subject(6)]
table=[]
for sub in subs:
    for i in range(5):
        sub.positions[i].split_normalize(0.5)
    accuracy=[]
    for i in range(5):
        Xtrain=pd.DataFrame()
        Ytrain=pd.DataFrame()
        Xtest=[]
        Ytest=[]
        for j in range(5):
            if(i!=j):
                Xtrain=Xtrain.append(pd.DataFrame(sub.positions[j].x_train),ignore_index=True)
                Ytrain=Ytrain.append(pd.DataFrame(sub.positions[j].y_train),ignore_index=True)
            else:
                Xtest=sub.positions[i].x_test
                Ytest=pd.get_dummies(sub.positions[i].y_test,prefix='target_')
        classifier=sub.positions[i].create_classifier(optimizer=tf.keras.optimizers.Adam(lr=0.005),dropout=0.5)
        classifier.fit(Xtrain,Ytrain,batch_size=256,epochs=150,validation_split=0.1)
        y_pred=classifier.predict(Xtest).round()
        accuracy.append(sklearn.metrics.accuracy_score(Ytest, y_pred))
    table.append(accuracy)
print(table)
print(np.mean(table,axis=0))
        