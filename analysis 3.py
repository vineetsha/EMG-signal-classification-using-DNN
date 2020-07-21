from subject import *
sub1=subject(2)
data=[]
for i in range(5):
    data.append(train_test_split(sub1.positions[i].position_data.iloc[:,:42],sub1.positions[i].position_data.iloc[:,42],test_size=0.3 ,random_state=4))
xtrain=normalize(np.concatenate((data[0][0].values,data[2][0].values,data[4][0].values),axis=0))
xtest=normalize(np.concatenate((data[0][1].values,data[1][1].values,data[2][1].values,data[3][1].values,data[4][1].values),axis=0))
ytrain=pd.get_dummies(np.concatenate((data[0][2].values,data[2][2].values,data[4][2].values),axis=0),prefix='target_')
ytest=pd.get_dummies(np.concatenate((data[0][3].values,data[1][3].values,data[2][3].values,data[3][3].values,data[4][3].values),axis=0),prefix='target_')

#analysis on Optimizing algorithm
opti=['adam','adagrad','adadelta','rmsprop','SGD','SGD_momentum']
optimizers=[
           tf.keras.optimizers.Adam(lr=0.005,beta_1=0.9,beta_2=0.999,epsilon=1e-07),
           tf.keras.optimizers.Adagrad(lr=0.01,epsilon=1e-07),
           tf.keras.optimizers.Adadelta(lr=0.001,rho=0.09,epsilon=1e-07),
           tf.keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=1e-07),
           tf.keras.optimizers.SGD(lr=0.05,momentum=0.0,nesterov=False),
           tf.keras.optimizers.SGD(lr=0.01,momentum=0.01,nesterov=False)
]
accuracy={}
model={}
history={}
for i in range(6):
    model[opti[i]]=sub1.positions[1].create_classifier(optimizers[i],dropout=0.5)
    history[opti[i]]=model[opti[i]].fit(x=xtrain,y=ytrain,batch_size=16,epochs=140,validation_split=0.1)
    
for i in model.keys():
    y_pred=model[i].predict(xtest).round()
    print(y_pred.shape)
    cm = sklearn.metrics.confusion_matrix(ytest.values.argmax(axis=1), y_pred.argmax(axis=1))
    accuracy[i]=sklearn.metrics.accuracy_score(ytest, y_pred)
    sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)
    plt.title('confusion matrix when %s optimizer is used \n accuracy=%f'%(i,accuracy[i]))
    plt.savefig('confusion matrix when %s optimizer is used \n accuracy=%f'%(i,accuracy[i]))
    plt.show()
    
plt.figure(figsize=(10,8))
for c1 in history.keys():
  plt.plot(history[c1].history['loss'])
plt.title(' training loss vs number of epochs for different optimizers')
plt.xlabel('num of epochs')
plt.ylabel('error')
plt.legend(history.keys())
plt.savefig(' training loss vs number of epochs for different optimizers')
plt.show()
    
#analysis on regularisation to avoid overfitting
without_reg=sub1.positions[0].create_classifier()
dropout=sub1.positions[1].create_classifier(dropout=0.2)
l2=sub1.positions[2].create_classifier(L2reg=0.001)

h1=without_reg.fit(x=xtrain,y=ytrain,batch_size=64,epochs=140,validation_split=0.1)
h2=dropout.fit(x=xtrain,y=ytrain,batch_size=64,epochs=140,validation_split=0.1)
h3=l2.fit(x=xtrain,y=ytrain,batch_size=64,epochs=140,validation_split=0.1)

plt.figure(figsize=(10,8))
for c1 in [h1,h2,h3]:
  plt.plot(c1.history['loss'])
plt.title(' training loss vs number of epochs for different regularizers')
plt.xlabel('num of epochs')
plt.ylabel('error')
plt.legend(['without regularisation','with dropout','with L2 regularisation'])
plt.savefig(' training loss vs number of epochs for different regularizers')
plt.show()

y_pred=without_reg.predict(xtest).round()
print("         Accuracy")
plt.figure(figsize=(12,10))
cm=sklearn.metrics.confusion_matrix(ytest.values.argmax(axis=1), y_pred.argmax(axis=1))
cm=cm / cm.astype(np.float).sum(axis=1)
sns.heatmap(cm,annot=True,cmap="Blues",fmt="0.7f",cbar=False)
plt.title('Normalized confusion matrix')
plt.show()
print('without regularisation',sklearn.metrics.accuracy_score(ytest,y_pred))

y_pred=dropout.predict(xtest).round()
print('DropOut',sklearn.metrics.accuracy_score(ytest,y_pred))

y_pred=l2.predict(xtest).round()
print('without L2 regularisation',sklearn.metrics.accuracy_score(ytest,y_pred))
