from modules import *
from getfTDD import *
def normalize(df):
  sc=StandardScaler()
  return sc.fit_transform(df)
class labels:
  def __init__(self):
    classes=['HandRest','HandOpen','ObjectGrip','PichGrip','WristExten','WristFlex','WristPron','WristSupi']
    self.label=dict([(classes[x],x) for x in range(8)])
  @abstractmethod
  def fun(self):
    pass 
class subject(labels):
  #instance variables
  # sex, directory, subject_num
  
  #inner class
  class position(labels):
    # instance variables
    #
    def __init__(self,pos_num=None,directory=None):
      super().__init__()
      self.position='Pos%x'%pos_num
      self.directory=directory
    
    def split_normalize(self,split_ratio):
      #train test split
      self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.position_data.iloc[:,:42],self.position_data.iloc[:,42],test_size=split_ratio,random_state=4)
      
      #preprocessing
      self.x_train=normalize(self.x_train)
      self.x_test=normalize(self.x_test)
      self.y_train=pd.get_dummies(self.y_train,prefix='target_')
    
    def import_data(self,split=False):
      self.position_data=pd.DataFrame()
      for lab in self.label.keys():
        for j in range(1,7):
          imported_data=pd.read_csv(self.directory + '\\' + self.position+'_%s_-%x.txt'%(lab,j),header=None,delim_whitespace=True)
          #print(imported_data.values.shape)
          ftdd_features=getfTDDfeat_v2(imported_data.values,1,400,100)[:-1,:]
          targets=np.full((ftdd_features.shape[0],1),self.label[lab])
          self.position_data= self.position_data.append(pd.DataFrame(np.concatenate((ftdd_features, targets), axis=1)), ignore_index=True)
      self.position_data.reindex(index=np.random.permutation(self.position_data.index))
      print(self.position,'successfully loaded')
      
      
    def create_classifier(self,optimizer='Adam',dropout=0,L2reg=0): 
      self.model=tf.keras.models.Sequential()
      self.model.add(tf.keras.layers.InputLayer(input_shape=(42,)))
      self.model.add(tf.keras.layers.Dense(units=40,kernel_initializer='uniform',bias_initializer='uniform',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2reg), bias_regularizer=tf.keras.regularizers.l2(L2reg)))
      self.model.add(tf.keras.layers.Dropout(rate=dropout))
      self.model.add(tf.keras.layers.Dense(units=32,kernel_initializer='uniform',bias_initializer='uniform',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2reg), bias_regularizer=tf.keras.regularizers.l2(L2reg)))
      self.model.add(tf.keras.layers.Dropout(rate=dropout/2))
      self.model.add(tf.keras.layers.Dense(units=24,kernel_initializer='uniform',bias_initializer='uniform',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(L2reg), bias_regularizer=tf.keras.regularizers.l2(L2reg)))
      self.model.add(tf.keras.layers.Dense(units=8,kernel_initializer='uniform',bias_initializer='uniform',activation='softmax'))
      self.model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=[tf.keras.metrics.categorical_accuracy])
      return self.model
  
    
    def train_classifier(self,x_train,y_train):
      self.train_history=self.model.fit(x=x_train,y=y_train,batch_size=16,epochs=140,validation_split=0.1)
      print('model trained successfully ')
    def make_predictions(self,x_test):
        return self.model.predict(x_test).round()
  
  def __init__(self,num):
    super().__init__()
    self.subject_num=num    #subject number 
    
    #sex of subject
    if(num==9):
      self.sex='Female'
    else:
      self.sex='Male'

    self.directory='D:\emg dataset\S%x_%s'%(num,self.sex)   #directory of the subject
    self.positions=[subject.position(x,self.directory) for x in np.arange(1,6)]
    for positions in self.positions:
      positions.import_data()