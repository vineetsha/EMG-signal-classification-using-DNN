from modules import *
from subject import *
subs=[subject(2),subject(3),subject(4),subject(5),subject(6)]
table=[]
for sub in subs:
    for i in range(5):
        sub.positions[i].split_normalize(0.5)
        sub.positions[i].create_classifier(dropout=0.5)
        sub.positions[i].train_classifier(sub.positions[i].x_train,sub.positions[i].y_train)
    a=[]
    for i in range(5):
        acc=[]
        for j in range(5):
            if i==j:
                x_test=sub.positions[j].x_test
                y_pred=sub.positions[i].make_predictions(x_test)
                acc.append(sklearn.metrics.accuracy_score(sub.positions[j].y_test, y_pred))
            else:
                x_test=normalize(sub.positions[j].position_data.iloc[:,:42])
                y_pred=sub.positions[i].make_predictions(x_test)
                acc.append(sklearn.metrics.accuracy_score(pd.get_dummies(sub.positions[j].position_data.iloc[:,42].values,prefix='target_'),y_pred))
        a.append(acc)
    table.append(a)
print(np.mean(table,axis=0))

            
    