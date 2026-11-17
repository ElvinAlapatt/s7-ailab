from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score

dataset = load_iris()

X = dataset.data 
Y = dataset.target

X_train , Xtest , Ytrain , Ytest = train_test_split(X,Y,test_size=0.3)

for k in [3,5,7,9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train,Ytrain)
    y_pred = model.predict(Xtest)

    print(k)
    print("Confusion Matrix : \n", confusion_matrix(Ytest,y_pred))
    print("\nClassification Report :  \n  ", classification_report(Ytest,y_pred))
    print("\nAccuracy : ", accuracy_score(Ytest,y_pred))