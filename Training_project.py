import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.svm  import SVC

class Project :
    def __init__(self):
        self.data = pd.read_csv('heart_2020_cleaned.csv')
        self.x = self.data.iloc[:,1:].values
        self.y=self.data.iloc[:,0].values

class EDA(Project):

    def __init__(self):
        super().__init__()
        
    def Exploration(self):
        print('The Dataset : \n',self.data)
        print('Information of the data : \n', self.data.info())
        print('first 5 rows : \n',self.data.head())
        print('last 5 rows : \n',self.data.tail())
        print('The name of columns : \n',self.data.head(0))
        print('The name of columns : \n',self.data.columns)
        print('Element without repating in each column : \n',self.data.nunique())
        print('Data type of  each column : \n',self.data.dtypes)
        print('Dimentions of the data : \n',self.data.shape)
        print('Number of empty cells of each column : \n',self.data.isnull().sum())
        print('Number of duplicated row in data : \n',self.data.duplicated().sum())
        print('statstics information of data : \n',self.data.describe())
    
    def Cleaning(self):
        print('Dimentions of the data before cleaning : \n',self.data.shape)
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(inplace=True)
        print('Dimentions of the data after cleaning : \n',self.data.shape)
    
    def scaled_encod(self):
        #encoding
        d_types=self.data.dtypes
        for i in range(self.data.shape[1]):
            if d_types[i]=='object':
                Pr_data = preprocessing.LabelEncoder()
                self.data[self.data.columns[i]] = Pr_data.fit_transform(self.data[self.data.columns[i]])
                
        #scaling
        new = self.data.select_dtypes(exclude=["object"])
        scaler = preprocessing.MinMaxScaler()
        scaled = scaler.fit_transform(new)
        self.scal = pd.DataFrame(scaled)
        self.x = self.scal.iloc[:,1:].values
        self.y=self.data.iloc[:,0].values
        return self.scal,self.y

    def visualiz(self):
        co = self.scal.corr()
        #heatmap of data
        sns.heatmap(co ,annot=True)
        plt.show()
        sns.catplot(data=self.data,kind="count",x="Stroke" ,hue="HeartDisease")
        plt.show()
        sns.catplot(data=self.data,kind="count",x="PhysicalHealth" ,hue="HeartDisease")
        plt.show()
        sns.catplot(data=self.data,kind="count",x="MentalHealth" ,hue="HeartDisease")
        plt.show()
    
    def visualizModel(self,ma):
        sns.heatmap(ma,annot=True)
        plt.show()
        
    def visualizModel2(self, la):
        sns.histplot(self.y, kde=True)
        plt.axvline(la, color='red', linestyle='--', label='Recall Score')
        plt.legend()
        plt.show()
    
    def visualizModel3(self, k):
        sns.histplot(self.y, kde=True)
        plt.axvline(k, color='green', linestyle='--', label='f1_Score')
        plt.legend()
        plt.show()


class Modeling(EDA):
    def __init__(self,scal,y):
        super().__init__()
        self.x = scal.iloc[:,1:].values
        self.y = y
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.x,self.y,test_size=0.3)
        print('X_train =\n',self.X_train)
        print('X_test =\n',self.X_test)
        print('Y_train =\n',self.Y_train)
        print('Y_test =\n',self.Y_test)

    def LR(self):
        #learning
        r=LogisticRegression()
        r.fit(self.X_train, self.Y_train)
        Y_pred=r.predict(self.X_test)
        #confusion  
        self.conf=confusion_matrix(self.Y_test, Y_pred)
        print('confusion_matrix(LR) =\n',self.conf)
        self.visualizModel(self.conf)
        #recall_score
        self.r2=recall_score(self.Y_test, Y_pred,average='micro')
        print('recall_score(LR) ',self.r2) 
        self.visualizModel2(self.r2)
        #f1_score
        self.f1=f1_score(self.Y_test,Y_pred,average='micro')
        print('f1_score(LR) =',self.f1)     
        self.visualizModel3(self.f1)
    def SVM(self):
        #learning
        clf=SVC(kernel='linear',degree=4)
        clf.fit(self.X_train,self.Y_train)
        Y_pred2=clf.predict(self.X_test)
        #confusion  
        self.conf2=confusion_matrix(self.Y_test, Y_pred2)
        print('confusion_matrix(SVM) =\n',self.conf2)
        self.visualizModel(self.conf2)
        #recall_score
        self.r22=recall_score(self.Y_test, Y_pred2,average='micro')
        print('recall_score(SVM) =',self.r22) 
        self.visualizModel2(self.r22)
        #f1_score
        self.f12=f1_score(self.Y_test,Y_pred2,average='micro')
        print('f1_score(SVM) =',self.f12)
        self.visualizModel3(self.f12)



m = EDA()
m.Exploration()
m.Cleaning()
scal,y = m.scaled_encod()
m.visualiz()
a = Modeling(scal,y)
a.LR() 
a.SVM()



