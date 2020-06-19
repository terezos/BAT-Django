from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

url="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
raw_data = pd.read_csv(url)
df = raw_data[((raw_data['days_b_screening_arrest'] <=30) & 
(raw_data['days_b_screening_arrest'] >= -30) &
(raw_data['is_recid'] != -1) &
(raw_data['c_charge_degree'] != 'O') & 
(raw_data['score_text'] != 'N/A'))]

df_encoded = pd.read_csv("https://raw.githubusercontent.com/adebayoj/fairml/master/doc/example_notebooks/propublica_data_for_fairml.csv")



class RaceDistribution(APIView):
    def get(self,request,*args,**kwargs):
        values = df['race'].value_counts()
        data = {
                'Black' : round((values[0]/len(df)) * 100,2) ,
                'White' : round((values[1]/len(df)) * 100,2),
                'Hispanic' : round((values[2]/len(df)) * 100,2),
                'Other' : round((values[3]/len(df)) * 100,2),
                'Asian' : round((values[4]/len(df) )* 100,2) ,
                'Native-Americans' : round(( values[5]/len(df) )* 100,2) ,
        }
        return Response(data)

class GenderDistribution(APIView):
    def get(self,request,*args,**kwargs):
        values = df['sex'].value_counts()
        goin = {
                'Male' : round((values[0]/len(df)) * 100,2) ,
                'Female' : round((values[1]/len(df)) * 100,2)
                
        }
        return Response(goin)

def calc_prop(data, group_col, group, output_col, output_val):
    new = data[data[group_col] == group]
    return len(new[new[output_col] == output_val])/len(new)

class MLoperations(APIView):
    def get(self,request,*args,**kwargs):
                data = df_encoded.drop(['Two_yr_Recidivism'],axis=1)
                y = df_encoded['Two_yr_Recidivism']
                goin = []
                train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.25, shuffle=True, random_state=42)
                ## LOGISTIC REGRESSION
                lr = LogisticRegression()
                model = lr.fit(train_x,train_y)
                acc = model.score(test_x,test_y)
                preds = model.predict(data)
                pred_df = pd.DataFrame({"race":df["race"],"Prediction":preds})
                pred_gender_df = pd.DataFrame({"sex":df["sex"],"Prediction":preds})
                lr_pr_unpriv = calc_prop(pred_df,"race",'African-American',"Prediction",1)
                lr_pr_priv = calc_prop(pred_df,"race",'Caucasian',"Prediction",1)
                DIRACECOMPAS = lr_pr_unpriv / lr_pr_priv
                lr_pr_priv_gend = calc_prop(pred_gender_df,"sex",'Female',"Prediction",1)
                lr_pr_unpriv_gend = calc_prop(pred_gender_df,"sex",'Male',"Prediction",1) 
                DIGENDERCOMPAS = lr_pr_unpriv_gend / lr_pr_priv_gend
                LR = {
                'model' : 'LR' ,    
                'acc' : round(acc,2) ,
                'DIlogisticRegRace' : round(DIRACECOMPAS,2) ,
                'DIlogisticRegGender' : round(DIGENDERCOMPAS,2)
                
                }
                goin.append(LR)

                ## RANDOM FOREST
                model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
                model.fit(train_x,train_y)
                acc = model.score(test_x,test_y)
                preds = model.predict(data)
                pred_df = pd.DataFrame({"race":df["race"],"Prediction":preds})
                pred_gender_df = pd.DataFrame({"sex":df["sex"],"Prediction":preds})
                lr_pr_unpriv = calc_prop(pred_df,"race",'African-American',"Prediction",1)
                lr_pr_priv = calc_prop(pred_df,"race",'Caucasian',"Prediction",1)
                DIrandomForestRace = lr_pr_unpriv / lr_pr_priv #DISPARATE IMPACT OF RANDOM FOREST RACE
                lr_pr_priv_gend = calc_prop(pred_gender_df,"sex",'Female',"Prediction",1)
                lr_pr_unpriv_gend = calc_prop(pred_gender_df,"sex",'Male',"Prediction",1) 
                DIrandomForestGender = lr_pr_unpriv_gend / lr_pr_priv_gend
                RF = {
                'model' : 'RF' ,    
                'acc' : round(acc,2) ,
                'DIrandomForestRace' : round(DIrandomForestRace,2) ,
                'DIrandomForestGender' : round(DIrandomForestGender,2)    
                }
                goin.append(RF)

                ## Desicion Tree
                tree = DecisionTreeClassifier()
                dt = tree.fit(train_x, train_y)
                acc = dt.score(test_x,test_y)
                preds = dt.predict(data)
                pred_df = pd.DataFrame({"race":df["race"],"Prediction":preds})
                pred_gender_df = pd.DataFrame({"sex":df["sex"],"Prediction":preds})
                lr_pr_unpriv = calc_prop(pred_df,"race",'African-American',"Prediction",1)
                lr_pr_priv = calc_prop(pred_df,"race",'Caucasian',"Prediction",1)
                DIdesicionTreeRace = lr_pr_unpriv / lr_pr_priv
                lr_pr_priv_gend = calc_prop(pred_gender_df,"sex",'Female',"Prediction",1)
                lr_pr_unpriv_gend = calc_prop(pred_gender_df,"sex",'Male',"Prediction",1) 
                DIrandomTreeGender = lr_pr_unpriv_gend / lr_pr_priv_gend

                DTree = {
                'model' : 'DT' ,    
                'acc' : round(acc,2) ,
                'DIdesicionTreeRace' : round(DIdesicionTreeRace,2) ,
                'DIrandomTreeGender' : round(DIrandomTreeGender,2)    
                }

                goin.append(DTree)

                # KNN
                neigh = KNeighborsClassifier(n_neighbors=3)
                neigh.fit(train_x, train_y)
                acc = neigh.score(test_x,test_y)
                preds = neigh.predict(data)
                pred_df = pd.DataFrame({"race":df["race"],"Prediction":preds})
                pred_gender_df = pd.DataFrame({"sex":df["sex"],"Prediction":preds})
                lr_pr_unpriv = calc_prop(pred_df,"race",'African-American',"Prediction",1)
                lr_pr_priv = calc_prop(pred_df,"race",'Caucasian',"Prediction",1)
                DIkNNRace = lr_pr_unpriv / lr_pr_priv
                lr_pr_priv_gend = calc_prop(pred_gender_df,"sex",'Female',"Prediction",1)
                lr_pr_unpriv_gend = calc_prop(pred_gender_df,"sex",'Male',"Prediction",1)
                DIkNNGender = lr_pr_unpriv_gend / lr_pr_priv_gend
                
                Knn = {
                'model' : 'KNN' ,    
                'acc' : round(acc,2) ,
                'DIkNNRace' : round(DIkNNRace,2) ,
                'DIkNNGender' : round(DIkNNGender,2)    
                }
                
                goin.append(Knn)

                ## Naive Bayes
                gnb = GaussianNB()
                gnb.fit(train_x, train_y)
                acc = gnb.score(test_x,test_y)
                preds = gnb.predict(data)
                pred_df = pd.DataFrame({"race":df["race"],"Prediction":preds})
                pred_gender_df = pd.DataFrame({"sex":df["sex"],"Prediction":preds})
                lr_pr_unpriv = calc_prop(pred_df,"race",'African-American',"Prediction",1)
                lr_pr_priv = calc_prop(pred_df,"race",'Caucasian',"Prediction",1)
                DInaiveBayesRace = lr_pr_unpriv / lr_pr_priv
                lr_pr_priv_gend = calc_prop(pred_gender_df,"sex",'Female',"Prediction",1)
                lr_pr_unpriv_gend = calc_prop(pred_gender_df,"sex",'Male',"Prediction",1)
                DInaiveBayesGender = lr_pr_unpriv_gend / lr_pr_priv_gend

                NB = {
                'model' : 'NB' ,    
                'acc' : round(acc,2) ,
                'DInaiveBayesRace' : round(DInaiveBayesRace,2) ,
                'DInaiveBayesGender' : round(DInaiveBayesGender,2)    
                }
                goin.append(NB)

                ## Adaboost
                clf = AdaBoostClassifier(n_estimators=100, random_state=0)
                clf.fit(train_x, train_y)
                acc = clf.score(test_x,test_y)
                preds = clf.predict(data)
                pred_df = pd.DataFrame({"race":df["race"],"Prediction":preds})
                pred_gender_df = pd.DataFrame({"sex":df["sex"],"Prediction":preds})            
                lr_pr_unpriv = calc_prop(pred_df,"race",'African-American',"Prediction",1)
                lr_pr_priv = calc_prop(pred_df,"race",'Caucasian',"Prediction",1)
                DIadaBoostRace = lr_pr_unpriv / lr_pr_priv
                lr_pr_priv_gend = calc_prop(pred_gender_df,"sex",'Female',"Prediction",1)
                lr_pr_unpriv_gend = calc_prop(pred_gender_df,"sex",'Male',"Prediction",1)
                DIadaBoostGender = lr_pr_unpriv_gend / lr_pr_priv_gend

                Adaboost = {
                'model' : 'ABoost' ,    
                'acc' : round(acc,2) ,
                'DIadaBoostRace' : round(DIadaBoostRace,2) ,
                'DIadaBoostGender' : round(DIadaBoostGender,2)    
                }

                goin.append(Adaboost)


                ## SVM
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                clf.fit(train_x, train_y)
                acc = clf.score(test_x,test_y)
                preds = clf.predict(data)
                pred_df = pd.DataFrame({"race":df["race"],"Prediction":preds})
                pred_gender_df = pd.DataFrame({"sex":df["sex"],"Prediction":preds})            
                lr_pr_unpriv = calc_prop(pred_df,"race",'African-American',"Prediction",1)
                lr_pr_priv = calc_prop(pred_df,"race",'Caucasian',"Prediction",1)
                DISVMRACE =lr_pr_unpriv / lr_pr_priv
                lr_pr_priv_gend = calc_prop(pred_gender_df,"sex",'Female',"Prediction",1)
                lr_pr_unpriv_gend = calc_prop(pred_gender_df,"sex",'Male',"Prediction",1)
                DISVMGender = lr_pr_unpriv_gend / lr_pr_priv_gend

                SVM = {
                'model' : 'SVM' ,    
                'acc' : round(acc,2) ,
                'DISVMRACE' : round(DISVMRACE,2) ,
                'DISVMGender' : round(DISVMGender,2)    
                }

                goin.append(SVM)


                return Response(goin)
