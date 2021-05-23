from django.http import JsonResponse
import os.path
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
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SITE_ROOT = os.path.dirname(os.path.realpath(__file__))
url="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
raw_data = pd.read_csv(url)
df = raw_data[((raw_data['days_b_screening_arrest'] <=30) & 
(raw_data['days_b_screening_arrest'] >= -30) &
(raw_data['is_recid'] != -1) &
(raw_data['c_charge_degree'] != 'O') & 
(raw_data['score_text'] != 'N/A'))]

df_encoded = pd.read_csv("https://raw.githubusercontent.com/adebayoj/fairml/master/doc/example_notebooks/propublica_data_for_fairml.csv")



class CompasRaceDistribution(APIView):
    def get(self,request,*args,**kwargs):
        values = df['race'].value_counts()
        data = {
                'African-American' : round((values[0]/len(df)) * 100,2) ,
                'Caucasian' : round((values[1]/len(df)) * 100,2),
                'Hispanic' : round((values[2]/len(df)) * 100,2),
                'Other' : round((values[3]/len(df)) * 100,2),
                'Asian' : round((values[4]/len(df) )* 100,2) ,
                'Native-Americans' : round(( values[5]/len(df) )* 100,2) ,
        }
        return Response(data)

class CompasGenderDistribution(APIView):
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

class CompasMLoperations(APIView):
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
                'model' : 'Logistic Regression' ,    
                'acc' : round(acc,2) ,
                'DIRace' : round(DIRACECOMPAS,2) ,
                'DIGender' : round(DIGENDERCOMPAS,2)
                
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
                'model' : 'Random Forest' ,    
                'acc' : round(acc,2) ,
                'DIRace' : round(DIrandomForestRace,2) ,
                'DIGender' : round(DIrandomForestGender,2)    
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
                'model' : 'Desicion Tree' ,    
                'acc' : round(acc,2) ,
                'DIRace' : round(DIdesicionTreeRace,2) ,
                'DIGender' : round(DIrandomTreeGender,2)    
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
                'DIRace' : round(DIkNNRace,2) ,
                'DIGender' : round(DIkNNGender,2)    
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
                'model' : 'Naive Bayes' ,    
                'acc' : round(acc,2) ,
                'DIRace' : round(DInaiveBayesRace,2) ,
                'DIGender' : round(DInaiveBayesGender,2)    
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
                'model' : 'AdaBoost' ,    
                'acc' : round(acc,2) ,
                'DIRace' : round(DIadaBoostRace,2) ,
                'DIGender' : round(DIadaBoostGender,2)    
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
                'DIRace' : round(DISVMRACE,2) ,
                'DIGender' : round(DISVMGender,2)    
                }

                goin.append(SVM)


                return Response(goin)

class GermanGenderDistribution(APIView):

    def get(self,request,*args,**kwargs):
        df_credit = pd.read_csv(SITE_ROOT + '/csvs/german_credit_data.csv')
        values = df_credit['Sex'].value_counts()
        gender_data = {
                'Male' : round((values[0]/len(df_credit)) * 100,2) ,
                'Female' : round((values[1]/len(df_credit)) * 100,2)
                
        }
        return Response(gender_data)
class GermanBadAndGoodDistribution(APIView):

    def get(self,request,*args,**kwargs):
        df_credit = pd.read_csv(SITE_ROOT + '/csvs/german_credit_data.csv')
        x = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().index.values

        badgood_data = {
             'Good' :  df_credit[df_credit["Risk"]== 'good']["Risk"].value_counts().values,
             'Bad' : df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().values
        }

        return Response(badgood_data)


class GermanMLoperations(APIView):

    def get(self,request,*args,**kwargs):
        goin = []
        df_credit = pd.read_csv(SITE_ROOT + '/csvs/german_credit_data.csv')
        df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
        df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

        #Purpose to Dummies Variable
        df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
        #Sex feature in dummies
        df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
        # Housing get dummies
        df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
        # Housing get Saving Accounts
        df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
        # Housing get Risk
        df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
        # Housing get Checking Account
        df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
        interval = (18,35,120)
        # [2,3,0,1]
        cats = ['Young', 'Old']
        df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)

          
        # label_encoder object knows how to understand word labels.
        label_encoder = preprocessing.LabelEncoder()
        
        # Encode labels in column 'species'.
        df_credit['Age_cat']= label_encoder.fit_transform(df_credit['Age_cat'])


        del df_credit["Saving accounts"]
        del df_credit["Checking account"]
        del df_credit["Purpose"]
        del df_credit["Sex"]
        del df_credit["Housing"]
        del df_credit["Risk"]
        del df_credit['Risk_good']
        del df_credit['Age']

        data = df_credit.drop(['Risk_bad'],axis=1)
        y = df_credit['Risk_bad']
        train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.25, shuffle=True, random_state=42)
        # LOGISTIC REGRESSION
        lr = LogisticRegression()
        model = lr.fit(train_x,train_y)
        acc = model.score(test_x,test_y)
        preds = model.predict(data)
        pred_df = pd.DataFrame({"Age_cat":df_credit["Age_cat"],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,"Age_cat",1,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,"Age_cat",0,"Prediction",1)
        DIRAGELR = lr_pr_unpriv / lr_pr_priv

        pred_gender_df = pd.DataFrame({"Sex_male":df_credit["Sex_male"],"Prediction":preds})
        lr_pr_priv_gend = calc_prop(pred_gender_df,"Sex_male",1,"Prediction",1)
        lr_pr_unpriv_gend = calc_prop(pred_gender_df,"Sex_male",0,"Prediction",1) 
        DIGENDERLR = lr_pr_unpriv_gend / lr_pr_priv_gend
        LR = {
        'model' : 'LR' ,    
        'acc' : round(acc,2) ,
        'DIAGE' : round(DIRAGELR,2) ,
        'DIGender' : round(DIGENDERLR,2)

        }

        goin.append(LR)


        ## RANDOM FOREST
        model = RandomForestClassifier(n_estimators=100, 
                    bootstrap = True,
                    max_features = 'sqrt')
        model.fit(train_x,train_y)
        acc = model.score(test_x,test_y)
        preds = model.predict(data)
        pred_df = pd.DataFrame({"Age_cat":df_credit["Age_cat"],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,"Age_cat",1,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,"Age_cat",0,"Prediction",1)
        DIrandomForestAge = lr_pr_unpriv / lr_pr_priv

        pred_gender_df = pd.DataFrame({"Sex_male":df_credit["Sex_male"],"Prediction":preds})
        lr_pr_priv_gend = calc_prop(pred_gender_df,"Sex_male",1,"Prediction",1)
        lr_pr_unpriv_gend = calc_prop(pred_gender_df,"Sex_male",0,"Prediction",1) 
        DIrandomForestGender = lr_pr_unpriv_gend / lr_pr_priv_gend
        RF = {
        'model' : 'RF' ,    
        'acc' : round(acc,2) ,
        'DIAGE' : round(DIrandomForestAge,2) ,
        'DIGender' : round(DIrandomForestGender,2)    
        }
        goin.append(RF)

        ## Desicion Tree
        tree = DecisionTreeClassifier()
        dt = tree.fit(train_x, train_y)
        acc = dt.score(test_x,test_y)
        preds = dt.predict(data)
        pred_df = pd.DataFrame({"Age_cat":df_credit["Age_cat"],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,"Age_cat",1,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,"Age_cat",0,"Prediction",1)
        DIdesicionTreeAge = lr_pr_unpriv / lr_pr_priv

        pred_gender_df = pd.DataFrame({"Sex_male":df_credit["Sex_male"],"Prediction":preds})
        lr_pr_priv_gend = calc_prop(pred_gender_df,"Sex_male",1,"Prediction",1)
        lr_pr_unpriv_gend = calc_prop(pred_gender_df,"Sex_male",0,"Prediction",1) 
        DIrandomTreeGender = lr_pr_unpriv_gend / lr_pr_priv_gend

        DTree = {
        'model' : 'DT' ,    
        'acc' : round(acc,2) ,
        'DIAGE' : round(DIdesicionTreeAge,2) ,
        'DIGender' : round(DIrandomTreeGender,2)    
        }

        goin.append(DTree)

        # KNN
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(train_x, train_y)
        acc = neigh.score(test_x,test_y)
        preds = neigh.predict(data)
        pred_df = pd.DataFrame({"Age_cat":df_credit["Age_cat"],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,"Age_cat",1,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,"Age_cat",0,"Prediction",1)
        DIkNNAge = lr_pr_unpriv / lr_pr_priv

        pred_gender_df = pd.DataFrame({"Sex_male":df_credit["Sex_male"],"Prediction":preds})
        lr_pr_priv_gend = calc_prop(pred_gender_df,"Sex_male",1,"Prediction",1)
        lr_pr_unpriv_gend = calc_prop(pred_gender_df,"Sex_male",0,"Prediction",1) 
        DIkNNGender = lr_pr_unpriv_gend / lr_pr_priv_gend

        
        Knn = {
        'model' : 'KNN' ,    
        'acc' : round(acc,2) ,
        'DIAGE' : round(DIkNNAge,2) ,
        'DIGender' : round(DIkNNGender,2)    
        }
        
        goin.append(Knn)

        ## Naive Bayes
        gnb = GaussianNB()
        gnb.fit(train_x, train_y)
        acc = gnb.score(test_x,test_y)
        preds = gnb.predict(data)
        pred_df = pd.DataFrame({"Age_cat":df_credit["Age_cat"],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,"Age_cat",1,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,"Age_cat",0,"Prediction",1)
        DInaiveBayesAge = lr_pr_unpriv / lr_pr_priv

        pred_gender_df = pd.DataFrame({"Sex_male":df_credit["Sex_male"],"Prediction":preds})
        lr_pr_priv_gend = calc_prop(pred_gender_df,"Sex_male",1,"Prediction",1)
        lr_pr_unpriv_gend = calc_prop(pred_gender_df,"Sex_male",0,"Prediction",1) 
        DInaiveBayesGender = lr_pr_unpriv_gend / lr_pr_priv_gend

        NB = {
        'model' : 'NB' ,    
        'acc' : round(acc,2) ,
        'DIAGE' : round(DInaiveBayesAge,2) ,
        'DIGender' : round(DInaiveBayesGender,2)    
        }
        goin.append(NB)

        ## Adaboost
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(train_x, train_y)
        acc = clf.score(test_x,test_y)
        preds = clf.predict(data)
        pred_df = pd.DataFrame({"Age_cat":df_credit["Age_cat"],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,"Age_cat",1,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,"Age_cat",0,"Prediction",1)
        DIadaBoostAge = lr_pr_unpriv / lr_pr_priv

        pred_gender_df = pd.DataFrame({"Sex_male":df_credit["Sex_male"],"Prediction":preds})
        lr_pr_priv_gend = calc_prop(pred_gender_df,"Sex_male",1,"Prediction",1)
        lr_pr_unpriv_gend = calc_prop(pred_gender_df,"Sex_male",0,"Prediction",1) 
        DIadaBoostGender = lr_pr_unpriv_gend / lr_pr_priv_gend

        Adaboost = {
        'model' : 'ABoost' ,    
        'acc' : round(acc,2) ,
        'DIAGE' : round(DIadaBoostAge,2) ,
        'DIGender' : round(DIadaBoostGender,2)    
        }

        goin.append(Adaboost)


        ## SVM
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(train_x, train_y)
        acc = clf.score(test_x,test_y)
        preds = clf.predict(data)
        pred_df = pd.DataFrame({"Age_cat":df_credit["Age_cat"],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,"Age_cat",1,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,"Age_cat",0,"Prediction",1)
        DISVMAge = lr_pr_unpriv / lr_pr_priv

        pred_gender_df = pd.DataFrame({"Sex_male":df_credit["Sex_male"],"Prediction":preds})
        lr_pr_priv_gend = calc_prop(pred_gender_df,"Sex_male",1,"Prediction",1)
        lr_pr_unpriv_gend = calc_prop(pred_gender_df,"Sex_male",0,"Prediction",1) 
        DISVMGender = lr_pr_unpriv_gend / lr_pr_priv_gend

        SVM = {
        'model' : 'SVM' ,    
        'acc' : round(acc,2) ,
        'DIAGE' : round(DISVMAge,2) ,
        'DIGender' : round(DISVMGender,2)    
        }

        goin.append(SVM)



        return Response(goin)
