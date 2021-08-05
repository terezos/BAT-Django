from django.http import JsonResponse
import os.path
import json
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
from django.http import HttpResponse
from sklearn.preprocessing import StandardScaler

SITE_ROOT = os.path.dirname(os.path.realpath(__file__))
url="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
raw_data = pd.read_csv(url)
df = raw_data[((raw_data['days_b_screening_arrest'] <=30) & (raw_data['days_b_screening_arrest'] >= -30) &
(raw_data['is_recid'] != -1) &
(raw_data['c_charge_degree'] != 'O') & 
(raw_data['score_text'] != 'N/A'))]

df_encoded = pd.read_csv("https://raw.githubusercontent.com/adebayoj/fairml/master/doc/example_notebooks/propublica_data_for_fairml.csv")


def article_detail(request):
    return Response('1')

class CompasRaceDistribution(APIView):
    def get(self,request,*args,**kwargs):
        values = df['race'].value_counts()
        data = {}
        data['analysis'] = {
                'African-American' : round((values[0]/len(df)) * 100,2) ,
                'Caucasian' : round((values[1]/len(df)) * 100,2),
                'Hispanic' : round((values[2]/len(df)) * 100,2),
                'Other' : round((values[3]/len(df)) * 100,2),
                'Asian' : round((values[4]/len(df) )* 100,2) ,
                'Native-Americans' : round(( values[5]/len(df) )* 100,2) ,   
        }
        
        data['recid'] = {
                'African-American' : len(df[(df['race'] == 'African-American') & (df['two_year_recid'] == 1)]) ,
                'Caucasian' : len(df[(df['race'] == 'Caucasian') & (df['two_year_recid'] == 1)]),
                'Hispanic' :len(df[(df['race'] == 'Hispanic') & (df['two_year_recid'] == 1)]),
                'Other' : len(df[(df['race'] == 'Other') & (df['two_year_recid'] == 1)]),
                'Asian' :len(df[(df['race'] == 'Asian') & (df['two_year_recid'] == 1)]) ,
                'Native-Americans' : len(df[(df['race'] == 'Native-Americans') & (df['two_year_recid'] == 1)]) ,
            }
       
        return Response(data)

class CompasGenderDistribution(APIView):
    def get(self,request,*args,**kwargs):
        values = df['sex'].value_counts()
        data = {}
        data['analysis'] = {
                'Male' : round((values[0]/len(df)) * 100,2) ,
                'Female' : round((values[1]/len(df)) * 100,2)
        }


        males = len(df[(df['sex'] == 'Male') & (df['two_year_recid'] == 1)]) 
        female = len(df[(df['sex'] == 'Female') & (df['two_year_recid'] == 1)])
        males = round((males/values[0])*100,2)
        female = round((female/values[1])*100,2)

        data['recid'] = {
                'Male' : males ,
                'Female': female,
        }

        return Response(data)
        
def calc_prop(data, group_col, group, output_col, output_val):
    new = data[data[group_col] == group]
    return len(new[new[output_col] == output_val])/len(new)

class CompasMLoperations(APIView):
    def get(self,request,*args,**kwargs):
                goin = []
                data = df_encoded.drop(['Two_yr_Recidivism'],axis=1)
                y = df_encoded['Two_yr_Recidivism']
                train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.25, shuffle=True, random_state=42)
                ## LOGISTIC REGRESSION
                lr = LogisticRegression()
                model = lr.fit(train_x,train_y)
                acc = model.score(test_x,test_y)
                preds = model.predict(data)
                pred_df = pd.DataFrame({"race":df["race"],"Prediction":preds})
                lr_pr_unpriv = calc_prop(pred_df,"race",'African-American',"Prediction",1)
                lr_pr_priv = calc_prop(pred_df,"race",'Caucasian',"Prediction",1)
                DIRACECOMPAS = lr_pr_unpriv / lr_pr_priv
                pred_gender_df = pd.DataFrame({"sex":df["sex"],"Prediction":preds})
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
        data = {}
        data['analysis'] = {
                'Male' : round((values[0]/len(df_credit)) * 100,2) ,
                'Female' : round((values[1]/len(df_credit)) * 100,2)    
        }
        data['risk-gender'] = {
            'Male' : len(df_credit[(df_credit['Sex'] == 'male') & (df_credit['Risk'] == 'good')]),
            'Female': len(df_credit[(df_credit['Sex'] == 'female') & (df_credit['Risk'] == 'good')]),
        }
        return Response(data)
class GermanAgeDistribution(APIView):

    def get(self,request,*args,**kwargs):
        df_credit = pd.read_csv(SITE_ROOT + '/csvs/german_credit_data.csv')
        x = df_credit[df_credit["Risk"]== 'bad']["Risk"].value_counts().index.values
        data = {}
        data['analysis'] = {
             '18-25' :  len(df_credit[(df_credit['Age'] >= 18) & (df_credit['Age'] <= 25)]),
             '26-35' :  len(df_credit[(df_credit['Age'] > 26) & (df_credit['Age'] <= 35)]) ,
             '36-120' : len(df_credit[(df_credit['Age'] > 35)])
        }
        
        data['risk-age'] = {
            '18-25': len(df_credit[(df_credit['Age'] >= 18) & (df_credit['Age'] <= 25) & (df_credit['Risk'] == 'good')]),
            '26-35': len(df_credit[(df_credit['Age'] > 25) & (df_credit['Age'] <= 35) & (df_credit['Risk'] == 'good')]),
            '36-120': len(df_credit[(df_credit['Age'] > 35) & (df_credit['Risk'] == 'good')])
        }

        return Response(data)

class CustomDatasetMloperation(APIView):

    def get(self,request,filename,sensitive,analysis,target,privileged,unprivileged,dropFirstColumn,encode):
        
        url="https://bias-auditing-tool.herokuapp.com/files/" + filename + '.csv' 
        df_credit = pd.read_csv(url)
        df_credit_not_encoded = df_credit
        if(dropFirstColumn):
            df_credit = df_credit.drop(df_credit.columns[0], axis=1)

        willReturn = {}
        willReturn['total_count'] = df_credit[df_credit.columns[0]].count()
        values = df_credit[analysis].value_counts()
        
        keys = values.keys()
        data = {}
        for i in keys:
            data[i] = round((values[i]/len(df_credit)) * 100,2)       
        
       
        
        willReturn['analysis'] = data
        
        data2 = {}

        answerUn = unprivileged.isnumeric()
        answerPr = privileged.isnumeric()

        if(answerUn):
            unprivileged = int(unprivileged)

        if(answerPr):
            privileged = int(privileged)

            
        data2['privileged'] =  len(df_credit_not_encoded[(df_credit_not_encoded[sensitive] == privileged) & (df_credit_not_encoded[target] == 1)])
        data2['unprivileged'] =  len(df_credit_not_encoded[(df_credit_not_encoded[sensitive] == unprivileged) & (df_credit_not_encoded[target] == 1)])
        willReturn['check-priv'] = data2
        
        target_col = df_credit.pop(target)
        df_credit.insert(len(df_credit.columns), target, target_col)
        

        for column in df_credit:
            df_credit[column] = df_credit[column].fillna('no_inf')
        
        if(encode):
            old_columns = list(df_credit)
            for column in df_credit:
                df_credit = df_credit.merge(pd.get_dummies(df_credit[column], drop_first=True, prefix=column+'_new'), left_index=True, right_index=True)
        
            for cols in old_columns:
                del df_credit[cols] 
        
        data = df_credit.drop(df_credit.columns[len(df_credit.columns) -1 ],axis=1)
        y = df_credit[df_credit.columns[len(df_credit.columns)-1 ]]
        train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.25, shuffle=True, random_state=42)
        
        
        # LOGISTIC REGRESSION
        lr = LogisticRegression()
        model = lr.fit(train_x,train_y)
        acc = model.score(test_x,test_y)
        preds = model.predict(data)
        
        
        
        pred_df = pd.DataFrame({sensitive:df_credit_not_encoded[sensitive],"Prediction":preds})
        
        lr_pr_unpriv = calc_prop(pred_df,sensitive,unprivileged,"Prediction",1)
        
        lr_pr_priv = calc_prop(pred_df,sensitive,privileged,"Prediction",1)
        DILR = lr_pr_unpriv / lr_pr_priv
        
        willReturn['LR'] = {
        'model' : 'LR' ,    
        'acc' : round(acc,2) ,
        'disparate_impact' : round(DILR,2) ,
        }
        
        model = RandomForestClassifier(n_estimators=100, 
                    bootstrap = True,
                    max_features = 'sqrt')
        model.fit(train_x,train_y)
        acc = model.score(test_x,test_y)
        preds = model.predict(data)
        pred_df = pd.DataFrame({sensitive:df_credit_not_encoded[sensitive],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,sensitive,unprivileged,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,sensitive,privileged,"Prediction",1)
        DIRANDOMFOREST = lr_pr_unpriv / lr_pr_priv


        willReturn['RF'] = {
        'model' : 'RF' ,    
        'acc' : round(acc,2) ,
        'disparate_impact' : round(DIRANDOMFOREST,2) ,   
        }
        

        ## Desicion Tree
        tree = DecisionTreeClassifier()
        dt = tree.fit(train_x, train_y)
        acc = dt.score(test_x,test_y)
        preds = dt.predict(data)
        pred_df = pd.DataFrame({sensitive:df_credit_not_encoded[sensitive],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,sensitive,unprivileged,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,sensitive,privileged,"Prediction",1)
        DITree = lr_pr_unpriv / lr_pr_priv


        willReturn['DTree'] = {
        'model' : 'DT' ,    
        'acc' : round(acc,2) ,
        'disparate_impact' : round(DITree,2)    
        }


        # KNN
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(train_x, train_y)
        acc = neigh.score(test_x,test_y)
        preds = neigh.predict(data)
        pred_df = pd.DataFrame({sensitive:df_credit_not_encoded[sensitive],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,sensitive,unprivileged,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,sensitive,privileged,"Prediction",1)
        DIkNN = lr_pr_unpriv / lr_pr_priv

        
        willReturn['KNN']  = {
        'model' : 'KNN' ,    
        'acc' : round(acc,2) ,
        'disparate_impact' : round(DIkNN,2),  
        }
      
        ## Naive Bayes
        # gnb = GaussianNB()
        # gnb.fit(train_x, train_y)
        # acc = gnb.score(test_x,test_y)
        # preds = gnb.predict(data)
        # pred_df = pd.DataFrame({sensitive:df_credit_not_encoded[sensitive],"Prediction":preds})
        # lr_pr_unpriv = calc_prop(pred_df,sensitive,unprivileged,"Prediction",1)
        # lr_pr_priv = calc_prop(pred_df,sensitive,privileged,"Prediction",1)
        # DInaiveBayes = lr_pr_unpriv / lr_pr_priv

        # willReturn['NB'] = {
        # 'model' : 'NB' ,    
        # 'acc' : round(acc,2) ,
        # 'disparate_impact' : round(DInaiveBayes,2) ,  
        # }
    

        ## Adaboost
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(train_x, train_y)
        acc = clf.score(test_x,test_y)
        preds = clf.predict(data)
        pred_df = pd.DataFrame({sensitive:df_credit_not_encoded[sensitive],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,sensitive,unprivileged,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,sensitive,privileged,"Prediction",1)
        DIadaBoost = lr_pr_unpriv / lr_pr_priv

        willReturn['ABoost'] = {
        'model' : 'ABoost' ,    
        'acc' : round(acc,2) ,
        'disparate_impact' : round(DIadaBoost,2), 
        }


        ## SVM
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(train_x, train_y)
        acc = clf.score(test_x,test_y)
        preds = clf.predict(data)
        pred_df = pd.DataFrame({sensitive:df_credit_not_encoded[sensitive],"Prediction":preds})
        lr_pr_unpriv = calc_prop(pred_df,sensitive,unprivileged,"Prediction",1)
        lr_pr_priv = calc_prop(pred_df,sensitive,privileged,"Prediction",1)
        DISVM = lr_pr_unpriv / lr_pr_priv

        willReturn['SVM'] = {
        'model' : 'SVM' ,    
        'acc' : round(acc,2) ,
        'disparate_impact' : round(DISVM,2),  
        }

        return Response(willReturn)
        # return HttpResponse(url)
        # return Response ()

class GermanMLoperations(APIView):

    def get(self,request,*args,**kwargs):
        goin = []
        df_credit = pd.read_csv(SITE_ROOT + '/csvs/german_credit_data.csv')
        df_credit.drop(df_credit.columns[0], axis=1) 
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
