"""================================
Classifying digits with adversary 
==================================="""
#%%
import matplotlib.pyplot as plt, numpy as np
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split#%% relevant parameteres
rw100=30; #print("\nrandomization weight = ",rw100/100) -- try 0
clfD = RandomForestClassifier(max_depth=8, n_estimators=10, max_features=3)
clfA = RandomForestClassifier(max_depth=8, n_estimators=10, max_features=3)
clfD_name='clfD: '+str(type(clfD)).split(".")[-1][:-2]
clfA_name='clfA: '+str(type(clfA)).split(".")[-1][:-2]
clfR_name='clfR: '+str(rw100)+"%"+' random + clfA'
#%%
np.random.seed(None) #new randoms at every run
#np.random.seed(0)   #reproducible results

#dataset preparation & modification ##########################################
digits = datasets.load_digits()
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

anomaly=8 #consider this as positive (anomaly) & other digits as negative
#count how many anomalies in digits.target 
anomalies=0
for i in np.arange(digits.target.size):
    if digits.target[i]==anomaly:
        digits.target[i]=1
    else:
        digits.target[i]=0
#so now we just have two classes: "1"=anomalous and "0"=normal
#%% prepare training and test sets, classifiers (same as for binary above)
# Split data into train and test subsets, fit clfD and clfA ##################
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False)
clfD.fit(X_train, y_train); clfA.fit(X_train, y_train)
testp = y_test.sum(); testn = len(y_test)-testp # number of pos & neg in test
#define predD, predA, predR (defender, adversary, randomized)
predD = clfD.predict_proba(X_test)[:, 1] #defender's predictions
predA = clfA.predict_proba(X_test)[:, 1] #adversary's predictions
rnums = np.random.randint(100, size=len(predA))/100 
predR = predA*(1-rw100/100)+rnums*rw100/100 #randomized predA
#%% define roc_curve and adversarial roc curve functions #####################
def thrs(scores):
    thresholds = np.unique(scores)
    thresholds = thresholds[::-1] #invert order, then add first element
    thresholds = np.concatenate(([thresholds[0]*2],thresholds))
    return(thresholds)
def myroc_curve(y,scores,thresholds): #similar to metrics.roc_curve
    fp=np.arange(len(thresholds)); fn=np.arange(len(thresholds))
    for j in np.arange(len(thresholds)):
        fp[j]=0; fn[j]=0
        for i in np.arange(len(y)):
            if y[i]==0 and scores[i]>=thresholds[j]: fp[j] += 1
            else: 
                if y[i]==1 and scores[i]<thresholds[j]: fn[j] += 1
    return(fp,fn) 
def aroc_curve(y,scores,ascores,thresholds): #adversarial roc curve
    fp=np.arange(len(thresholds)) #false defender positives
    afn=np.arange(len(thresholds)) #adversarial false negatives
    kfn=np.arange(len(thresholds)) #known false negative
    for j in np.arange(len(thresholds)):
        fp[j]=0; afn[j]=0 #false defender&adversarial negatives;
        kfn[j]=0 #known false negatives
        for i in np.arange(len(y)):
            if y[i]==0 and scores[i]>=thresholds[j]: fp[j] += 1
            else:  
                if y[i]==1 and ascores[i]<thresholds[j]: 
                    afn[j]+=1; 
                    if scores[i]<thresholds[j]: kfn[j] += 1 
    return(fp,kfn,afn) 
#%% compute utpr and afr for predD/predR wrt predA and plot curves ###########
def plot_tpr(pred,clf_name): 
    t = thrs(pred); fp,fn=myroc_curve(y_test,pred,t)   
    fpr=fp/testn; tpr=(testp-fn)/testp
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr,tpr,label=clf_name+' (AUC='+str(round(roc_auc,2))+')')
    plt.legend()
    plt.title('ROC curves: True Positive Rate (tpr) for clfD,clfA,clfR')
fig=plt.figure() #plot roc curves 
plot_tpr(predD,clfD_name); plot_tpr(predR,clfR_name); plot_tpr(predA,clfA_name)
def plot_utpr(fp,kfn,afn,clf_name): 
    fpr=fp/testn; kfnr=kfn/testp; utpr=1-kfnr
    aroc_auc = round(metrics.auc(fpr, utpr),2)
    plt.plot(fpr,utpr,label=clf_name+' (AUC='+str(aroc_auc)+')')
    plt.legend(); 
    plt.title('Unknown True Positive Rate (utpr) for clfD,clfR')
    return(utpr)
def plot_afr(fp,kfn,afn,clf_name):
    fpr=fp/testn; 
    with np.errstate(divide='ignore', invalid='ignore'):
       asr = np.nan_to_num(kfn/afn) 
    afr=1-asr 
    aroc_auc = round(metrics.auc(fpr, afr),2)
    plt.plot(fpr,afr,label=clf_name+' (AUC='+str(aroc_auc)+')')
    plt.legend(); 
    plt.title('Adversarial Failure Rate (afr) for clfD,clfR')
    return(afr)
fpD,kfnD,afnD = aroc_curve(y_test,predD,predA,thrs(predD))  
fpR,kfnR,afnR = aroc_curve(y_test,predR,predA,thrs(predR))  
fig=plt.figure() #plot adversarial roc curves
utpr1=plot_utpr(fpD,kfnD,afnD,clfD_name)
utpr2=plot_utpr(fpR,kfnR,afnR,clfR_name)
fig=plt.figure() #plot adversarial failure curves
afr3=plot_afr(fpD,kfnD,afnD,clfD_name)
afr4=plot_afr(fpR,kfnR,afnR,clfR_name)