import numpy as np
from sklearn import metrics as skmetrics
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.metrics import f1_score

def F1(num_classes, y_pred, y_true, average):
    return f1_score(y_true, y_pred, average=average, labels=range(num_classes))
def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'macro_f1', float(F1(preds.shape[0], Y_pre, Y, 'macro')), True

def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'precision', float(Counter(Y==Y_pre)[True]/len(Y)), True

import scipy
# from scipy.stats import spearmanr,pearsonr
class Metrictor:
    def __init__(self, batchList=False, contrastLearn=False, affinityFloor=None):
        self._reporter_ = {"MSE":self.MSE, "MAE":self.MAE, "Pearson":self.Pearson, "ACC":self.ACC, 
                           "CrossEntropyLOSS":self.CrossEntropyLOSS,
                           "ScipyPearsonr":self.ScipyPearsonr, "ScipySpearmanr":self.ScipySpearmanr,
                           "RANK_SCORE":self.RANK_SCORE, 
                           "AF_AUROC":self.AF_AUROC, "AF_AUPRC":self.AF_AUPRC,
                           "ProtPearson":self.ProtPearson, "ProtPearsonForPos":self.ProtPearsonForPos,
                           "wProtPearson":self.wProtPearson, "wProtPearsonForPos":self.wProtPearsonForPos,
                           "CI":self.CI,
                           "CalinskiHarabaszScore":self.CalinskiHarabaszScore}
        self.batchList = batchList
        self.contrastLearn = contrastLearn
        self.affinityFloor = affinityFloor
    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res
    
    def set_data(self, res):
        self.Y = res['y_true']
        self.Y_pre = res['y_prob']
        if self.batchList:
            self.N = len(self.Y[0])*(len(self.Y)-1) + len(self.Y[-1])
            self.maskPAD = res['maskPAD']

            if len(self.Y_pre[0].shape) == len(self.Y[0].shape)+1:
                self.Y_prob = self.Y_pre
                self.Y_pre = [i.argmax(-1) for i in self.Y_pre]
        elif len(self.Y_pre.shape) == len(self.Y.shape)+1:
            self.Y_prob = self.Y_pre
            self.Y_pre = self.Y_pre.argmax(-1)

    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i,res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))
    def each_class_indictor_show(self, id2lab):
        print('Waiting for finishing...')
    def ScipyPearsonr(self):
        return scipy.stats.pearsonr(self.Y_pre, self.Y)[0]
    def ScipySpearmanr(self):
        return scipy.stats.spearmanr(self.Y_pre, self.Y)[0]
    
    def ACC(self):
        if self.affinityFloor is None:
            if self.batchList:
                return np.sum([ACC(yp, yt, pad)*np.sum(pad) for yp,yt,pad in zip(self.Y_pre,self.Y,self.maskPAD)]) / np.sum([pad.sum() for pad in self.maskPAD])
            else:
                return ACC(self.Y_pre, self.Y, self.maskPAD)
        assert self.affinityFloor is not None
        if self.batchList:
            return np.sum([ACC(yp>0.5, yt>self.affinityFloor, pad)*np.sum(pad) for yp,yt,pad in zip(self.Y_pre,self.Y,self.maskPAD)]) / np.sum([pad.sum() for pad in self.maskPAD])
        else:
            return ACC(self.Y_pre>0.5, self.Y>self.affinityFloor, self.maskPAD)
    # def AUC(self):
    #     return AUC(self.Y_prob_pre,self.Y)
    def CrossEntropyLOSS(self):
        return CrossEntropyLOSS(self.Y_prob_pre,self.Y)
    def MSE(self):
        if self.batchList:
            return np.sum([MSE(yp,yt)*len(yp) for yp,yt in zip(self.Y_pre, self.Y)]) / self.N
        else:
            return MSE(self.Y_pre, self.Y)
    def MAE(self):
        if self.batchList:
            return np.sum([MAE(yp,yt)*len(yp) for yp,yt in zip(self.Y_pre, self.Y)]) / self.N
        else:
            return MAE(self.Y_pre, self.Y)
    def RANK_SCORE(self):
        if self.batchList:
            return np.sum([RANK_SCORE(yp,yt,pad)*len(yp) for yp,yt,pad in zip(self.Y_pre, self.Y, self.maskPAD)]) / self.N
        else:
            return RANK_SCORE(self.Y_pre, self.Y, self.maskPAD)
    def AF_AUROC(self):
        yt = self.Y[self.Y!=self.Y[0]]
        yp = self.Y_pre[self.Y!=self.Y[0]]
        
        yt = np.array(yt>self.Y[0], dtype=np.int32)
        yp = (yp-yp.min())/(yp.max()-yp.min())
        return skmetrics.roc_auc_score(yt, yp)
    def AF_AUPRC(self):
        yt = self.Y[self.Y!=self.Y[0]]
        yp = self.Y_pre[self.Y!=self.Y[0]]
        
        yt = np.array(yt>self.Y[0], dtype=np.int32)
        yp = (yp-yp.min())/(yp.max()-yp.min())
        return skmetrics.average_precision_score(yt, yp)
    def Pearson(self):
        return Pearson(self.Y_pre, self.Y)
    def ProtPearson(self):
        if self.batchList:
            return np.sum([Pearson(yp,yt,pad)*len(yp) for yp,yt,pad in zip(self.Y_pre, self.Y, self.maskPAD)]) / self.N
        else:
            return Pearson(self.Y_pre, self.Y, self.maskPAD)
    def ProtPearsonForPos(self):
        if self.batchList:
            return np.sum([Pearson(yp,yt,pad,self.affinityFloor)*len(yp) for yp,yt,pad in zip(self.Y_pre, self.Y, self.maskPAD)]) / self.N
        else:
            return Pearson(self.Y_pre, self.Y, self.maskPAD, self.affinityFloor)
    def wProtPearson(self):
        if self.batchList:
            tmp = [wPearson(yp,yt,pad, reduction='none') for yp,yt,pad in zip(self.Y_pre, self.Y, self.maskPAD)]
            return np.sum([i[0] for i in tmp]) / np.sum([i[1] for i in tmp])
        else:
            return wPearson(self.Y_pre, self.Y, self.maskPAD)

    def wProtPearsonForPos(self):
        if self.batchList:
            tmp = [wPearson(yp,yt,pad,self.affinityFloor, reduction='none') for yp,yt,pad in zip(self.Y_pre, self.Y, self.maskPAD)]
            return np.sum([i[0] for i in tmp]) / np.sum([i[1] for i in tmp])
        else:
            return wPearson(self.Y_pre, self.Y, self.maskPAD, self.affinityFloor)

    def CalinskiHarabaszScore(self):
        return skmetrics.calinski_harabasz_score(self.Y_pre, self.Y)
    def CI(self):
        return CI(self.Y_pre, self.Y)
    # def pMLM(self):
    #     return LOSS(self.pSeqArrPre,self.pSeqArr)
    # def dMLM(self):
    #     return LOSS(self.dSeqArrPre,self.dSeqArr)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ACC(Y_pre, Y, maskPAD=None):
    # print(Y_pre)
    # print(Y)
    # print()
    if maskPAD is None:
        return (Y_pre==Y).sum() / len(Y)
    else:
        return ((Y_pre==Y)*maskPAD).sum() / maskPAD.sum()

def AUC(Y_prob_pre, Y):
    return skmetrics.roc_auc_score(Y, Y_prob_pre)

def CrossEntropyLOSS(Y_prob_pre, Y):
    Y_prob_pre[Y_prob_pre>0.99] -= 1e-3
    Y_prob_pre[Y_prob_pre<0.01] += 1e-3
    #return -np.mean(Y*np.log(Y_prob_pre) + (1-Y)*np.log(1-Y_prob_pre))
    return -np.mean(np.log(Y_prob_pre[range(len(Y)),Y]))
    
def MSE(Y_pre, Y):
    return np.mean((Y_pre-Y)**2)

def MAE(Y_pre, Y):
    return np.mean(np.abs(Y_pre-Y))

def RANK_SCORE(yp, yt, maskPAD=None):
    yp,yt = np.array(yp),np.array(yt)
    # yp, yt: batchSize (× rankingNum); maskPAD: batchSize (× rankingNum)
    if len(yp.shape)==1:
        x_ = yp.reshape(1,-1) - yp.reshape(-1,1)
        y_ = yt.reshape(1,-1) - yt.reshape(-1,1)
        
        x_[x_>0] = y_[y_>0] = 1
        x_[x_<0] = y_[y_<0] = -1
    
        return (x_==y_).sum() / len(yp)**2
    else:
        B,R = yp.shape

        x_ = yp.reshape(B,1,-1) - yp.reshape(B,-1,1) # => batchSize × rankingNum × rankingNum
        y_ = yt.reshape(B,1,-1) - yt.reshape(B,-1,1) # => batchSize × rankingNum × rankingNum

        x_[x_>0] = y_[y_>0] = 1
        x_[x_<0] = y_[y_<0] = -1

        if maskPAD is None:
            return (x_==y_).sum() / (B*R**2)
        else:
            maskPAD = maskPAD.reshape(B,1,-1) & maskPAD.reshape(B,-1,1) # => batchSize × rankingNum × rankingNum
            maskPAD[:,range(R),range(R)] = False
            return ((x_==y_)*maskPAD).sum() / maskPAD.sum()

# from scipy.stats import stats
def Pearson(yp, yt, maskPAD=None, affinityFloor=None):
    yp,yt = np.array(yp),np.array(yt)
    # yp, yt: batchSize (× rankingNum); maskPAD: batchSize (× rankingNum)
    if len(yp.shape)==1 or maskPAD is None:
        if affinityFloor is not None:
            yp,yt = yp[yt>affinityFloor],yt[yt>affinityFloor]
        return pearsonr(yp, yt)
    else:
        if affinityFloor is not None:
            maskPAD &= (yt>affinityFloor)
        tmp,cnt = 0,0
        for a,b,mask in zip(yp,yt,maskPAD):
            a,b = a[mask],b[mask]
            if len(a)>1:
                tmp += pearsonr(a,b)
                cnt += 1
        return tmp/cnt

def IE(yt):
    s = np.sum(yt)
    yt = yt/s
    return -np.sum(yt*np.log(yt))

def wPearson(yp, yt, maskPAD=None,affinityFloor=None, reduction='mean'):
    yp,yt = np.array(yp),np.array(yt)
    # yp, yt: batchSize × rankingNum; maskPAD: batchSize × rankingNum
    if maskPAD is None:
        maskPAD = np.ones(yt.shape, dtype=np.bool)
    if affinityFloor is not None:
        maskPAD &= (yt>affinityFloor)

    tmp,cnt = 0,0
    for a,b,mask in zip(yp,yt,maskPAD):
        a,b = a[mask],b[mask]
        if len(a)>1:
            # w = np.sum([np.log(i) for i in range(1,len(a)+1)])
            w = IE(b)
            tmp += w*pearsonr(a,b)
            cnt += w
    if reduction=='mean':
        return tmp/cnt
    else:
        return tmp,cnt

def pearsonr(yp,yt):
    assert (yp.shape==yt.shape and len(yt.shape)<=2)
    ypMean,ytMean = yp.mean(axis=-1, keepdims=True),yt.mean(axis=-1, keepdims=True)
    ypStd,ytStd = yp.std(axis=-1, keepdims=True),yt.std(axis=-1, keepdims=True)
    tmp = np.mean((yp-ypMean) * (yt-ytMean), axis=-1, keepdims=True) / (ypStd*ytStd+1e-8)
    return tmp.mean()

def CI(yp,yt):
    ind = np.argsort(yt)
    yp,yt = yp[ind],yt[ind]
    yp_diff = yp.reshape(1,-1) - yp.reshape(-1,1)
    yt_diff = yt.reshape(1,-1) - yt.reshape(-1,1)
    yp_diff,yt_diff = np.triu(yp_diff,1),np.triu(yt_diff,1)
    tmp = yp_diff[yt_diff>0].reshape(-1)
    return (1.0*(tmp>0).sum() + 0.5*(tmp==0).sum()) / len(tmp)

# def CI(yp,yt):
#     y,f = yt,yp
#     ind = np.argsort(y)
#     y = y[ind]
#     f = f[ind]
#     i = len(y)-1
#     j = i-1
#     z = 0.0
#     S = 0.0
#     while i > 0:
#         while j >= 0:
#             if y[i] > y[j]:
#                 z = z+1
#                 u = f[i] - f[j]
#                 if u > 0:
#                     S = S + 1
#                 elif u == 0:
#                     S = S + 0.5
#             j = j - 1
#         i = i - 1
#         j = i-1
#     ci = S/z
#     return ci