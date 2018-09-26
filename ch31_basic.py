
# coding: utf-8

# In[158]:


import os,sys
#get_ipython().run_line_magic('matplotlib','inline')
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import time
from collections import Counter
import pathlib
hdir = ''
wdir = ''


# In[159]:


def nearest(items,pivot):
    temp = min(items, key=lambda x: abs(x - pivot))
    return items.index(temp)


# In[160]:


#Callback ft
class CustomHistory(tf.keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []        
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        


# In[161]:


#Early stopper
def EarlyStop(history,patience):
    if len(history.val_loss)<=1 : 
        print('hislength')
        return patience
    
    nowloss = history.val_loss[-1]
    lastloss = history.val_loss[-2]
    if nowloss > lastloss :
        patience -= 1
    
    print('patience = '+str(patience))
    print('----------------------')
    return patience


# In[162]:


#kp to idx
kp2idx = {0.0:0, 0.3:1, 0.7:2, 1.0:3, 1.3:4, 1.7:5, 2.0:6, 2.3:7, 2.7:8, 3.0:9, 3.3:10, 3.7:11, 4.0:12, 4.3:13,
           4.7:14, 5.0:15, 5.3:16, 5.7:17, 6.0:18, 6.3:19, 6.7:20, 7.0:21, 7.3:22, 7.7:23, 8.0:24, 8.3:25, 8.7:26, 9.0:27}
#idx to kp
idx2kp = {0:0.0, 1:0.3, 2:0.7, 3:1.0, 4:1.3, 5:1.7, 6:2.0, 7:2.3, 8:2.7, 9:3.0, 10:3.3, 11:3.7, 12:4.0, 13:4.3,
           14:4.7, 15:5.0, 16:5.3, 17:5.7, 18:6.0, 19:6.3, 20:6.7, 21:7.0, 22:7.3, 23:7.7, 24:8.0, 25:8.3, 26:8.7, 27:9.0}    

max_idx_value = 27


# In[163]:


#Kp index
data_kp = np.genfromtxt(hdir + 'output.txt', names="year, month, day, hour, kp", dtype=(int, int, int, float,float))


# In[164]:


#kp index daily number check
#total 6574 days, each 8 values
kp_time = []
for i in range(len(data_kp["kp"])):
    kp_time.append(datetime.datetime(data_kp["year"][i],data_kp["month"][i],data_kp["day"][i],int(data_kp["hour"][i]),30))

kp_date = []
for i in range(len(data_kp["kp"])):
    kp_date.append(datetime.date(data_kp["year"][i],data_kp["month"][i],data_kp["day"][i]))

tdel = kp_date[-1]-kp_date[0]


# In[165]:


#max kp extraction
#kp2idx 
kp_max = []
kp_max_date = []
kp_timeHist = []
for i in range(tdel.days + 1):
    j=8*i
    maxidx = np.argmax(data_kp["kp"][j:j+8])
    kp_max.append(kp2idx[data_kp["kp"][j+maxidx]])
    kp_max_date.append(kp_date[j+maxidx])
    temp = kp_time[j+maxidx].time()
    kp_timeHist.append(temp.hour)


# In[166]:


#CH
data_ch_reg = np.genfromtxt(hdir + 'input.txt', dtype=[('date', '<i8'), ('loc', '<U136')])


# In[167]:


ch_time = []
ch_date = []
for i in range(len(data_ch_reg)):
    stamp = data_ch_reg["date"][i]
    ch_time.append(datetime.datetime(int(str(stamp)[0:4]),int(str(stamp)[4:6]),int(str(stamp)[6:8]),int(str(stamp)[8:10]),int(str(stamp)[10:12])))
    ch_date.append(datetime.date(int(str(stamp)[0:4]),int(str(stamp)[4:6]),int(str(stamp)[6:8])))

ch_date = list(set(ch_date))
ch_date.sort()


# In[236]:


ch_reg = []

for i in range(len(ch_date)):
    nindx = nearest(ch_time,datetime.datetime.combine(ch_date[i],datetime.datetime.min.time()))
    ch_reg.append([data_ch_reg[nindx][0]]+[int(n) for n in data_ch_reg[nindx][1]])


# In[237]:


#same observation data for kp and ch
dat_tot = []
for i in range(3,len(ch_date)):
    idx = kp_max_date.index(ch_date[i])
    dat_tot.append([ch_date[i],ch_reg[i-3][1:],ch_reg[i-2][1:],ch_reg[i-1][1:],kp_max[idx]])


# In[238]:


## train, valid, test data set ##
div_tr = 4
div_va = 4
div_te = 2
div = int(len(dat_tot)/10*div_tr)

train = dat_tot[:int(div)]
val = dat_tot[int(div):int(div/div_tr*(div_tr+div_va))]
test = dat_tot[int(div/div_tr*(div_tr+div_va)):int(div/div_tr*(div_tr+div_va+div_te))]
print(len(train),len(val),len(test))


# In[239]:


x_train = np.array([xi[1:4] for xi in train])
y_train = np.array([yi[4] for yi in train]) 
y_train = tf.keras.utils.to_categorical(y_train,28)

x_val =  np.array([xi[1:4] for xi in val])
y_val = np.array([yi[4] for yi in val])
y_val = tf.keras.utils.to_categorical(y_val,28)


# In[240]:


x_te = np.array([xi[1:4] for xi in test])
y_te = np.array([yi[4] for yi in test])
y_te = tf.keras.utils.to_categorical(y_te,28)


# In[242]:


def corr_SS(temp,yhat):
    y_te_la = np.asarray(temp)
    sigref = np.std(y_te_la)
    avgref = np.mean(y_te_la)

    sigpre = np.std(yhat)
    avgpre = np.mean(yhat)

    co = np.sum((y_te_la - avgref)*(yhat-avgpre))/sigref/sigpre/float(len(y_te_la))
    
    mse_target = np.sum((y_te_la-yhat)**2)/len(y_te_la)
    mse_ref = np.sum((y_te_la-avgref)**2)/len(yhat)
    SS = 1-mse_target/mse_ref
    return co, SS


# In[243]:


def draw_process(idx,path,typ,temp,yhat,res):
    plt.figure(figsize=(80,6))
    plt.title('%s_%i_%i_%i_%0.2f_%0.2f'%(str(typ),idx,res[0],res[1],res[2],res[3]),fontsize=25)
    plt.xlabel('Index',fontsize=20)
    plt.ylabel('Kp Label',fontsize=20)
    plt.plot(temp,'-')
    plt.plot(yhat,'-',alpha=1)
    plt.tight_layout()
    plt.savefig(path+'%s_%i_%i_%i_%0.2f_%0.2f.png'%(str(typ),idx,res[0],res[1],res[2],res[3]),dpi=300)
    plt.close()
    plt.clf()


# In[244]:


def custom_draw(path,custom_hist,nb_epoch, n_neu1, n_neu2, dropout):

    get_ipython().run_line_magic('matplotlib', 'inline')

    loss_ax = plt.subplot()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(custom_hist.train_loss, 'y', label='train loss')
    loss_ax.plot(custom_hist.val_loss, 'r', label='val loss')

    acc_ax.plot(custom_hist.train_acc, 'b', label='train acc')
    acc_ax.plot(custom_hist.val_acc, 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.savefig(path+'history.png')
    plt.show()
    plt.close()
    plt.clf()


# In[245]:


def evaluation(idx,path,model,typ,X,Y,n_batch):
    #test evaluate & predict
    scores = model.evaluate(X,Y,batch_size=n_batch)
    #print("Evaluate")
    #print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

    yhat = model.predict_classes(X,batch_size=n_batch)
    temp = [np.where(y == 1)[0][0] for y in Y]
    count = 0
    for i in range(len(temp)):
        if temp[i] == yhat[i]: count +=1
    
    acc = count/len(temp)
    
    co, SS = corr_SS(temp,yhat)
    res = [scores[1]*100, acc*100, co, SS]
    draw_process(idx,path,typ,temp,yhat,res)
    


# In[249]:


# 3ch input 1 kp output
def fit_lstm(X,Y,Xv,Yv,Xte,Yte, n_batch, nb_epoch, n_neu1, n_neu2, dropout):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(n_neu1, batch_input_shape = (n_batch,X.shape[1],X.shape[2]), stateful=False, return_sequences=True, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.LSTM(n_neu2, input_shape = (n_batch,X.shape[1],X.shape[2]),stateful=False, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(28,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy','mse'])
    print(model.summary())
    
    custom_hist = CustomHistory()
    custom_hist.init()
    
    fpath = wdir+'summary\\'+'%i_%i_%0.1f_%i\\' %(n_neu1,n_neu2,dropout,nb_epoch)
    pathlib.Path(fpath).mkdir(parents=True, exist_ok=True) 
    for i in range(nb_epoch):
        print('epochs : ' + str(i))
        model.fit(X,Y, epochs=1, batch_size = n_batch, verbose=1, shuffle=False,callbacks=[custom_hist], validation_data = (Xv,Yv))
        model.reset_states()
        
        evaluation(i,fpath,model,'test',Xte,Yte,n_batch)
        evaluation(i,fpath,model,'train',X,Y,n_batch)
        
    model.save(fpath+'ch31.h5')
    custom_draw(fpath, custom_hist, nb_epoch, n_neu1, n_neu2, dropout)
    return model,custom_hist


# In[251]:


n_batch = 1
nb_epochs = 100
n_neu1 = 32
n_neu2 = 16
n_dropout = [0.6,0.8]
pat = 1


# In[252]:


for dropout in n_dropout:
    model, custom_hist = fit_lstm(x_train,y_train,x_val,y_val,x_te,y_te,n_batch, nb_epochs,n_neu1,n_neu2,dropout)

