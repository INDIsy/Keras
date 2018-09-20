
# coding: utf-8

# In[4]:


import os,sys
import tensorflow as tf
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
import datetime
import time
from collections import Counter
hdir = ''
wdir = ''


# In[5]:


def nearest(items,pivot):
    temp = min(items, key=lambda x: abs(x - pivot))
    return items.index(temp)


# In[6]:


#kp to idx
kp2idx = {0.0:0, 0.3:1, 0.7:2, 1.0:3, 1.3:4, 1.7:5, 2.0:6, 2.3:7, 2.7:8, 3.0:9, 3.3:10, 3.7:11, 4.0:12, 4.3:13,
           4.7:14, 5.0:15, 5.3:16, 5.7:17, 6.0:18, 6.3:19, 6.7:20, 7.0:21, 7.3:22, 7.7:23, 8.0:24, 8.3:25, 8.7:26, 9.0:27}
#idx to kp
idx2kp = {0:0.0, 1:0.3, 2:0.7, 3:1.0, 4:1.3, 5:1.7, 6:2.0, 7:2.3, 8:2.7, 9:3.0, 10:3.3, 11:3.7, 12:4.0, 13:4.3,
           14:4.7, 15:5.0, 16:5.3, 17:5.7, 18:6.0, 19:6.3, 20:6.7, 21:7.0, 22:7.3, 23:7.7, 24:8.0, 25:8.3, 26:8.7, 27:9.0}    

max_idx_value = 27


# In[7]:


data_kp = np.genfromtxt(hdir + 'output.txt', names="year, month, day, hour, kp", dtype=(int, int, int, float,float))


# In[8]:


kp_time = []
for i in range(len(data_kp["kp"])):
    kp_time.append(datetime.datetime(data_kp["year"][i],data_kp["month"][i],data_kp["day"][i],int(data_kp["hour"][i]),30))

kp_date = []
for i in range(len(data_kp["kp"])):
    kp_date.append(datetime.date(data_kp["year"][i],data_kp["month"][i],data_kp["day"][i]))

tdel = kp_date[-1]-kp_date[0]


# In[9]:


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


# In[10]:


#CH
data_ch_reg = np.genfromtxt(hdir + 'CH_Regions_025scl2.dat', dtype=[('date', '<i8'), ('loc', '<U136')])


# In[11]:


ch_time = []
ch_date = []
for i in range(len(data_ch_reg)):
    stamp = data_ch_reg["date"][i]
    ch_time.append(datetime.datetime(int(str(stamp)[0:4]),int(str(stamp)[4:6]),int(str(stamp)[6:8]),int(str(stamp)[8:10]),int(str(stamp)[10:12])))
    ch_date.append(datetime.date(int(str(stamp)[0:4]),int(str(stamp)[4:6]),int(str(stamp)[6:8])))

ch_date = list(set(ch_date))
ch_date.sort()


# In[12]:


ch_reg = []

for i in range(len(ch_date)):
    nindx = nearest(ch_time,datetime.datetime.combine(ch_date[i],datetime.datetime.min.time()))
    ch_reg.append([data_ch_reg[nindx][0]]+[int(n) for n in data_ch_reg[nindx][1]])


# In[13]:


dat_tot = []
for i in range(3,len(ch_date)):
    idx = kp_max_date.index(ch_date[i])
    dat_tot.append([ch_date[i],ch_reg[i-3][1:],ch_reg[i-2][1:],ch_reg[i-1][1:],kp_max[idx]])


# In[14]:


## train, valid, test data set ##
div_tr = 6
div_va = 3
div_te = 3
div = int(len(dat_tot)/10*div_tr)

train = dat_tot[:int(div)]
val = dat_tot[int(div):int(div/div_tr*(div_tr+div_va))]
test = dat_tot[int(div/div_tr*(div_tr+div_va)):int(div/div_tr*(div_tr+div_va+div_te))]
print(len(train),len(val),len(test))


# In[15]:


x_train = np.array([xi[1:4] for xi in train])
y_train = np.array([yi[4] for yi in train])
y_train = tf.keras.utils.to_categorical(y_train,28)

x_val =  np.array([xi[1:4] for xi in val])
y_val = np.array([yi[4] for yi in val])
y_val = tf.keras.utils.to_categorical(y_val,28)


# In[16]:


x_te = np.array([xi[1:4] for xi in test])
y_te = np.array([yi[4] for yi in test])
y_te = tf.keras.utils.to_categorical(y_te,28)


# In[60]:


# 3ch input 1 kp output
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20, batch_input_shape = (1,3,136), stateful=True))
model.add(tf.keras.layers.Dense(28,activation='softmax'))


# In[61]:


print(model.summary())


# In[62]:


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy','mse'])


# In[63]:


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
        


# In[64]:


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


# In[65]:


num_epochs = 100
pat = 1
custom_hist = CustomHistory()
custom_hist.init()


# In[56]:


#no early stopper
#model.fit(x_train,y_train,epochs=num_epochs,batch_size=1, verbose=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_val,y_val))


# In[ ]:


#No early stopper & prediction process image
#temp1 = [ np.where(r==1)[0][0] for r in y_te[:100]]
#temp2 = x_te[0:100]
for epoch_idx in range(100):    
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_val,y_val))
    model.reset_states()
        


# In[ ]:


#with early stopper & prediction process image
'''
temp1 = [ np.where(r==1)[0][0] for r in y_te[:100]]
temp2 = x_te[0:100]
for epoch_idx in range(num_epochs):
    
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_val,y_val))
    model.reset_states()
    
    temp = model.predict_classes(temp2, batch_size = 1)
    
    plt.figure(figsize=(18,6))
    plt.xlabel('Index',fontsize=12)
    plt.ylabel('Kp Label',fontsize=12)
    plt.title('epochs_'+str(epoch_idx),fontsize=15)
    plt.plot(temp1)
    plt.plot(temp)
    plt.savefig('epochs_'+str(epoch_idx)+'.png')
    plt.close()
    
    pat = EarlyStop(custom_hist,patience=pat)
    if pat <= 0: 
        print('Early stopped')
        break
'''

# In[138]:


model.save(wdir+'ch31.h5')


# In[57]:


%matplotlib inline

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
plt.savefig(wdir+'ch31.png')
plt.show()
plt.close()


# In[58]:


#test evaluate & predict
scores = model.evaluate(x_te,y_te,batch_size=1)
print("Evaluate")
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

yhat = model.predict_classes(x_te,batch_size=1)
temp = [np.where(y == 1)[0][0] for y in y_te]
count = 0
for i in range(len(temp)):
    if temp[i] == yhat[i]: count +=1
print("Predict")
print("acc : %.2f%%" %(count/len(temp)*100))


plt.figure(figsize=(80,6))
plt.plot(temp,'-')
plt.plot(yhat,'-',alpha=1)
plt.savefig(wdir+'test_predict.png',dip=300)


# In[59]:


#train evaluate & predict 
scores = model.evaluate(x_train,y_train,batch_size=1)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

yhat_tr = model.predict_classes(x_train,batch_size=1)
temp2 = [np.where(y == 1)[0][0] for y in y_train]
count = 0
for i in range(len(temp2)):
    if temp2[i] == yhat_tr[i]: count +=1
print("predict")
print("acc : %.2f%%" %(count/len(temp2)*100))

plt.figure(figsize=(240,6))
plt.plot(temp2,'-')
plt.plot(yhat_tr,'-',alpha=1)
plt.savefig(wdir+'train_predict.png',dip=300)

