#!/usr/bin/env python
# coding: utf-8

# In[66]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas
import scipy.odr.odrpack as odrpack
import random as rand



angles=np.array([20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340])
numbers=np.array([28,26,36,38,64,64,68,101,86,61,69,47,43,38,27,22,13])
error=np.round(np.sqrt(numbers),1)
print(error)
plt.bar(angles,numbers,width=16,align='center',yerr=error)
plt.show()
plt.savefig('hist')


# In[29]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas
import scipy.odr.odrpack as odrpack
import random as rand

ang=np.array([20,40,60,80,100,120,140,160,180])
x=np.round(np.sin(np.radians(ang/2)),2)  
y=(41,48,63,76,107,111,137,162,172)
error=np.round(np.sqrt(y),2)
yerr1=np.array(error)

def f(p, x):
    B,c=p
   
    return B*x+c


linear = odrpack.Model(f)

mydata = odrpack.RealData(x, y ,sy=yerr1)

myodr = odrpack.ODR(mydata, linear, beta0=[0.,1.])

myoutput = myodr.run()

myoutput.pprint()


x_fit = np.linspace(x[0], x[-1], 100)


y_fit = f(myoutput.beta, x_fit)

plt.plot(x_fit,y_fit, label='ODR', lw=3, color='purple')

plt.xlabel('Sin(Theta/2)[Degrees]')
plt.ylabel('dN')
plt.title('Sin(Theta/2) vs dN')

plt.scatter(x,y)

plt.errorbar(x,y,yerr=yerr1, linestyle="None")
print(error)
plt.savefig('linear')


# In[67]:


import numpy as np
import uncertainties
from uncertainties import ufloat
from uncertainties.umath import *

#method1
d_theta=ufloat(np.deg2rad(20),0.001)
#dN/sin(theta/2)
slope=ufloat(171,22)
#distance per turn
a=4.32/30
#taking 2 significant number
dpt=ufloat(0.14,0.01)
#number of shots per each turn
npt=ufloat(20,1)
#flux
I=npt/dpt

#calculating radius
r1=(2*slope)/(I*d_theta)

print(d_theta)
print('Flux=',I)
print('Radius=',r1)


# In[65]:


#method2
#Total number of shots
TN=ufloat(831,29)
r2=TN/(2*I)
print('Radius',r2)


# In[ ]:




