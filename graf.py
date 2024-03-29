import numpy as np
import matplotlib.pyplot as plt

f=open("newdata.txt","r")

v1=[]
v2=[]
v3=[]
for i in range(0,1680):
    a=f.readline()
    valor1,valor2,valor3 = a.split()
    if float(valor3)==1:
        v1.append(float(valor1))
        v2.append(float(valor2))
        v3.append(float(valor3))

print(v1)
fig,ax=plt.subplots()
plt.scatter(v2, v1)
plt.ylim(0,3)
plt.xlim(0,0.0200)
plt.grid(True)
ax.set(xlabel='$Im(n_1)$',ylabel='$Re(n_1)$',title='Região de pinçamento da microesfera \n em função do seu índice de refração $n_1$ \n $raio = 0.5\mu m $ , $n_{água}=1.332$')
plt.show()
