#Rotina para ajustar os parametros na pinça ótica por meio do calculo de kappa phi e rho.
from numpy import pi, exp, sqrt, sin, cos, arcsin, arccos, conj, real, imag, log
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import KappaMDSA as k
from multiprocessing import Pool
from scipy import optimize, stats

################################################ KAPPA PHI / KAPPA RHO ################################################

def Qz(z,rho,phiV,r,L,paramastigmat,phizero,paramesferico,m_2):
    return k.Qz(z,rho,phiV,r,L,paramastigmat,phizero,paramesferico,m_2)

def f_Qz(Psi_,d1,d2,c1,c2):
    return k.f_Qz(Psi_,d1,d2,c1,c2)

class QZ:
    '''
    Rotina para calcular posição de equilibrio em z
    '''
    def Qz(self,z):
        return Qz(z,self.rho,self.phiV,self.r,self.L,self.paramastigmat,self.phizero,self.paramesferico,self.m_2)

    def lista1(self,span_z):# Cria uma lista de ( listas de 4 pontos ), referentes ao Qz em qualquer polarização.
        p=Pool(4)
        dat=list(p.map(self.Qz,span_z))
        return dat

    def lista2(self): # Nessa função cria-se uma lista de pontos (escalares) de Qz em funcao de z(span), de acordo com o psi dado. 
        A=[]
        for i in range(len(self.pontos)):
            A.append(f_Qz(self.psi,self.pontos[i][0],self.pontos[i][1],self.pontos[i][2],self.pontos[i][3]))
        return np.array(A)

    def __init__(self,rho,phiV,r,L,phizero,paramesferico,paramastigmat,zi=.05,zf=1.75,pz=11):
        self.rho=rho
        self.r=r
        self.L=L
        self.phiV=phiV
        self.paramastigmat=paramastigmat
        self.phizero=phizero
        self.paramesferico=paramesferico
        self.psi=0.
        self.span=np.linspace(zi,zf,pz)
        assert zf>zi
        if not sqrt(zi**2.)<=L-1.:
            print('Microsphere is out of the bounds of the sample chamber.')

    def dupla(self): 
        self.Q=self.lista2()
        eq_e=0
        eq_i=0
        a=[0,0]

        for i in range(len(self.pontos)-1):
            if self.Q[i]>0. and self.Q[i+1]<0. :
                #print(self.Q[i],self.Q[i+1])
                a=stats.linregress([self.span[i],self.span[i+1]],[self.Q[i],self.Q[i+1]])
                eq_e=-a[1]/a[0]
        
        #barreira=( self.span[eq_e[1]] - eq_e[0] )*self.Q[eq_e[1]]/2. + ( eq_i[0] - self.span[eq_i[1]] )*self.Q[eq_i[1]]/2. + np.trapz(self.Q[eq_e[1]+1:eq_i[1]],x=self.span[eq_e[1]+1:eq_i[1]])

        return eq_e


    def __call__(self,m_2): #calcula a posição de equilibrio a partir da dupla de pontos
        self.m_2=m_2
        self.pontos=self.lista1(self.span)
        #self.Q=self.lista2()
        return self.dupla()
        
        '''if self.trap==1:
                                    b=stats.linregress(a[0],a[1])
                                    return -b[1]/b[0]
                                if self.trap==0:
                                    return '''



if __name__== '__main__':

    raio=2.47
    rho=0.
    phiV=0.
    phizero=0.
    paramesferico=0.
    paramastigmat=0.
    Psi=45

    L=6.34 #+ 3*k.N_a/raio

    ot_z=QZ(rho,phiV,raio,L,phizero,paramesferico,paramastigmat,zi=.38,zf=.47,pz=28)
    
    #span_n=np.linspace(1.,1.8,15)
    #span_k=np.linspace(0.,.0012,4)#list(np.linspace(0.00105,0.00121,5))+list(np.linspace(0.00155,0.0017,5))+list(np.linspace(.002,.004,10))+list(np.linspace(.002,.004,10))
    #sorted(span_k)
    #span_k=np.array(span_k)

    #print(span_k)

    m_2=1.4496
    pin=ot_z(m_2)
    print(pin)

    list_z=k.Qz(pin,rho,phiV,raio,0,paramastigmat,phizero,paramesferico,m_2)
    print( f_Qz(Psi,list_z[0],list_z[1],list_z[2],list_z[3]) )


    kappa_calc=k.Krho(pin,0,raio,0,0,paramastigmat,phizero,paramesferico,Psi,m_2)
    print(kappa_calc)
    #fig,ax=plt.subplots()
    #ax.plot(ot_z.span,ot_z.Q,'ro',label='numerico')
    #ax.set(xlabel='z',ylabel='$Q_z$',title='$Q_z x Z $\n raio = '+ str(raio) +'; n_esf = '+ str(m_2))
    #ax.grid(True)
    #ax.legend()
    #plt.show()

    #print("valor de Delta para n="+str()+":")
    #print(pin)


    