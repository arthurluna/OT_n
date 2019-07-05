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

    def __init__(self,rho,phiV,r,L,phizero,paramesferico,paramastigmat,zi=-1.,zf=3.,pz=10):
        self.rho=rho
        self.r=r
        self.L=L
        self.phiV=phiV
        self.paramastigmat=paramastigmat
        self.phizero=phizero
        self.paramesferico=paramesferico
        self.psi=0.
        self.span=np.linspace(zi,zf,pz)
        self.trap=1
        assert zf>zi
        if not sqrt(zi**2.)<=L-1.:
            print('Microsphere is out of the bounds of the sample chamber.')

    def dupla(self): #dupla de pontos ao redor do ponto de equilibrio, do qual se obtem uma reta
        self.Q=self.lista2()
        self.trap=0
        for i in range(len(self.pontos)-1):
            if self.Q[i]>0. and self.Q[i+1]<0. :
                self.trap=1
                return np.array([self.span[i:i+2],np.array([self.Q[i],self.Q[i+1]])])
        return

    def __call__(self,m_2): #calcula a posição de equilibrio a partir da dupla de pontos
        self.m_2=m_2
        self.pontos=np.array(self.lista1(self.span))
        self.dupla()
        return self.trap
        '''if self.trap==1:
                                    b=stats.linregress(a[0],a[1])
                                    return -b[1]/b[0]
                                if self.trap==0:
                                    return '''



if __name__== '__main__':

    raio=0.5
    rho=0.
    phiV=0.
    phizero=0.
    paramesferico=0.
    paramastigmat=0.

    L=6.34 + 3*k.N_a/raio

    ot_z=QZ(rho,phiV,raio,L,phizero,paramesferico,paramastigmat)
    
    span_n=np.linspace(1.,1.8,15)
    span_k=np.linspace(0.01516666,.0175,16)
    print(span_n)
    print(span_k)
    print("   ")

    DATA=[]


    for i in range(len(span_n)):
        print(i)
        m_2=span_k*1j+span_n[i]
        pin=list(map(ot_z,m_2))
        for l in range(len(pin)):
            DATA.append([span_n[i],span_k[l],pin[l]])

    print(DATA)
    '''
    m_2=1.576+.001*1j
    ot_z(m_2)
    fig,ax=plt.subplots()
    ax.plot(ot_z.span,ot_z.Q,'ro',label='experimental')
    ax.set(xlabel='z',ylabel='$Q_z$',title='$Q_z x z $\n raio = 0.5 ; n_esf = 1.576 + i*0.0011')
    ax.legend()
    plt.show()
    '''