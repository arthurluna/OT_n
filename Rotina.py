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

    def __init__(self,rho,phiV,r,L,phizero,paramesferico,paramastigmat,zi=-4.,zf=1.,pz=20):
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
        Q=self.lista2()
        for i in range(len(self.pontos)-1):
            if Q[i]>0. and Q[i+1]<0. :
                self.trap=1
                return np.array([self.span[i:i+2],np.array([Q[i],Q[i+1]])])
            else:
                self.trap=0
                return

    def __call__(self,m_2): #calcula a posição de equilibrio a partir da dupla de pontos
        self.m_2=m_2
        self.pontos=np.array(self.lista1(self.span))
        a=self.dupla()
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

    L=3+3*k.N_a

    ot_z=QZ(rho,phiV,raio,L,phizero,paramesferico,paramastigmat)
    
    span_n=np.linspace(0.2,3.,15)
    span_k=np.linspace(0,.0015,16)

    DATA=[]

    for i in range(len(span_n)):
        print(i)
        m_2=span_k*1j+span_n[i]
        pin=list(map(ot_z,m_2))
        for l in range(len(pin)):
            DATA.append([span_n[i],span_k[l],pin[l]])

    print(DATA)



    span_chi=list(map(chi_paramastig,span_astig))
    print(list(span_chi)) 
    idx_min=np.where(span_chi==min(list(span_chi))) #retorna o índice do paramastigmat minimo
    print('astigmatismo minimo ='+str(span_astig[idx_min]))

    
    '''
    #script para fazer grafico kphi/krho por psi. 
    exp_plot = [list(data[0]),list(data[1])] #pontos experimentais em forma de lista do python
    chi_paramastig=DataRun(rho,raio,phiV,phizero,paramesferico,data) #define o objeto
    paramastigmat_teste=.233
    teor_plot = list(chi_paramastig.grafico_kappa(paramastigmat_teste))
    print(exp_plot)
    print(teor_plot)
    '''
