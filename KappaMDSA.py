#Total Kappa 4
#MDSA+ Theory of Optical Tweezers. This Program calculates the optical trapping stiffness per unit power times the radius (r*dQ/d[rho]) 
#(or K_phi and K_rho, differing only from a constant), with rho in radius units,
#incident on a microsphere, for a microsphere of arbitrary radius and refractive index and a arbitrarily polarized beam.
#
#
#  Cabecalho
from math import pi, ceil
import numpy as np
#import quadpy
from scipy import constants, optimize
from scipy.integrate import quadrature
from scipy.special import jv, jn, yv, gamma, hyp2f1, gammaln
import matplotlib.pyplot as plt
#from cython.parallel import prange

#from numba import autojit, prange

def lnfac(z):
    return gammaln(z+1)


def fac(z):
    return gamma(z+1)
#
#
######################## PARAMETROS DE ENTRADA ##############################################
#metodo de quadratura da integracao; 1 para GaussKronrod (fixo) e 2 para Gauss
#intquad=1

#ordem da quadratura de GaussKronrod
GKO=30
#scheme = quadpy.line_segment.gauss_kronrod(30)


#radial position of the microsphere center
#o objetivo é fazer disso uma variavel que possamos pedir o valor no começo do prog. 
rho=0.

#beam waist in nm defined as follows:
#cte*exp((-2*rho^2)/(w_0^2))
w_0=5.82 

#trapping beam wavelength in micrometers 
lamb_0=1.064 

#numerical aperture of objective 
NA=1.2 #agua->1.2

#radius of objective in mm
Rp=3.6#3.15

#microsphere radius in micrometers  
raio=0.5

#microsphere density in kg/m**3
Sdens=1.04*1e3

#density of the medium in kg/m**3
Mdens=0.997*1e3

#refractive index of medium around microsphere
n_1=1.332 #agua

#real part of refractive index of the microsphere 
n_2=1.576 #poliestireno

#imaginary part of refractive index of microsphere, for absorbing materials
k_2=0.0011

#refractive index of glass
n_3=1.332#(1.51 #igual da silica) #1.332 é igual a agua -> desaparece com a aberração esferica.

#distancia da esfera da interface de vidro, em unidade de raio
L= 5

#Laser power in Watts
LW=.1

######################## PARAMETROS DERIVADOS ##############################################

#volume of the sphere in meters**3
Svol = lambda raio:(4./3.)*pi*(raio**3)*(1e-18)
#print(Svol) #works

#weight - thrust , in efficiency factor dimensionless units
Q_offset = lambda r:( (Sdens-Mdens)*Svol(r)*constants.g )*(constants.c/(n_1*LW))
#print(Q_offset)
#refractive kndex of water relative to glass
N_a=n_1/n_3

#complex refractive index of microsphere
m_2=n_2 + k_2*1j

#wavelength in external medium around microsphere in micrometers
lamb_1=lamb_0/n_1

#wavelength inside microsphere in micrometers
lamb_2=lambda m_2 :lamb_0/m_2

#wavelength inside glass in micrometers
lamb_3=lamb_0/n_3

#wave number in vacuum in micromtrs^-1
K_0=2*pi/lamb_0

#wave number in external medium of microsphere in micromtrs^-1
K_1=2*pi/lamb_1

#wave number inside microsphere in micromtrs^-1
K_2=lambda m_2: 2*pi/lamb_2(m_2)

#wave number in glass in micromtrs^-1
K_3=2*pi/lamb_3

#half-aperture angle of beam in glass
if NA>n_1:
    theta_0 = np.arcsin(n_1/n_3)
else:
    theta_0 = np.arcsin(NA/n_3)
#print(theta_0)
#print(sin(theta_0))

#half-aperture angle of beam in glass in the absence of glass-water interface
theta_1 = np.arcsin(NA/n_3)

#cuidado aqui, temos a funcao gamma da bib. scipy. nao renomear!
gammaV=(n_3*Rp)/(w_0*NA)
#print(gamma)
#print(N_a)

#speed of light in vacuum ->Nao e usado
#c=299792458

#Filling factor
def integrand(s):
    a = np.exp(-2*(s*gammaV)**2)
    b = np.sqrt(( 1 - s**2 )*( N_a**2 - s**2 ))
    c = (np.sqrt(1-s**2) + np.sqrt(N_a**2 - s**2))**2
    return s*a*b/c


#A_tup=quadrature(e,0.,sin(theta_0))
A_tup=quadrature(integrand, 0., np.sin(theta_0))[0] 
#print(A_tup) #works -> tupla contem o resultado da integral em [0] e o erro em [1] 

A=A_tup*16*(gammaV**2)
#print(A) #works

######################## PARAMETROS DE TAMANHO ##############################################

q=lambda r: K_0*r #size parameter in vacuum

q_1=lambda r: K_1*r #size parameter in water

def q_2(r,m_2):
    return K_2(m_2)*r #size parameter inside miscrosphere

q_3=lambda r: K_3*r #size parameter in glass

######## CRITERIO DE WISCOMBE PARA TRUNCAR SERIE DE MULTIPOLO DADO UM PARAM DE TAMANHO #########

def Largest(r,m_2):
    return max(q(r),abs(q_1(r)),abs(q_2(r,m_2)))
#print(Largest(.5))#works

def LastTerm(r,m_2):
    return ceil(abs(Largest(r,m_2)+ 4.05*( Largest(r,m_2)**(1/3) ) +2))
#print(LastTerm(.5))#works

################################# ESPALHAMENTO MIE ###########################################

def psi(j,p):
    return p*np.sqrt(pi/(2*p))*jv(j+1/2,p)

def psiprime(j,p):
    a=(np.sqrt(pi/(2.*p))*jn(j+1/2,p))/2.
    b=np.sqrt(pi/2.)*(jn(j-1/2,p) - jn(j+3/2,p))*(np.sqrt(p/4.))
    return  a + b
#print(psiprime(3,5.8)) #works

def zeta(j,p):
    return psi(j,p) + 1j*np.sqrt(p*pi/2)*yv(j+1/2,p)
#print(zeta(3,5.8)) #works

def zeta1(j,p):
    a=np.sqrt(pi/(2.*p))*jv(j+1/2,p)/2.
    b=np.sqrt(p*pi/2.)*(jv(j-1/2,p)-jv(j+3/2,p))/2. 
    c=1j*np.sqrt(pi/(2*p))*yv(j+1/2,p)/2.
    d=1j*np.sqrt(pi/2.)*(yv(j-1/2,p)-yv(j+3/2,p))/(2*np.sqrt(1/p))
    return a+b+c+d
#print(zeta1(3,2.9)) #real part works #imag part works

################################# COEFICIENTE MIE ###########################################

def an(j,r,m_2):
    r1=q_1(r)
    r2=q_2(r,m_2)
    a = m_2*psiprime(j,r1)*psi(j,r2) - n_1*psi(j,r1)*psiprime(j,r2)
    b = m_2*zeta1(j,r1)*psi(j,r2) - n_1*zeta(j,r1)*psiprime(j,r2)
    return a/b
#print(an(3,4.32)) #works

def bn(j,r,m_2):
    r1=q_1(r)
    r2=q_2(r,m_2)
    a = m_2*psi(j,r1)*psiprime(j,r2) - n_1*psiprime(j,r1)*psi(j,r2)
    b = m_2*zeta(j,r1)*psiprime(j,r2) - n_1*zeta1(j,r1)*psi(j,r2)
    return a/b
#print(bn(3,4.32)) #works

##########  MATRIZES DE ROTACAO DEFINIDAS EM TERMOS DE FUNCOES HIPERGEOMETRICAS  ##########
#############  FUNCAO THETA USADA PARA CALCULO DOS COEFICIENTES DE MULTIPOLO  #############

#atencao: essa funcao nao funciona em python2, pois la nao se pode elevar um numero negativo a uma potencia nao inteira.
def koppa(m):
    if (1.>=m):
        return 1.
    else:
        return (-1.)**(1-m)
#print(koppa(2.))
#print(koppa(.5))
#print(koppa(2.4))
#print(koppa(3.7))  #works

def THETA1(theta):

    return np.arcsin(np.sin(theta)/N_a) #works -> scipy funciona melhor q numpy para arcsin


#print(koppa(1.6))

#Para esferas grandes!
def dMore(t,m,j):       # t->theta , m->M
        sub = np.sqrt((m-1)*(m-1))  #works
        som = np.sqrt((m+1)*(m+1))  #works
        a = np.log(koppa(m)+0j)-lnfac(sub)    #works
        b = lnfac( j - 0.5*(sub + som) + sub + som )  #works
        c = lnfac( j - 0.5*(sub + som) + sub )  #works
        d = lnfac( j - 0.5*(sub + som) )  #works
        f = lnfac( j - 0.5*(sub + som) + som )  #works
        g = np.log(np.sin(THETA1(t)/2.)+0j)*(sub)+np.log(np.cos(THETA1(t)/2.)+0j)*(som)  #works
        #o problema aqui e a funcao theta1, que ainda nao foi definida (no mathematica ela e definida depois)
        #nao houve erro a principio porq ja temos uma variavel com esse nome.
        ha = -( j - 0.5*(sub+som) ) #works
        hb = j - 0.5*( sub+som ) + sub + som + 1.  #works
        hc = sub + 1                #works
        hz = np.sin(THETA1(t)/2.)**2   #works
        h = np.log(hyp2f1(ha, hb, hc, hz)+0j)  #works
        return a + (b+c-d-f)*(0.5) + g + h  #works
#print(dMore(0.4,1.3,0.7)) 

def dLess(t,m,j):       # t->theta , m->M
        return ((1j*pi)*(m+1))+dMore(t,-m,j)  #works
'''

def dMore(t,m,j):       # t->theta , m->M
    sub = np.sqrt((m-1)*(m-1))  #works
    som = np.sqrt((m+1)*(m+1))  #works
    a = koppa(m)/fac(sub)    #works
    b = fac( j - 0.5*(sub + som) + sub + som )  #works
    c = fac( j - 0.5*(sub + som) + sub )  #works
    d = fac( j - 0.5*(sub + som) )  #works
    f = fac( j - 0.5*(sub + som) + som )  #works
    g = (np.sin(THETA1(t)/2.)**(sub))*(np.cos(THETA1(t)/2.)**(som))  #works

    ha = -( j - 0.5*(sub+som) ) #works
    hb = j - 0.5*( sub+som ) + sub + som + 1.  #works
    hc = sub + 1                #works
    hz = np.sin(THETA1(t)/2.)**2   #works
    h = hyp2f1(ha, hb, hc, hz)  #works
    return a * (b*c/d/f)**(0.5) * g * h  #works
#print(dMore(0.4,1.3,0.7)) 


def dLess(t,m,j):       # t->theta , m->M
    return ((-1.)**(m+1))*dMore(t,-m,j)  #works

'''

#############  Amplitude de Transmissão e Funcao de Aberração Esférica  #############

def T(t): #t->theta
    return(2*np.cos(t))/(np.cos(t)+N_a*np.cos(THETA1(t)))  #works
#print(T(0.3))

def fy(theta,paramastigmat):
    return 2.*pi*paramastigmat*(np.sin(theta)/np.sin(theta_1))**2

def Funcao1(rho,t,phiV,m,r,paramastigmat,phizero): #atencao: ja existe variavel phi. phiV e a variavel de dentro desta funcao somente.
    if (type(m)==type(1) and m%2!=0): 
        return (-1j)**(-0.5*(m-1))*jv(-0.5*(m-1),fy(t,paramastigmat))*np.exp(-1j*(m-1)*(phizero-phiV))
    else:
        return 0.
#print(Funcao1(0,0.7,0.3,5,3.2)) #works

def Funcao2(rho,t,phiV,m,r,paramastigmat,phizero):
    if (type(m)==type(1) and m%2!=0): 
        return (-1j)**(-0.5*(m+1))*jv(-0.5*(m+1),fy(t,paramastigmat))*np.exp(-1j*(m+1)*(phizero-phiV))
    else:
        return 0.

def psif(z,t,r,L,paramesferico):   #t->theta
    return q_3(r)*( -(L/N_a)*np.cos(t) + N_a*(L+z)*np.cos(THETA1(t)) ) + (2*pi*paramesferico)*( (np.sin(t)/np.sin(theta_1))**4 - (np.sin(t)/np.sin(theta_1))**2 )
#print(psif(4.,.2,2.1,4)) #works

##########################  COEFICIENTES DE MULTIPOLO  ##########################

def G1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))    
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dMore(t,m,j))  #works
    c=lambda t: T(t)*Funcao1(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def G2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico):#works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))   
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dLess(t,m,j))  #works
    c=lambda t: T(t)*Funcao2(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def GC1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico):  #antigo hone, tambem Glinha #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.cos(THETA1(t)) 
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dMore(t,m,j)) 
    c=lambda t: T(t)*Funcao1(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def GC2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico):  #antigo htwo #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.cos(THETA1(t)) 
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) +  dLess(t,m,j)) 
    c=lambda t: T(t)*Funcao2(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

#print(GC(4.1, 0, 0, 0.3, 5, 3.2, 12)) #works

def Gplus1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t)) 
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dMore(t,m+1,j)) 
    c=lambda t: T(t)*Funcao1(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def Gplus2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t)) 
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dLess(t,m+1,j)) 
    c=lambda t: T(t)*Funcao2(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def Gminus1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t)) 
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dMore(t,m-1,j)) 
    c=lambda t: T(t)*Funcao1(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def Gminus2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t)) 
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dLess(t,m-1,j)) 
    c=lambda t: T(t)*Funcao2(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

############### Derivadas dos Coeficientes de Multipolo ###############

def Gssd1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t))**2    #works
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dMore(t,m,j))  #works
    c=lambda t: T(t)*Funcao1(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def Gssd2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t))**2    #works
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dLess(t,m,j))  #works
    c=lambda t: T(t)*Funcao1(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def Gssq1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t))**2    #works
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dMore(t,m,j))  #works
    c=lambda t: T(t)*Funcao2(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def Gssq2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t))**2    #works
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dLess(t,m,j))  #works
    c=lambda t: T(t)*Funcao2(rho,t,phiV,m,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def Gsst1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t))**2    #works
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dMore(t,m,j))  #works
    c=lambda t: T(t)*Funcao1(rho,t,phiV,m-2,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def Gsst2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico): #works
    a=lambda t: np.sin(t)*np.sqrt(np.cos(t))*np.sin(THETA1(t))**2    #works
    b=lambda t: np.exp(-(gammaV*np.sin(t))**2 + 1j*psif(z,t,r,L,paramesferico) + dLess(t,m,j))  #works
    c=lambda t: T(t)*Funcao2(rho,t,phiV,m+2,r,paramastigmat,phizero)
    e=lambda t: a(t)*b(t)*c(t)
    return quadrature(e,0.,theta_0,maxiter=100)[0]
    #return scheme.integrate(e,[0.,theta_0])

def fator1(Psi):
    return (np.cos(pi*Psi/180.)-np.sin(pi*Psi/180.))*np.exp(-1j*pi*Psi/180.)

def fator2(Psi):
    return (np.cos(pi*Psi/180.)+np.sin(pi*Psi/180.))*np.exp(1j*pi*Psi/180.)

##### Componente axial do fator de eficiencia usado para calcular a posicao de equilibrio axial (valor usado para Krho e Kphi) #####

def Qz(z,rho,phiV,r,L,paramastigmat,phizero,paramesferico,m_2):
    Sum=0.
    fp = np.exp(2*1j*phiV)

    for j in range(1,LastTerm(r,m_2)+1):
        SumS = 0.
        SumE = 0.

        as1 = np.sqrt(j*(j+2))/(j+1)*(an(j,r,m_2)*np.conj(an(j+1,r,m_2))+bn(j,r,m_2)*np.conj(bn(j+1,r,m_2)))
        as2 = np.sqrt(j*(j+2))/(j+1)*(an(j,r,m_2)*np.conj(an(j+1,r,m_2))-bn(j,r,m_2)*np.conj(bn(j+1,r,m_2)))
        bs1 = (2*j+1)/(j*(j+1))*an(j,r,m_2)*np.conj(bn(j,r,m_2))

        ae1 = (2*j+1)*(an(j,r,m_2)+bn(j,r,m_2))
        ae2 = (2*j+1)*(an(j,r,m_2)-bn(j,r,m_2))

        for m in range(-j,j+1):

            c1 = np.sqrt((j+m+1)*(j-m+1))

            Gjm1=G1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)
            Gjm2=G2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)

            Gjm_1=G1(j+1,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)
            Gjm_2=G2(j+1,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)

            GCjm1=GC1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)
            GCjm2=GC2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)

            #termo de espalhamento
            ds=np.array([
                #diretos
                as1*c1*Gjm1*np.conj(Gjm_1) + bs1*m*Gjm1*np.conj(Gjm1),
                as1*c1*Gjm2*np.conj(Gjm_2) - bs1*m*Gjm2*np.conj(Gjm2),
                #cruzados
                as2*c1*Gjm2*np.conj(Gjm_1*fp) + bs1*m*Gjm2*np.conj(Gjm1*fp),
                as2*c1*Gjm1*np.conj(Gjm_2)*fp - bs1*m*Gjm1*np.conj(Gjm2)*fp
                ])
            SumS = SumS + ds
			
            #termo de extinção
            de=np.array([
                #diretos
                ae1*Gjm1*np.conj(GCjm1),
                ae1*Gjm2*np.conj(GCjm2),
                #cruzados
                ae2*Gjm2*np.conj(GCjm1*fp),
                ae2*Gjm1*np.conj(GCjm2)*fp
                ])
            SumE = SumE + de
	
        Sum = Sum + SumE - 2.*SumS

    return Sum*(2*gammaV**2)/(A*N_a)


def interval(a,b,rho,phiV,r,L,paramastigmat,phizero,paramesferico,Psi): #calcula o intervalo [a,b] usado na funcao de achar a raiz
    while Qz(a,rho,phiV,r,L,paramastigmat,phizero,paramesferico,Psi)<0.:
        b=a
        a=a-.1
        #print(a)
    while Qz(b,rho,phiV,r,L,paramastigmat,phizero,paramesferico,Psi)>0.:
        a=b
        b=b+.1
        #print(b)
    return[a,b]     

#AB=interval(-.1,.6) #O intervalo onde a esfera pode ser pinçada nao e mt distante do foco da objetiva, 
#                      e não deve ser muito antes deste, apesar dos efeitos de aberracao esferica
#print(AB)

def zero_Z(Psi):
    QzRaio=lambda z: Qz_eq(z,Psi)
    return optimize.brentq(Qz,AB[0],AB[1],xtol=1e-4) #posição de esquilibrio em z
#print(zero_Z(0.))
#print(Qz(zero_Z))

#CONSTANTE PARA CONSTANTES ELÁSTICAS -> valor em picoNewton
Kc=(n_1/constants.c)*1e9
#print(Kc)

############### KAPPA PHI / KAPPA RHO ###############

def Kphi_Krho_Sum(z,rho,r,L,phiV,paramastigmat,phizero,paramesferico,m_2):
    Sump=0.
    Sumr=0.

    fp = np.exp(2*1j*phiV)

    for j in range(1,LastTerm(r,m_2)+1,1):
        a1 = np.sqrt(j*(j+2))/(j+1)*(an(j,r,m_2)*np.conj(an(j+1,r,m_2))+bn(j,r,m_2)*np.conj(bn(j+1,r,m_2)))
        b1 = np.sqrt(j*(j+2))/(j+1)*(an(j,r,m_2)*np.conj(an(j+1,r,m_2))-bn(j,r,m_2)*np.conj(bn(j+1,r,m_2)))
        a2 = (2*j+1)/(j*(j+1))*np.real(an(j,r,m_2)*np.conj(bn(j,r,m_2)))
        b2 = (2*j+1)/(j*(j+1))*np.imag(an(j,r,m_2)*np.conj(bn(j,r,m_2)))
        ae = (2.*j+1)*np.conj(an(j,r,m_2)+bn(j,r,m_2))
        be = (2.*j+1)*np.conj(an(j,r,m_2)-bn(j,r,m_2))
        
        Sum1p=0.
        Sum2p=0.
        Sumep=0.

        Sum1r=0.
        Sum2r=0.
        Sumer=0.

        #Sum1=np.array([0.,0.,0.,0.])
        #Sum2=np.array([0.,0.,0.,0.])
        #Sume=np.array([0.,0.,0.,0.])
        Gplus1_1=Gplus1(j,z,rho,phiV,-j-1,r,L,paramastigmat,phizero,paramesferico)#
        Gminus1_1=Gminus1(j,z,rho,phiV,-j+1,r,L,paramastigmat,phizero,paramesferico)#
        G1_2=G1(j,z,rho,phiV,-j,r,L,paramastigmat,phizero,paramesferico)#

        Gplus2_1=Gplus2(j,z,rho,phiV,-j-1,r,L,paramastigmat,phizero,paramesferico)#
        Gminus2_1=Gminus2(j,z,rho,phiV,-j+1,r,L,paramastigmat,phizero,paramesferico)#
        G2_2=G2(j,z,rho,phiV,-j,r,L,paramastigmat,phizero,paramesferico)#

        for m in range(-j,j+1):

            c1 = np.sqrt((j+m+1)*(j+m+2))
            c2 = np.sqrt((j-m)*(j+m+1))

            #Gplus1_1=Gplus1(j,z,rho,phiV,m-1,r,L,paramastigmat,phizero,paramesferico)#
            Gplus1_2=Gplus1(j+1,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)#
            Gplus1_3=Gplus1(j,z,rho,phiV,-m-1,r,L,paramastigmat,phizero,paramesferico)#
            Gplus1_4=Gplus1(j+1,z,rho,phiV,-m-2,r,L,paramastigmat,phizero,paramesferico)#
            Gplus1_5=Gplus1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)#

            #Gminus1_1=Gminus1(j,z,rho,phiV,m+1,r,L,paramastigmat,phizero,paramesferico)#
            Gminus1_2=Gminus1(j+1,z,rho,phiV,m+2,r,L,paramastigmat,phizero,paramesferico)#
            Gminus1_3=Gminus1(j,z,rho,phiV,-m+1,r,L,paramastigmat,phizero,paramesferico)#
            Gminus1_4=Gminus1(j+1,z,rho,phiV,-m,r,L,paramastigmat,phizero,paramesferico)#
            Gminus1_5=Gminus1(j,z,rho,phiV,m+2,r,L,paramastigmat,phizero,paramesferico)#

            G1_1=G1(j+1,z,rho,phiV,m+1,r,L,paramastigmat,phizero,paramesferico)#
            #G1_2=G1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)#
            G1_3=G1(j+1,z,rho,phiV,-m-1,r,L,paramastigmat,phizero,paramesferico)#
            G1_4=G1(j,z,rho,phiV,-m,r,L,paramastigmat,phizero,paramesferico)#
            G1_5=G1(j,z,rho,phiV,m+1,r,L,paramastigmat,phizero,paramesferico)#

            #Gplus2_1=Gplus2(j,z,rho,phiV,m-1,r,L,paramastigmat,phizero,paramesferico)#
            Gplus2_2=Gplus2(j+1,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)#
            Gplus2_3=Gplus2(j,z,rho,phiV,-m-1,r,L,paramastigmat,phizero,paramesferico)#
            Gplus2_4=Gplus2(j+1,z,rho,phiV,-m-2,r,L,paramastigmat,phizero,paramesferico)#
            Gplus2_5=Gplus2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)#

            #Gminus2_1=Gminus2(j,z,rho,phiV,m+1,r,L,paramastigmat,phizero,paramesferico)#
            Gminus2_2=Gminus2(j+1,z,rho,phiV,m+2,r,L,paramastigmat,phizero,paramesferico)#
            Gminus2_3=Gminus2(j,z,rho,phiV,-m+1,r,L,paramastigmat,phizero,paramesferico)#
            Gminus2_4=Gminus2(j+1,z,rho,phiV,-m,r,L,paramastigmat,phizero,paramesferico)#
            Gminus2_5=Gminus2(j,z,rho,phiV,m+2,r,L,paramastigmat,phizero,paramesferico)#

            G2_1=G2(j+1,z,rho,phiV,m+1,r,L,paramastigmat,phizero,paramesferico)#
            #G2_2=G2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)#
            G2_3=G2(j+1,z,rho,phiV,-m-1,r,L,paramastigmat,phizero,paramesferico)#
            G2_4=G2(j,z,rho,phiV,-m,r,L,paramastigmat,phizero,paramesferico)#
            G2_5=G2(j,z,rho,phiV,m+1,r,L,paramastigmat,phizero,paramesferico)#

            Gsst1_=Gsst1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)
            Gsst2_=Gsst2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)

            Gssq1_=Gssq1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)
            Gssq2_=Gssq2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)
            
            Gssd1_=Gssd1(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)
            Gssd2_=Gssd2(j,z,rho,phiV,m,r,L,paramastigmat,phizero,paramesferico)                    

            #Termos de Espalhamento 1 (phi)
            
            d1p=np.array([
                    #DIRETOS #works
                    (( Gplus1_1-Gminus1_1 )*np.conj(G1_1) +
                    np.conj( Gplus1_2-Gminus1_2 )*G1_2 
                    -( Gplus1_3-Gminus1_3 )*np.conj(G1_3) -
                    np.conj( Gplus1_4-Gminus1_4 )*G1_4 )*a1, 

                    (( Gplus2_1-Gminus2_1 )*np.conj(G2_1) +
                    np.conj( Gplus2_2-Gminus2_2 )*G2_2 -
                    ( Gplus2_3-Gminus2_3 )*np.conj(G2_3) -
                    np.conj( Gplus2_4-Gminus2_4 )*G2_4)*a1, 
                    #CRUZADOS #works
                    ( ( Gplus2_1-Gminus2_1 )*np.conj(G1_1) +
                    np.conj( Gplus1_2-Gminus1_2 )*G2_2 -
                    ( Gplus2_3-Gminus2_3 )*np.conj(G1_3) -
                    np.conj( Gplus1_4-Gminus1_4 )*G2_4 )*np.conj(fp)*b1, 

                    ( ( Gplus1_1-Gminus1_1 )*np.conj(G2_1) +
                    np.conj( Gplus2_2-Gminus2_2 )*G1_2  -
                    ( Gplus1_3-Gminus1_3 )*np.conj(G2_3) -
                    np.conj( Gplus2_4-Gminus2_4 )*G1_4 )*fp*b1
                    ])                              

            #Sum1+= (real(a1*d1) + real(b1*e1))*c1
            Sum1p = Sum1p + d1p*c1
            

            #Termos de Espalhamento 1 (rho)
            #DIRETOS
            d1r=np.array([
                    #DIRETOS
                    ( ( Gplus1_1-Gminus1_1 )*np.conj(G1_1) + 
                    np.conj( Gplus1_2-Gminus1_2 )*G1_2 + 
                    ( Gplus1_3-Gminus1_3 )*np.conj(G1_3) +
                    np.conj( Gplus1_4-Gminus1_4 )*G1_4 )*a1,

                    ( ( Gplus2_1-Gminus2_1 )*np.conj(G2_1) +
                    np.conj( Gplus2_2-Gminus2_2 )*G2_2 +
                    ( Gplus2_3-Gminus2_3 )*np.conj(G2_3) +
                    np.conj( Gplus2_4-Gminus2_4 )*G2_4 )*a1,
                    #CRUZADOS
                    ( ( Gplus2_1-Gminus2_1 )*np.conj(G1_1) +
                    np.conj( Gplus1_2-Gminus1_2 )*G2_2  +
                    ( Gplus2_3-Gminus2_3 )*np.conj(G1_3) +
                    np.conj( Gplus1_4-Gminus1_4 )*G2_4 )*np.conj(fp)*b1,

                    ( ( Gplus1_1-Gminus1_1 )*np.conj(G2_1) +
                    np.conj( Gplus2_2-Gminus2_2 )*G1_2  +
                    ( Gplus1_3-Gminus1_3 )*np.conj(G2_3) +
                    np.conj( Gplus2_4-Gminus2_4 )*G1_4 )*fp*b1
                    ])
            
            Sum1r = Sum1r + d1r*c1


            #Termos de Espalhamento 2 (phi)

            d2p=np.array([
                    #DIRETOS #works
                    (-( Gplus1_1-Gminus1_1 )*np.conj(G1_5) -
                    np.conj( Gplus1_5-Gminus1_5 )*G1_2 )*a2, 

                    (( Gplus2_1-Gminus2_1 )*np.conj(G2_5) +
                    np.conj( Gplus2_5-Gminus2_5 )*G2_2 )*a2,
                    #CRUZADOS #works up to 10e-6
                    0.+0j,

                    (np.conj( Gplus2_1-Gminus2_1 )*G1_5 +
                    ( Gplus1_5-Gminus1_5 )*np.conj(G2_2) +
                    np.conj( Gplus2_5-Gminus2_5 )*G1_2 +
                    ( Gplus1_1-Gminus1_1 )*np.conj(G2_5))*1j*fp*b2 
                    ])

            #Sum2+= (a2*real(d2) + b2*real(1j*e2*fp))*c2
            Sum2p = Sum2p + d2p*c2
            
            #Termos de Espalhamento 2 (rho)
            d2r=np.array([
                    #DIRETOS
                    (-( Gplus1_1-Gminus1_1 )*np.conj(G1_5) -
                    np.conj( Gplus1_5-Gminus1_5 )*G1_2 )*a2, 

                    ( ( Gplus2_1-Gminus2_1 )*np.conj(G2_5) +
                    np.conj( Gplus2_5-Gminus2_5 )*G2_2 )*a2,
                    #CRUZADOS
                    (-( Gplus2_1-Gminus2_1 )*np.conj(G1_5) -
                    np.conj( Gplus1_5-Gminus1_5 )*G2_2 )*b2*fp*1j,

                    ( ( Gplus1_1-Gminus1_1 )*np.conj(G2_5) +
                    np.conj( Gplus2_5-Gminus2_5 )*G1_2 )*b2*fp*1j                              
                    ])

            #Sum2+= (a2*imag(d2) + b2*real(e2*fp))*c2
            Sum2r = Sum2r + d2r*c2

            #Termos de Extinção (phi)
        
            dep=np.array([
                    #DIRETOS #works 
                    (( Gplus1_1+Gminus1_1 )*np.conj( Gplus1_1-Gminus1_1 ) +
                    (Gsst1_-Gssq1_)*np.conj(G1_2) )*ae,

                    (( Gplus2_1+Gminus2_1 )*np.conj( Gplus2_1-Gminus2_1 ) +
                    (Gssd2_-Gsst2_)*np.conj(G2_2) )*ae,
                    #CRUZADOS
                    (( Gminus2_1+Gplus2_1 )*np.conj( Gplus1_1-Gminus1_1 ) +
                    (Gssd2_-Gsst2_)*np.conj(G1_2))*np.conj(fp)*be,

                    (( Gplus1_1+Gminus1_1 )*np.conj( Gplus2_1-Gminus2_1 ) +
                    (Gsst1_-Gssq1_)*np.conj(G2_2))*fp*be
                    ])

            Sumep = Sumep + dep

            #Termos de Extinção (rho)

            der=np.array([
                    #DIRETOS
                    ( ( Gplus1_1-Gminus1_1 )*np.conj( Gplus1_1-Gminus1_1 ) +
                    (Gsst1_+Gssq1_-2.*Gssd1_)*np.conj(G1_2) )*ae,

                    ( ( Gplus2_1-Gminus2_1 )*np.conj( Gplus2_1-Gminus2_1 ) +
                    (Gsst2_+Gssd2_-2.*Gssq2_)*np.conj(G2_2) )*ae,
                    #CRUZADOS
                    -(( Gminus2_1-Gplus2_1 )*np.conj( Gplus1_1-Gminus1_1 ) +
                    (-Gsst2_-Gssd2_-2.*Gssq2_)*np.conj(G1_2))*np.conj(fp)*be,

                    (( Gplus1_1-Gminus1_1 )*np.conj( Gplus2_1-Gminus2_1 ) +
                    (Gsst1_+Gssq1_-2.*Gssd1_)*np.conj(G2_2))*fp*be
                    ])

            Sumer = Sumer + der	

            Gplus1_1=Gplus1_5
            Gminus1_1=Gminus1_5
            G1_2=G1_5
            Gplus2_1=Gplus2_5
            Gminus2_1=Gminus2_5
            G2_2=G2_5

        Sump+=np.array([Sum1p, 2.*Sum2p, Sumep/2.])

        Sumr+=np.array([Sum1r,2.*Sum2r,Sumer/2.])


    return -( np.array([Sump,Sumr])*(gammaV**2)*K_3/A)*Kc
    #return -Sum*(gammaV**2)*K_3/A

def Kphi(z,rho,r,L,phiV,paramastigmat,phizero,paramesferico,Psi,m_2):
    f1=fator1(Psi)
    f2=fator2(Psi)
    f11=f1*np.conj(f1)
    f22=f2*np.conj(f2)
    f12=f2*np.conj(f1)
    f21=f1*np.conj(f2)
    K2=Kphi_Krho_Sum(z,rho,r,L,phiV,paramastigmat,phizero,paramesferico,m_2)[0]
    return np.real(K2[0][0]*f11+K2[0][1]*f22+K2[0][2]*f12+K2[0][3]*f21+K2[1][0]*f11+K2[1][1]*f22+K2[1][2]*f12+K2[1][3]*f21+K2[2][0]*f11+K2[2][1]*f22+K2[2][2]*f12+K2[2][3]*f21)

def Krho(z,rho,r,L,phiV,paramastigmat,phizero,paramesferico,Psi,m_2):
    f1=fator1(Psi)
    f2=fator2(Psi)
    f11=f1*np.conj(f1)
    f22=f2*np.conj(f2)
    f12=f2*np.conj(f1)
    f21=f1*np.conj(f2)
    K=Kphi_Krho_Sum(z,rho,r,L,phiV,paramastigmat,phizero,paramesferico,m_2)[1]
    return np.imag(K[0][0]*f11+K[0][1]*f22+K[0][2]*f12+K[0][3]*f21+K[1][0]*f11+K[1][1]*f22)+np.real(K[1][2]*f12+K[1][3]*f21)+np.imag(K[2][0]*f11+K[2][1]*f22+K[2][2]*f12+K[2][3]*f21)

def Kphi_Krho1(z,rho,r,L,phiV,paramastigmat,phizero,paramesferico,m_2):
    K=Kphi_Krho_Sum(z,rho,r,L,phiV,paramastigmat,phizero,paramesferico,m_2)
    Kp=K[0]
    Kr=K[1]
    X=[]
    Y=[]
    Psi_=0.
    while Psi_<180.5:
        f1=fator1(Psi_)
        f2=fator2(Psi_)
        f11=f1*np.conj(f1)
        f22=f2*np.conj(f2)
        f12=f2*np.conj(f1)
        f21=f1*np.conj(f2)
        Y.append(
            np.real(Kp[0][0]*f11+Kp[0][1]*f22+Kp[0][2]*f12+Kp[0][3]*f21+Kp[1][0]*f11+Kp[1][1]*f22+Kp[1][2]*f12+Kp[1][3]*f21+Kp[2][0]*f11+Kp[2][1]*f22+Kp[2][2]*f12+Kp[2][3]*f21)
            /(np.imag(Kr[0][0]*f11+Kr[0][1]*f22+Kr[0][2]*f12+Kr[0][3]*f21+Kr[1][0]*f11+Kr[1][1]*f22+Kr[1][2]*f12+Kr[1][3]*f21+Kr[2][0]*f11+Kr[2][1]*f22+Kr[2][2]*f12+Kr[2][3]*f21))
            )
        X.append(Psi_)
        Psi_+=1.
    return[Y,X]

def f_K(Psi_,a1p,b1p,c1p,c2p,a2r,b2r,c1r,c2r):
    f1=fator1(Psi_)
    f2=fator2(Psi_)
    f11=f1*np.conj(f1)
    f22=f2*np.conj(f2)
    f12=f2*np.conj(f1)
    return np.real(
        (a1p*f11+b1p*f22+c1p*np.real(f12)+c2p*np.imag(f12))
        /(a2r*f11+b2r*f22+c1r*np.imag(f12)+c2r*np.real(f12)) 
        )

def f_Qz(Psi_,d1,d2,c1,c2):
    f1=fator1(Psi_)
    f2=fator2(Psi_)
    f11=f1*np.conj(f1)
    f22=f2*np.conj(f2)
    f12=f2*np.conj(f1)
    return np.real(d1*f11+d2*f22+c1*f12+c2*np.conj(f12))


#print(Krho(.2,.01,3.,5.,0.,1.,0.,0.1,22.))
