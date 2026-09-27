import numpy as np
import miepython as mp
import scatter as sc
import pylab as pl
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import time
import os
import myfig
import attnmie
from constants import *
from myparms import *


des = AU

def do_attn(Mdust,dustradius,rho,nindex):
    global Lsolar,Msolar,Rsolar,Mearth,des,GNewt,clight
    c = clight; G  = GNewt
    qext, qsca, g = attnmie.do_mie(dustradius,nindex)
    qpr = qext-g*qsca
    beta = 3*Lsolar*qpr/(16*np.pi*G*c*rho*dustradius*Msolar)
    d1 = attnmie.get_d1(Msolar,Mearth,des,beta,G)
    rshadow = Rsolar*d1/(des-d1)
    fshadow = Rearth**2/rshadow**2
    msk = d1<0
    fshadow[msk] = 0.0
    attn = np.zeros(dustradius.shape)
    attn = fshadow * qext * dustradius**2/Rearth**2 * Mdust/(4/3*np.pi*rho*dustradius**3)
    return attn, beta, d1

#ncoal = 1.85-0.23j; rhocoal = 1.4 # coal  McCartney et al 1965 Fuel 44, 349
#nsalt = 1.5 - 1e-6j; rhosalt = 2.0
#nsand = 1.45 - 1e-7j; rhosand = 2.7
#nglass = 1.51 - 1e-8j; rhoglass = 2.51
#ngold = 0.27 - 2.92j; rhogold = 19.3
#nal = 1.37-7.6j; rhoal = 2.7

#nolivine = 0.729 - 0.289j; rhoolivine = 2.7 # D. Fabian, Th. Henning, C. Jäger, H. Mutschke, J. Dorschner, O. Werhan, Astron. Astrophys. 378, 228 (2001).
#nolivine = 1.67 - 1e-3j; rhoolivine = 2.7 # D. Fabian, Th. Henning, C. Jäger, H. Mutschke, J. Dorschner, O. Werhan, Astron. Astrophys. 378, 228 (2001).

#print('# n:',nindex,G,c)

nlis = [nsalt,ncoal,nolivine,nal,ngold]
slis = ['sea salt','coal dust','lunar dust','aluminum spheres','gold spheres']
rholis = [rhosalt,rhocoal,rhoolivine,rhoal,rhogold]
clis = ['#8888ff','#000000','#226600','#aaaaaa','#c7a033']

nr = 100
dustradius = np.logspace(np.log10(0.0007*micron),np.log10(1200*micron),nr)

#Mdust = 5e13

pthis = pl.semilogx

for nindex, stuff, rho, color in zip(nlis,slis,rholis,clis):
    attn, beta, d1 = do_attn(Mdust,dustradius,rho,nindex)
    msk = d1<=0
    d1[msk] = 1e99
    #pl.loglog(dustradius/micron,beta,'-',lw=2, label=stuff, c=color)
    #pl.loglog(dustradius/micron,attn,'-',lw=2, label=stuff, c=color)
    pthis(dustradius/micron,d1/AU,'-', label=stuff, c=color)
    rball = (Mdust/(4*np.pi/3*rho))**(1/3)
    print('rball (m):',rball/1e2,rho)

pl.xlim((dustradius[0]/micron,dustradius[-1]/micron))
pl.ylim(0,0.08)


xx, yy = [dustradius[0]/micron,dustradius[-1]/micron], [0.01]*2
pthis(xx,yy,'--',c='#aaaaaa')
pl.xlabel(r'particle radius ($\mu$m)')
pl.ylabel(r'L$_1$-Earth distance (au)')
#pl.figtext(0.18,0.8,r'cloud mass: $5\times 10^{13}$ g')
pl.legend(loc = 'upper right')


myscript = __file__
fignm = myscript.split('_')[1]
print(fignm)
pl.savefig(fignm+".pdf")

ofil = 'mied1.pdf'
pl.savefig(ofil)

import os
os.system('convert '+ofil+' ~/www/tmp.jpg')

