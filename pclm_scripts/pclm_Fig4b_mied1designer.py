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
G = GNewt
c = clight

nindex = nglass; rho = rhoglass



nr = 100
# dustradius is volumne equiv radius
#dustradius = np.logspace(np.log10(0.0007*micron),np.log10(800*micron),nr)
dustradius = np.logspace(np.log10(0.0007*micron),np.log10(1200*micron),nr)
dustVol = 4*np.pi/3*dustradius**3
mdust = dustVol*rho

# dustVol = np.pi*(dustD/2)**3*asp
aspect = [30,300]; clis = ['brown','goldenrod']
aspect = [100]; clis = ['goldenrod']
for asp,col in zip(aspect,clis):
    dustD = (dustVol/np.pi/asp)**(1/3)
    dustL = asp*dustD
    Aproj = dustL*dustD*0.785
    radAreaEquiv = np.sqrt(Aproj/np.pi)
    radVolEquiv = dustradius
    # qext, qsca, g = attnmie.do_mie(radAreaEquiv,nindex)
    qext, qsca, g = attnmie.do_mie(radVolEquiv,nindex) # could use area-equiv...
    qpr = qext-g*qsca
    beta = Lsolar*qpr*radAreaEquiv**2/(4*G*c*(dustVol*rho)*Msolar)
    d1 = attnmie.get_d1(Msolar,Mearth,des,beta,G)
    rshadow = Rsolar*d1/(des-d1)
    fshadow = Rearth**2/rshadow**2
    msk = d1<0
    d1[msk]=1e33; d1 /= AU
    fshadow[msk] = 0.0
    attn = fshadow * qext * (radAreaEquiv/d1)**2 / (Rsolar/des)**2 * Mdust/(dustVol*rho)
    #pl.loglog(dustradius/micron,attn,'-',lw=2, c=col,label='rod (%g:1)'%(asp))
    pl.semilogx(dustradius/micron,d1,'-',lw=2, c=col,label='rod (%g:1)'%(asp))

# dustVol = np.pi*(dustD/2)**3*asp
aspect = [30,300]; clis = ['darkviolet','violet']
aspect = [100]; clis = ['darkviolet']
for asp,col in zip(aspect,clis):
    cylthick = 0.2 # frac of radius
    ff = 1-(1-cylthick)**2  # filling frac, hollow cylinder
    print('ff',ff)
    dustD = (dustVol/np.pi/asp/ff)**(1/3)
    dustL = asp*dustD
    ef = np.real(nindex)**2
    fff = lambda x: (1-ff)*(1-x)/(1+2*x)+ff*(ef-x)/(ef+2*x)
    neff = np.sqrt(root_scalar(fff,x0=1,x1=ef,rtol=1e-8).root)
    Aproj = dustL*dustD*0.785
    radAreaEquiv = np.sqrt(Aproj/np.pi)
    radVolEquiv = dustradius
    # qext, qsca, g = attnmie.do_mie(radAreaEquiv,nindex)
    qext, qsca, g = attnmie.do_mie(radVolEquiv,neff) # could use area-equiv...
    qpr = qext-g*qsca
    beta = Lsolar*qpr*radAreaEquiv**2/(4*G*c*(dustVol*rho)*Msolar)
    d1 = attnmie.get_d1(Msolar,Mearth,des,beta,G)
    rshadow = Rsolar*d1/(des-d1)
    fshadow = Rearth**2/rshadow**2
    msk = d1<0
    d1[msk]=1e33; d1 /= AU
    fshadow[msk] = 0.0
    attn = fshadow * qext * (radAreaEquiv/d1)**2 / (Rsolar/des)**2 * Mdust/(dustVol*rho)
    pl.semilogx(dustradius/micron,d1,'-',lw=2, c=col,label='tube (%g:1:%g)'%(asp,cylthick/2))




    
flufffill = [0.5,0.1]
clis = ['#4477aa','#99aaff']
for ff,col in zip(flufffill,clis):
    ef = np.real(nindex)**2
    fff = lambda x: (1-ff)*(1-x)/(1+2*x)+ff*(ef-x)/(ef+2*x)
    neff = np.sqrt(root_scalar(fff,x0=1,x1=ef,rtol=1e-8).root)
    # for this radius, the eff vall radius, find the physics radius....
    # Vfluff = f*Veff so
    rfluff = dustradius/ff**(1/3)
    qext, qsca, g = attnmie.do_mie(rfluff,neff) # could use area-equiv...
    qpr = qext-g*qsca
    mdust = 4*np.pi/3*dustradius**3*rho
    beta = Lsolar*qpr*rfluff**2/(4*G*c*mdust*Msolar)
    d1 = attnmie.get_d1(Msolar,Mearth,des,beta,G)
    rshadow = Rsolar*d1/(des-d1)
    fshadow = Rearth**2/rshadow**2
    msk = d1<0
    d1[msk]=1e33; d1 /= AU
    fshadow[msk] = 0.0
    attnf = fshadow * qext * (rfluff/d1)**2 / (Rsolar/des)**2 * Mdust/mdust
    #pl.loglog(dustradius/micron,attnf,'-',lw=2, c=col,label='fluff ball (%g)'%(ff))
    pl.semilogx(dustradius/micron,d1,'-',lw=2, c=col,label='fluff ball (%g)'%(ff))





qext, qsca, g = attnmie.do_mie(dustradius,nindex) # could use area-equiv...
qpr = qext-g*qsca
mdust = 4*np.pi/3*dustradius**3*rho
beta = Lsolar*qpr*dustradius**2/(4*G*c*mdust*Msolar)
d1 = attnmie.get_d1(Msolar,Mearth,des,beta,G)
rshadow = Rsolar*d1/(des-d1)
fshadow = Rearth**2/rshadow**2
msk = d1<0
fshadow[msk] = 0.0
d1[msk]=1e33; d1 /= AU
attns = fshadow * qext * (dustradius/d1)**2 / (Rsolar/des)**2 * Mdust/mdust
#pl.loglog(dustradius/micron,attns,'-',lw=2, c='k',label='solid sphere')

pl.semilogx(dustradius/micron,d1,'-',lw=2, c='k',label='solid sphere')

rball = (Mdust/(4*np.pi/3*rho))**(1/3)
print('rball (m):',rball/1e2,rho)

#pl.figtext(0.18,0.8,r'cloud mass: $5\times 10^{12}$ g')

pl.xlim((dustradius[0]/micron,dustradius[-1]/micron))
pl.ylim(0,0.08)

xx, yy = [dustradius[0]/micron,dustradius[-1]/micron], [0.01]*2
pl.semilogx(xx,yy,'--',c='#aaaaaa')
pl.xlabel(r'particle radius ($\mu$m)')
pl.ylabel(r'L$_1$-Earth distance (au)')

pl.legend(loc='upper right')


myscript = os.path.basename(__file__)
fignm = myscript.split('_')[1]
print(fignm)
pl.savefig(fignm+".pdf")

ofil = 'mied1designer.pdf'
pl.savefig(ofil)
import os
os.system('convert '+ofil+' ~/www/tmp.jpg')

