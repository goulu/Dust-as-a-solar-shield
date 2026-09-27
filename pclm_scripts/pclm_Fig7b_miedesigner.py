import numpy as np
import pylab as pl
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import attnmie
import myfig
from myparms import *
from constants import *


G = GNewt
c = clight
des = AU

nindex = nglass; rho = rhoglass

nr = 100
# dustradius is volumne equiv radius
dustradius = np.logspace(np.log10(0.0007*micron),np.log10(800*micron),nr)
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
    fshadow[msk] = 0.0
    #attn = fshadow * qext * (radAreaEquiv/d1)**2 / (Rsolar/des)**2 * Mdust/(dustVol*rho)
    attn = fshadow * qext * (radAreaEquiv/Rearth)**2 * Mdust/(dustVol*rho)
    pl.loglog(dustradius/micron,attn,'-',lw=2, c=col,label='rod (%g:1)'%(asp))
    stuff = 'rod (%g:1)'%(asp)
    idmax = np.argmax(attn)
    rdmax = dustradius[idmax]/micron
    camax = attn[idmax]*365.25
    d1max = d1[idmax]/AU
    mneed = Mdust*6.0/camax/1e3 # kg
    print(stuff,'rmax: %1.3g microns, attn*year: %1.3g, 6-day cumattn mass: %1.3e kg @ %1.4f au'%(rdmax,camax,mneed,d1max))


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
    fshadow[msk] = 0.0
    #attn = fshadow * qext * (radAreaEquiv/d1)**2 / (Rsolar/des)**2 * Mdust/(dustVol*rho)
    attn = fshadow * qext * (radAreaEquiv/Rearth)**2 * Mdust/(dustVol*rho)
    pl.loglog(dustradius/micron,attn,'-',lw=2, c=col,label='tube (%g:1:%g)'%(asp,cylthick/2))
    stuff = 'tube (%g:1:%g)'%(asp,cylthick/2)
    idmax = np.argmax(attn)
    rdmax = dustradius[idmax]/micron
    camax = attn[idmax]*365.25
    d1max = d1[idmax]/AU
    mneed = Mdust*6.0/camax/1e3 # kg
    print(stuff,'rmax: %1.3g microns, attn*year: %1.3g, 6-day cumattn mass: %1.3e kg @ %1.4f au'%(rdmax,camax,mneed,d1max))





    
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
    fshadow[msk] = 0.0
    #attnf = fshadow * qext * (rfluff/d1)**2 / (Rsolar/des)**2 * Mdust/mdust
    attnf = fshadow * qext * (rfluff/Rearth)**2 * Mdust/mdust
    pl.loglog(dustradius/micron,attnf,'-',lw=2, c=col,label='fluff ball (%g)'%(ff))
    stuff = 'fluff ball (%g)'%(ff)
    idmax = np.argmax(attnf)
    rdmax = dustradius[idmax]/micron
    camax = attnf[idmax]*365.25
    d1max = d1[idmax]/AU
    mneed = Mdust*6.0/camax/1e3 # kg
    print(stuff,'rmax: %1.3g microns, attn*year: %1.3g, 6-day cumattn mass: %1.3e kg @ %1.4f au'%(rdmax,camax,mneed,d1max))



qext, qsca, g = attnmie.do_mie(dustradius,nindex) # could use area-equiv...
qpr = qext-g*qsca
mdust = 4*np.pi/3*dustradius**3*rho
beta = Lsolar*qpr*dustradius**2/(4*G*c*mdust*Msolar)
d1 = attnmie.get_d1(Msolar,Mearth,des,beta,G)
rshadow = Rsolar*d1/(des-d1)
fshadow = Rearth**2/rshadow**2
msk = d1<0
fshadow[msk] = 0.0
attns = fshadow * qext * (dustradius/d1)**2 / (Rsolar/des)**2 * Mdust/mdust
attns = fshadow * qext * (dustradius/Rearth)**2 * Mdust/mdust
pl.loglog(dustradius/micron,attns,'-',lw=2, c='k',label='solid sphere')
stuff = 'solid sphere'
idmax = np.argmax(attns)
rdmax = dustradius[idmax]/micron
camax = attns[idmax]*365.25
d1max = d1[idmax]/AU
mneed = Mdust*6.0/camax/1e3 # kg
print(stuff,'rmax: %1.3g microns, attn*yr: %1.3g, 6-day cumattn mass: %1.3e kg @ %1.4f au'%(rdmax,camax,mneed,d1max))




rball = (Mdust/(4*np.pi/3*rho))**(1/3)
print('rball (m):',rball/1e2,rho)

pl.xlim((dustradius[0]/micron,dustradius[-1]/micron))
pl.ylim(3e-9,3e-2)

xx, yy = [dustradius[0]/micron,dustradius[-1]/micron], [attnlevel]*2
pl.loglog(xx,yy,'--',c='#aaaaaa')
pl.xlabel(r'particle radius ($\mu$m)')
pl.ylabel(r'attenuation')
pl.figtext(0.18,0.8,r'cloud mass: '+Mduststr)
pl.legend(loc = 'lower right')

ofil = 'miedesigner.pdf'
pl.savefig(ofil)
import os
os.system('convert '+ofil+' ~/www/tmp.jpg')

myscript = __file__
fignm = myscript.split('_')[1]
print(fignm)
pl.savefig(fignm+".pdf")
