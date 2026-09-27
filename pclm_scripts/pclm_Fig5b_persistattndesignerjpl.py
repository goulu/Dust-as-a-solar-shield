#
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import time
import pylab as pl
import os
import glob
from nbody import *
from constants import *
import attnmie
import myfig
from myparms import *

def setorb(b,drdv,tracerindex=3): # reconsider returning, not this crap
    xs,ys,zs = b[0].x,b[0].y,b[0].z
    vxs,vys,vzs = b[0].vx,b[0].vy,b[0].vz
    # work in sun frame
    xem = np.sum((b[1:3].x-xs)*b[1:3].m)/np.sum(b[1:3].m)
    yem = np.sum((b[1:3].y-ys)*b[1:3].m)/np.sum(b[1:3].m)
    zem = np.sum((b[1:3].z-zs)*b[1:3].m)/np.sum(b[1:3].m)
    rem = np.sqrt(xem**2+yem**2+zem**2)
    vxem = np.sum((b[1:3].vx-vxs)*b[1:3].m)/np.sum(b[1:3].m)
    vyem = np.sum((b[1:3].vy-vys)*b[1:3].m)/np.sum(b[1:3].m)
    vzem = np.sum((b[1:3].vz-vzs)*b[1:3].m)/np.sum(b[1:3].m)
    #dr0 = -rem*(b[1].m/3/b[0].m)**(1/3)
    dr0fid = -rem * (b[1].m/b[0].m/3)**(1/3) # interior... 
    rfac = 1+dr0fid/rem
    dx,dy,dz,dvx,dvy,dvz = drdv
    b[tracerindex:].x,b[tracerindex:].y,b[tracerindex:].z = rfac*xem+xs+dx,rfac*yem+ys+dy,rfac*zem+zs+dz
    vfac = 1+dr0fid/rem
    b[tracerindex:].vx,b[tracerindex:].vy,b[tracerindex:].vz=vfac*vxem+vxs+dvx,vfac*vyem+vys+dvy,vfac*vzem+vzs+dvz       
    return

def do_attn(Mdust,dustradius,rho,nindex):
    c = clight
    G = GNewt
    global Lsolar,Msolar,Rsolar,Mearth,des
    qext, qsca, g = attnmie.do_mie(dustradius,nindex)
    qpr = qext-g*qsca
    beta = 3*Lsolar*qpr/(16*np.pi*G*c*rho*dustradius*Msolar)
    d1 = attnmie.get_d1(Msolar,Mearth,des,np.zeros(dustradius.shape),G)
    rshadow = Rsolar*d1/(des-d1)
    fshadow = Rearth**2/rshadow**2
    msk = d1<0
    fshadow[msk] = 0.0
    attn = np.zeros(dustradius.shape)
    attn = fshadow * qext * dustradius**2/Rearth**2 * Mdust/(4/3*np.pi*rho*dustradius**3)
    return attn, beta, d1

# ---- Main -----

np.random.seed(94123421)

nr = 100
dustradius = np.logspace(np.log10(0.0007*micron),np.log10(1200*micron),nr)
ntracer = nr

fil = 'solarsystem.csv'
dfsolsys = pd.read_csv(fil)
    
sun = dfsolsys.iloc[0]
earth = dfsolsys.iloc[3]
moon = dfsolsys.iloc[4]
venus = dfsolsys.iloc[2]
mars = dfsolsys.iloc[5]
jupiter = dfsolsys.iloc[6]
saturn = dfsolsys.iloc[7]

des = np.sqrt((earth.x-sun.x)**2+(earth.y-sun.y)**2+(earth.z-sun.z)**2)
Msolar = sun.m
Mearth = earth.m
Rsolar = sun.r
Rearth = earth.r
Lsolar = 3.828e33

# nb = 3 # number of massive nbodies
nb = 7 # include nearby, big planets
b = np.zeros(nb+ntracer,dtype=bodyt).view(np.recarray)
b[0] = (sun.m,sun.r,sun.x,sun.y,sun.z,sun.vx,sun.vy,sun.vz,0,0,0,0,0)
b[1] = (earth.m,earth.r,earth.x,earth.y,earth.z,earth.vx,earth.vy,earth.vz,0,0,0,0,0)
b[2] = (moon.m,moon.r,moon.x,moon.y,moon.z,moon.vx,moon.vy,moon.vz,0,0,0,0,0)
if nb>3: b[3] = (venus.m,venus.r,venus.x,venus.y,venus.z,venus.vx,venus.vy,venus.vz,0,0,0,0,0)
if nb>4: b[4] = (mars.m,mars.r,mars.x,mars.y,mars.z,mars.vx,mars.vy,mars.vz,0,0,0,0,0)
if nb>5: b[5] = (jupiter.m,jupiter.r,jupiter.x,jupiter.y,jupiter.z,jupiter.vx,jupiter.vy,jupiter.vz,0,0,0,0,0)
if nb>6: b[6] = (saturn.m,saturn.r,saturn.x,saturn.y,saturn.z,saturn.vx,saturn.vy,saturn.vz,0,0,0,0,0)

ti = tracerindex = nb
bm = b[:nb] # all massive....
comr = np.sum(np.array([bm.m*bm.x,bm.m*bm.y,bm.m*bm.z])/np.sum(bm.m),axis=1)
comv = np.sum(np.array([bm.m*bm.vx,bm.m*bm.vy,bm.m*bm.vz])/np.sum(bm.m),axis=1)
b.x -= comr[0]; b.y -= comr[1]; b.z -= comr[2];
b.vx -= comv[0]; b.vy -= comv[1]; b.vz -= comv[2];

drdv = np.array([9.7672356395251e+07, 1.0145546625866e+08, 1.0474139184824e+05, 1.0271868687687e+01, 9.7177468883876e+00, 9.7381395186548e+00]) # best quad
setorb(b,drdv,tracerindex=ti)
bstart = b.copy()        

tmax = 1.0*yr
ddt = 1*yr/4000
npl = 400 # numper of plot calls
rhop = 2.7
micron = 1e-4
rp = micron

tmax = 200*24*3600
npl = 2000
dt = tmax/npl
ntsub = int(dt/ddt)+1

nindex = 1.51 - 1e-8j; rho = 2.51 # glass
nindex = nglass; rho = rhoglass

nr = 100
# dustradius is volumne equiv radius
dustradius = np.logspace(np.log10(0.005*micron),np.log10(1200*micron),nr)
dustVol = 4*np.pi/3*dustradius**3
dustmass = dustVol*rho



clis = ['goldenrod']
slis = ['rod (100:1)']
c=clight
# dustVol = np.pi*(dustD/2)**3*asp
aspect = 100; 
asp = aspect
dustD = (dustVol/np.pi/asp)**(1/3)
dustL = asp*dustD
Aproj = dustL*dustD*0.785
radAreaEquiv = np.sqrt(Aproj/np.pi)
radVolEquiv = dustradius
# qext, qsca, g = do_mie(radAreaEquiv,nindex)
qextrod, qscarod, g = attnmie.do_mie(radVolEquiv,nindex) # could use area-equiv...
qprrod = qextrod-g*qscarod
radarearod = np.copy(radAreaEquiv)
betarod = Lsolar*qprrod*radAreaEquiv**2/(4*np.pi*GNewt*clight*dustmass*Msolar)
d1 = attnmie.get_d1(Msolar,Mearth,des,0*betarod,GNewt)
rshadow = Rsolar*d1/(des-d1)
fshadow = Rearth**2/rshadow**2
attnrod = fshadow * qextrod * (radAreaEquiv/Rearth)**2 * Mdust/dustmass

# dustVol = np.pi*(dustD/2)**3*asp
aspect = 100; clis.append('darkviolet'); slis.append('tube (100:1:0.1)')
cylthick = 0.2 # frac of radius
ff = 1-(1-cylthick)**2  # filling frac, hollow cylinder
dustD = (dustVol/np.pi/asp/ff)**(1/3)
dustL = aspect*dustD
ef = np.real(nindex)**2
fff = lambda x: (1-ff)*(1-x)/(1+2*x)+ff*(ef-x)/(ef+2*x)
neff = np.sqrt(root_scalar(fff,x0=1,x1=ef,rtol=1e-8).root)
Aproj = dustL*dustD*0.785
radAreaEquiv = np.sqrt(Aproj/np.pi)
radVolEquiv = dustradius
# qext, qsca, g = do_mie(radAreaEquiv,nindex)
qexttube, qscatube, g = attnmie.do_mie(radVolEquiv,neff) # could use area-equiv...
qprtube = qexttube-g*qscatube
betatube = Lsolar*qprtube*radAreaEquiv**2/(4*GNewt*clight*(dustVol*rho)*Msolar)
radareatube = np.copy(radAreaEquiv)
d1 = attnmie.get_d1(Msolar,Mearth,des,0*betatube,GNewt)
rshadow = Rsolar*d1/(des-d1)
fshadow = Rearth**2/rshadow**2
attntube = fshadow * qexttube * (radAreaEquiv/Rearth)**2 * Mdust/(dustVol*rho)
    
flufffill = [0.5,0.1]
clis.extend(['#4477aa','#99aaff'])
slis.extend(['fluff ball (0.5)','fluff ball (0.1)'])
for i,ff in enumerate(flufffill):
    ef = np.real(nindex)**2
    fff = lambda x: (1-ff)*(1-x)/(1+2*x)+ff*(ef-x)/(ef+2*x)
    neff = np.sqrt(root_scalar(fff,x0=1,x1=ef,rtol=1e-8).root)
    # for this radius, the eff vall radius, find the physics radius....
    # Vfluff = f*Veff so
    rfluff = dustradius/ff**(1/3)
    qext, qsca, g = attnmie.do_mie(rfluff,neff) # could use area-equiv...
    qpr = qext-g*qsca
    mdust = 4*np.pi/3*dustradius**3*rho
    betaf = Lsolar*qpr*rfluff**2/(4*GNewt*clight*mdust*Msolar)
    d1 = attnmie.get_d1(Msolar,Mearth,des,0*betaf,GNewt)
    rshadow = Rsolar*d1/(des-d1)
    fshadow = Rearth**2/rshadow**2
    msk = d1<0
    fshadow[msk] = 0.0
    attn = fshadow * qext * (rfluff/Rearth)**2 * Mdust/mdust
    if i==0:
        attnfluff0p5 = np.copy(attn)
        radareafluff0p5 = np.copy(rfluff)
        qextfluff0p5 = np.copy(qext)
        qprfluff0p5 = np.copy(qpr)
    else:
        attnfluff0p1 = np.copy(attn)
        radareafluff0p1 = np.copy(rfluff)
        qextfluff0p1 = np.copy(qext)
        qprfluff0p1 = np.copy(qpr)

clis.append('black'); slis.append('sphere')
qextsphere, qscasphere, g = attnmie.do_mie(dustradius,nindex) # could use area-equiv...
qprsphere = qextsphere-g*qscasphere
mdust = 4*np.pi/3*dustradius**3*rho
d1 = attnmie.get_d1(Msolar,Mearth,des,0*dustradius,GNewt)
rshadow = Rsolar*d1/(des-d1)
fshadow = Rearth**2/rshadow**2
radareasphere = dustradius
attnsphere = fshadow * qextsphere * (dustradius/Rearth)**2 * Mdust/mdust
# pl.loglog(dustradius/micron,attns,'-',lw=2, c='k',label='solid sphere')

#print(slis,clis);quit()

for j in range(5):
    if 'rod' in slis[j]:
        attn = attnrod;  qpr = qprrod; qext = qextrod; radarea = radarearod
    elif 'tube' in slis[j]:
        attn = attntube;  qpr = qprtube; qext = qexttube; radarea = radareatube
    elif 'fluff' in slis[j] and '0.5' in slis[j]:
        attn = attnfluff0p5; qpr = qprfluff0p5; qext = qextfluff0p5; radarea = radareafluff0p5
    elif 'fluff' in slis[j] and '0.1' in slis[j]:
        attn = attnfluff0p1; qpr = qprfluff0p1; qext = qextfluff0p1; radarea = radareafluff0p1
    else:
        attn = attnsphere; qpr = qprsphere; qext = qextsphere; radarea = radareasphere
    b = bstart.copy()
    b[ti:].r = radarea
    b[ti:].m = dustmass; b[ti:].eta = 1/3.
    b[ti:].Q = 0

    tpersist = np.zeros(ntracer)
    startt = 0.0*tmax
    started = False
    attng0 = attnmie.attngeo(radarea,np.zeros(dustradius.shape),d1=d1,asemi=AU) # an array

    for i in range(npl):
        steps(b,dt,ntsub)
        re = pairsep(b[1],b[0])
        tnow = dt*(i+1)
        if not started:
            if tnow >= startt:
                started = True
                b[ti:].Q = qpr
        else:
            #th, ph = skycoords(b[:],ti)
            #msk = np.sqrt(th**2+ph**2)<b[0].r/pairsep(b[1],b[0])
            #tpersist[msk] += dt
            lat, lon = attnmie.skycoords(b,tracerindex=ti)
            ang = np.sqrt(lat**2+lon**2)
            attng = attnmie.attngeo(radarea,ang,d1=d1,asemi=AU)
            if (rp <= 1*micron) and (tnow-startt)>40*(24*3600):
                break
            tpersist += dt * attng/attng0

    # pl.loglog(dustradius/micron,beta,'-b',lw=2)
    pl.loglog(dustradius/micron,tpersist*attn/(24*3600),'-',lw=2, label=slis[j], c=clis[j])
    stuff = slis[j]
    cumattn = tpersist*attn
    idmax = np.argmax(cumattn)
    rdmax = dustradius[idmax]/micron
    camax = cumattn[idmax]/(24*3600)
    mneed = Mdust*6.0/camax/1e3
    print(stuff,'rdmax: %1.4f microns, %1.3f attn-days, mass for 6 attn-days: %1.5e kg'%(rdmax,camax,mneed))

pl.xlim(dustradius[0]/micron,dustradius[-1]/micron)
pl.ylim(3e-06,7e0)
pl.xlabel(r'dust radius ($\mu$m)',size=14)
#pl.ylabel(r'$T_{per}\times{\cal{A}}_0$ (days)',size=14)
pl.ylabel(r'cumulative attenuation (days)',size=14)
pl.legend(loc='upper right')
pl.figtext(0.18,0.8,r'cloud mass: '+Mduststr)

out = 'persistattndesignerjpl.pdf'
pl.savefig(out)
os.system('convert '+out+' ~/public_html/tmp.jpg')

myscript = os.path.basename(__file__)
fignm = myscript.split('_')[1]
print(fignm)
pl.savefig(fignm+".pdf")
