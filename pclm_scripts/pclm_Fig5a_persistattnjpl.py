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

des = AU

def do_attn(Mdust,dustradius,dustarea,dustmass,nindex):
    c=2.99792458e10
    G = 6.67403e-8
    global Lsolar,Msolar,Rsolar,Mearth,des
    qext, qsca, g = attnmie.do_mie(dustradius,nindex)
    qpr = qext-g*qsca
    beta = Lsolar*qpr*dustarea/(4*np.pi*G*c*dustmass*Msolar)
    d1 = attnmie.get_d1(Msolar,Mearth,des,np.zeros(dustradius.shape),G)
    rshadow = Rsolar*d1/(des-d1)
    fshadow = Rearth**2/rshadow**2
    attn = np.zeros(dustradius.shape)
    attn = fshadow * qext * dustradius**2/Rearth**2 * Mdust/dustmass
    return attn, beta, d1, qpr, qext

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

ncoal = 1.85-0.23j; rhocoal = 1.4 # coal  McCartney et al 1965 Fuel 44, 349
nsalt = 1.5 - 1e-6j; rhosalt = 2.0
nsand = 1.45 - 1e-7j; rhosand = 2.7
nglass = 1.51 - 1e-8j; rhoglass = 2.51
ngold = 0.27 - 2.92j; rhogold = 19.3
nal = 1.37-7.6j; rhoal = 2.7

nolivine = 0.729 - 0.289j; rhoolivine = 2.7 # D. Fabian, Th. Henning, C. Jäger, H. Mutschke, J. Dorschner, O. Werhan, Astron. Astrophys. 378, 228 (2001).
nolivine = 1.67 - 1e-3j; rhoolivine = 2.7 # D. Fabian, Th. Henning, C. Jäger, H. Mutschke, J. Dorschner, O. Werhan, Astron. Astrophys. 378, 228 (2001).

#print('# n:',nindex,G,c)

nlis = [nsalt,ncoal,nolivine,nal,ngold]
slis = ['sea salt','coal dust','lunar dust','aluminum spheres','gold spheres']
rholis = [rhosalt,rhocoal,rhoolivine,rhoal,rhogold]
clis = ['#8888ff','#000000','#226600','#aaaaaa','#c7a033']

for nindex, stuff, rho, color in zip(nlis,slis,rholis,clis):
    dustarea, dustmass = np.pi*dustradius**2, 4*np.pi/3*rho*dustradius**3
    attn, beta, d1, qpr, qext = do_attn(Mdust,dustradius,dustarea,dustmass,nindex)
    b = bstart.copy()
    b[ti:].r = dustradius
    b[ti:].m = dustmass; b[ti:].eta = 1/3.
    b[ti:].Q = 0

    tpersist = np.zeros(ntracer)
    cumattn = np.zeros(ntracer)
    startt = 0.0*tmax
    started = False
    attng0 = attnmie.attngeo(dustradius,np.zeros(dustradius.shape),d1=0.01*AU,asemi=AU) # an array
    for i in range(npl):
        steps(b,dt,ntsub)
        re = pairsep(b[1],b[0])
        tnow = dt*(i+1)
        if not started:
            if tnow >= startt:
                started = True
                # b[ti:].Q = (beta/0.213)*(rho/2.7)*b[ti:].r/micron # beta is 0.213 Q / (rhop/2.7 cgs) / (rp/micron)
                b[ti:].Q = qpr
        else:
            #th, ph = skycoords(b[:],ti)
            #msk = np.sqrt(th**2+ph**2)<b[0].r/pairsep(b[1],b[0])
            #tpersist[msk] += dt
            lat, lon = attnmie.skycoords(b,tracerindex=ti)
            ang = np.sqrt(lat**2+lon**2)
            d1 = np.ones(dustradius.shape)*0.01*AU
            attng = attnmie.attngeo(dustradius,ang,d1=d1,asemi=AU)
            if (rp <= 1*micron) and (tnow-startt)>60*(24*3600):
                break
            tpersist += dt * attng/attng0
            cumattn += dt * qext * attng * Mdust/dustmass

    # pl.loglog(dustradius/micron,beta,'-b',lw=2)
    #pl.loglog(dustradius/micron,tpersist*attn/(24*3600),'-',lw=2, label=stuff, c=color)
    #print(cumattn[-1],tpersist[-1]*attn[-1])
    pl.loglog(dustradius/micron,cumattn/(24*3600),'-',lw=2, label=stuff, c=color)
    idmax = np.argmax(cumattn)
    rdmax = dustradius[idmax]/micron
    camax = cumattn[idmax]/(24*3600)
    mneed = Mdust*6.0/camax/1e3
    print(stuff,'rdmax: %1.4f microns, %1.3f attn-days, mass for 6 attn-days: %1.5e kg'%(rdmax,camax,mneed))
    
pl.xlim(dustradius[0]/micron,dustradius[-1]/micron)
pl.ylim(3e-06,7e-1)
pl.xlabel(r'dust radius ($\mu$m)',size=14)
#pl.ylabel(r'$T_{per}\times{\cal{A}}_0$ (days)',size=14)
pl.ylabel(r'cumulative attenuation (days)',size=14)
pl.legend(loc='upper right')
pl.figtext(0.18,0.8,r'cloud mass: '+Mduststr)


myscript = os.path.basename(__file__)
fignm = myscript.split('_')[1]
print(fignm)
pl.savefig(fignm+".pdf")


out = 'persistattnjpl.pdf'
pl.savefig(out)
os.system('convert '+out+' ~/public_html/tmp.jpg')

