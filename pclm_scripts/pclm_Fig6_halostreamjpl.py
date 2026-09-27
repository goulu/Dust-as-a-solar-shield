#
import numpy as np
import pandas as pd
import miepython as mp
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import time
import pylab as pl
import os
import glob
from nbody import *

import attnmie
from constants import *
import sys
from myparms import *

do_mie = attnmie.do_mie
L1beta = attnmie.L1beta
attengeo = attnmie.attngeo
skycoords = attnmie.skycoords
G = GNewt
c = clight
des = AU

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

def unisphere(n,radius=1):
    u = np.random.uniform(0,1,n) # uniform random vector of size nsamp
    r = radius*np.cbrt(u)
    phi   = np.random.uniform(0,2*np.pi,n)
    theta = np.arccos( np.random.uniform(-1,1,n) )
    return r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)

def powersphere(n,radius=1,gamma=1): # rho = r^-gamma
    u = np.random.uniform(0,1,n) # uniform random vector of size nsamp
    r = radius*u**(1/(3-gamma))
    phi   = np.random.uniform(0,2*np.pi,n)
    theta = np.arccos( np.random.uniform(-1,1,n) )
    return r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)

# funcs to find L1....
def dist2lineEarthSun(b,tracerindex=-1):
    xe,ye,ze = b[1].x-b[0].x,b[1].y-b[0].y,b[1].z-b[0].z  # Earth rel Sun
    xt,yt,zt = b[tracerindex:].x-b[0].x,b[tracerindex:].y-b[0].y,b[tracerindex:].z-b[0].z # tracer rel sun
    q = (xt*xe+yt*ye+zt*ze)/(xe**2+ye**2+ze**2) # distance to the Earth-Sun line (not Earth+moon - Sun)
    dist_to_line =  np.sqrt((xe*q-xt)**2 + (ye*q-yt)**2 + (ze*q-zt)**2)
    return dist_to_line
    
def dist2Earth(b,tracerindex=-1):
    xt,yt,zt = b[tracerindex].x-b[1].x,b[tracerindex].y-b[1].y,b[tracerindex].z-b[1].z # tracer rel sun
    dist_to_point =  np.sqrt((xt)**2 + (yt)**2 + (zt)**2)
    return dist_to_point
        
def halotrack(drdv,b0,tracerindex=-1,trun=1*yr,nt=3000):
    b = b0.copy()
    re = pairsep(b[1],b[0])
    setorb(b,drdv,nb)
    nsub = 200
    percentrhill = re * (b[1].m/b[0].m/3)**(1/3) * 0.01 
    istarttrack = 0 # 0 to track very substep
    tsub, ntsub = trun/nsub,nt//nsub
    retval = 0.0
    d2E,d2ES = [],[]
    for i in range(nsub):
        steps(b,tsub,ntsub)
        if (i>=istarttrack):
            d2E.append(dist2Earth(b,tracerindex))
            d2ES.append(dist2lineEarthSun(b,tracerindex))
    d2E,d2ES = np.array(d2E),np.array(d2ES)
    retval = np.mean((d2ES/percentrhill)**4) # + np.std(d2E)**2
    #retval = np.sqrt(np.mean((d2ES/percentrhill)**2)) # + np.std(d2E)**2
    #print('xxx',b[1].x/AU,b[1].y/AU,b[1].vx,b[1].vy,'zzz',L1test(b,dr0))
    print('xxx  [%1.13e, %1.13e, %1.13e, %1.13e, %1.13e, %1.13e]  %g '%(drdv[0],drdv[1],drdv[2],drdv[3],drdv[4],drdv[5],retval))
    # print('xxx',b[tracerindex].x/AU,b[tracerindex].y/AU,b[tracerindex].vx,b[tracerindex].vy,'zzz %g'%retval)
    return retval

def puff(bref,ntracer, R = 1*km, puffspeed = 200, puffdisp = 10, ang = 130*np.pi/180., rdust = 1e-4, mdust = 3e-12, Q = 0, eta = 1/3):
    bt = np.array([bref]*(ntracer),dtype=b.dtype).view(np.recarray)
    bt.r = rdust; bt.m = mdust; bt.eta = eta; bt.Q = Q
    dr = np.array(unisphere(ntracer,radius=R)) if R>0 else 0
    bt.x, bt.y, bt.z = np.array([bt.x,bt.y,bt.z]) + dr
    erad = np.array([bref.x,bref.y,bref.z]*ntracer).reshape((ntracer,3)).T/dist(bref)
    #print(erad.shape)
    evel = np.array([bref.vx,bref.vy,bref.vz]*ntracer).reshape((ntracer,3)).T/speed(bref)
    ezed = np.array([0,0,1]*ntracer).T
    dv = puffspeed * (np.cos(ang)*evel - np.sin(ang)*erad) + np.random.normal(0,puffdisp,(3,ntracer))
    dv[2,:] *= 1
    bt.vx, bt.vy, bt.vz = np.array([bt.vx,bt.vy,bt.vz]) + dv
    bt.r,bt.m = rdust,mdust
    return bt

# ---- Main -----

np.random.seed(94123421)
    
fil = 'solarsystem.csv'
dfsolsys = pd.read_csv(fil)
    
sun = dfsolsys.iloc[0]
earth = dfsolsys.iloc[3]
moon = dfsolsys.iloc[4]
venus = dfsolsys.iloc[2]
mars = dfsolsys.iloc[5]
jupiter = dfsolsys.iloc[6]
saturn = dfsolsys.iloc[7]
    
nb = 3 # number of massive nbodies
nb = 7 # include nearby, big planets
b = np.zeros(nb+1,dtype=bodyt).view(np.recarray)
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

dr0fid = -pairsep(b[1],b[0])*(b.m[1]/3/b.m[0])**(1./3)
dr0 = dr0fid
# for solar eclipse of 2017

drdv = np.array([9.87060899e+07, 9.94944750e+07, 1.01205252e+06, 1.01626852e+01,
       1.01174070e+01, 9.99609883e+01]) # cute solution optimized on square of dist to Earth-Sun Line....

drdv = np.array([9.7819160134069e+07, 1.0086054786768e+08, 1.0406685899666e+06, 1.0386484846233e+01, 9.8161719507052e+00, 9.7691723511862e+01]) # best quad, still "cute"...grr i thought i had a better one...

drdv = np.array([9.7672356395251e+07, 1.0145546625866e+08, 1.0474139184824e+05, 1.0271868687687e+01, 9.7177468883876e+00, 9.7381395186548e+00]) # best quad

setorb(b,drdv,tracerindex=ti)
bstart = b.copy()

print('# sun-tracer dist/AU:',pairsep(b[ti],b[0])/AU)
print('# earth-tracer dist/AU:',pairsep(b[ti],b[1])/AU)
print('# moon-tracer dist/AU:',pairsep(b[ti],b[2])/AU)
print('# earth-sun dist/AU:',pairsep(b[0],b[1])/AU)
print('# moon-sun dist/AU:',pairsep(b[0],b[2])/AU)
        
make_movie = True

# defaults 
ntracer = 1
tmax = 1.0*yr
tmax = 3*(28*24*3600)
ddt = 1*yr/4000
npl = 300 # numper of plot calls

nindex = nglass; rhop = rhoglass
rp = 2e-4 # rad vol equiv
ff = 0.1 # filling frac
ef = np.real(nindex)**2
fff = lambda x: (1-ff)*(1-x)/(1+2*x)+ff*(ef-x)/(ef+2*x)
neff = np.sqrt(root_scalar(fff,x0=1,x1=ef,rtol=1e-8).root)
# for this radius, the eff vall radius, find the physics radius....
rfluff = rp/ff**(1/3)
qext, qsca, g = do_mie(rfluff,neff) # could use area-equiv...
qpr = qext-g*qsca
mdust = 4*np.pi/3*rp**3*rhop
Lsolar = 3.828e33
c = 2.99792458e10
des = AU
# beta = Lsolar*qpr*rfluff**2/(4*G*c*mdust*sun.m)
eta = 1/3
b[ti:].r = rfluff
b[ti:].m = mdust
b[ti:].Q = qpr
b[ti:].eta = 1/3
b[ti].Q = 0.0

ntracer = 40000
create_cloud = 'at the beginning'
create_cloud = 'stream'
#create_cloud = 'burst'

ngrainsPerTracer = Mdust/mdust/ntracer

if 'beginning' in create_cloud:
    ntracer = 5000
    ngrainsPerTracer = Mdust/mdust/ntracer
    ang = puffspeed = 0
    bp = puff(b[ti], ntracer, R=0, puffspeed=puffspeed, ang=ang, puffdisp=0, rdust=rfluff, mdust=mdust, Q=qpr, eta=eta)
    #bp.r += np.random.normal(0,0.00001*b[ti].r,ntracer//npl)
    b = np.append(b,bp).view(np.recarray)
    tdeliv = 0.0
elif 'stream' in create_cloud:
    tdeliv = 3*28*24*3600
elif 'burst' in create_cloud:
    ntracer = 5000
    tdeliv = 0.0

    
bstart = b.copy()
Estart = energy(bstart)

dt = tmax/npl
ntsub = int(dt/ddt)+1
tstart = time.time()  
phlis = np.zeros(3)

L = 1.0*np.pi/180
na = 600
ell = np.linspace(-0.5*L,0.5*L,na)
dell = ell[1]-ell[0]
elledge = np.linspace(-0.5*(L+dell),0.5*(L+dell),na+1)
X,Y = np.meshgrid(ell,ell)

if True:
    cmap = 'gnuplot2'
    csky,csun = -0.75,0.96 # color vals working off of whatever cmap
    csky,csun = 0.5,0.96 # color vals working off of whatever cmap
    cmap = 'gist_gray'
    fmovcstr = 'tmp'+str(os.getpid())+'*.png'
    # print(glob.glob(fmovcstr))
    if len(glob.glob(fmovcstr))>0:
        print('# removing files',fmovcstr)
        os.system('rm '+fmovcstr)

cumattn = 0.0
radpress = True
phlast = -99
burst_done = burst_primed = False
for i in range(npl):
    #steps(b,dt,ntsub)
    steps(b,dt,ntsub)
    re = pairsep(b[1],b[0])
    tnow = dt*(i+1)
    th,ph = skycoords(b,ti) # lat, lon
    ang = np.sqrt(ph**2*np.cos(th)**2 + th**2)
    msk = ang < 8*Rsun/des
    #d1 = get_d1(Msolar,Mearth,des,0*betaf,G)
    d1 = pairsep(b[1],b[ti:])
    #rshadow = Rsun*d1/(des-d1)
    # return fs*rp**2/d1**2*asemi**2/Rsun**2
    # attn = fshadow * qext * (rfluff/d1)**2 / (Rsun/des)**2 * Mdust/mdust
    attn = qext * attengeo(b[ti:].r[msk],ang[msk],d1=d1[msk],asemi=AU) * ngrainsPerTracer
    cumattn += np.sum(attn) * dt
    dphi = ph[0]-phlast
    phlast = ph[0]
    print('%g yr, attn: %g,  cumulative attn: %g'%(tnow/yr,np.sum(attn),cumattn/day), ' ph,dph:',ph[0],dphi)
    if 'stream' in create_cloud:
        if tnow < tdeliv:
            # 90 is to the right, -90 to the left
            ang = -np.arcsin(min(1,max(-1,ph[0]/(Rsun/AU))))
            puffspeed = 300*np.abs(np.sin(ang))
            puffspeed = 1000; ang = 65*np.pi/180  # ang == 0 is for dir of Earth motion...cum attn = 0.08, 40 deg 0.06, 50->0.079, 65 is best
            #puffspeed = 0; ang = 0*np.pi/180  #  baeline
            ntthis = int(ntracer/(tmax/dt)*(tmax/tdeliv))
            print('ntthiss',ntthis)
            bp = puff(b[ti], ntthis, R=0, puffspeed=puffspeed, ang=ang, puffdisp=30, rdust=rfluff, mdust=mdust, Q=qpr, eta=eta)
            #bp.r += np.random.normal(0,0.00001*b[ti].r,ntracer//npl)
            b = np.append(b,bp).view(np.recarray)
    elif 'burst' in create_cloud:
        # print(burst_primed,burst_done)
        if not burst_primed:
            if dphi < 0 and ph[0] < 0:
                burst_primed = True
        if dphi > 0 and burst_done == False and burst_primed:
            #print('cool',ntracer,dphi); quit()
            puffspeed = 1500; ang = 65*np.pi/180  # ang == 0 is
            ntthis = ntracer
            bp = puff(b[ti], ntthis, R=0, puffspeed=puffspeed, ang=ang, puffdisp=30, rdust=rfluff, mdust=mdust, Q=qpr, eta=eta)
            b = np.append(b,bp).view(np.recarray)
            burst_done = True
    Z = 0*X + csky
    msk = X**2 + Y**2 < (Rsun/re)**2; Z[msk] = csun
    #th,ph = skycoords(b,ti)
    ZZ,_,_ = np.histogram2d(th,ph,bins=[elledge,elledge],)
    Z = np.where(ZZ>0,-1,Z)
    deg = np.pi/180
    pl.imshow(Z,extent=(elledge[0]/deg,elledge[-1]/deg,elledge[0]/deg,elledge[-1]/deg),cmap=cmap,vmin=-1,vmax=1)
    pl.xlabel(r'$\Delta\phi$ (deg)',size=14)
    pl.ylabel(r'$\Delta\theta$ (deg)',size=14)
    #pl.scatter(phti*180/np.pi,thti*180/np.pi,c='#ffffff',s=10)
    pl.figtext(0.6,0.2,'%0.3f'%(tnow/yr),color='#660066')
    out = fmovcstr.replace('*','%04d')%(i)
    print('# saving',out)
    noccult = np.sum(ZZ[msk])
    #print('# this time, n tot, n occult:',tnow/yr,len(b),noccult)
    pl.savefig(out)
    pl.clf()
    if cumattn > 0.0 and tnow > tdeliv:
        if np.sum(attn) == 0.0:
            break
    
print('cumulative attn:',cumattn/day)
print("# looptime %s s"%(time.time()-tstart))

if make_movie:
    os.system('convert '+fmovcstr+' ~/public_html/tmp.gif')
    print('# removing files',fmovcstr)
    os.system('rm '+fmovcstr)

# the figure itself is one frame in this output. 
