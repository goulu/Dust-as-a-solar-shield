#
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time
import pylab as pl
import os
import glob
from nbody import *
import myfig
import attnmie
from constants import *
from myparms import *
import argparse
import sys

def estL1pt(b,fac=1):
    x1,y1,z1 = (b[0].x-b[1].x),(b[0].y-b[1].y),(b[0].z-b[1].z) # Sun in Earth frame
    r1 = fac*0.01*AU/np.sqrt(x1**2+y1**2+z1**2); x1*=r1; y1*=r1; z1*=r1; # L1 in Earth frome
    return x1+b[1].x, y1+b[1].y, z1+b[1].z,
    
def setlaunch(b,dvsp,tracerindex=-1): # dvang are angle corrections to my guess
    veject,dth,dph = dvsp
    xm,ym,zm = b[2].x,b[2].y,b[2].z
    vxm,vym,vzm = b[2].vx,b[2].vy,b[2].vz
    x,y,z = xm,ym,zm+b[2].r
    # L1
    x1,y1,z1 = estL1pt(b)
    # L1 rel to moon launch point
    dx = x1-x; dy = y1-y; dz = z1-z
    rr = np.sqrt(dx**2+dy**2+dz**2); dx /= rr; dy /= rr; dz /= rr
    th,ph = np.arccos(dz), np.arctan2(dy,dx)
    th += dth; ph += dph
    vx,vy,vz = vxm+veject*np.sin(th)*np.cos(ph), vym+veject*np.sin(th)*np.sin(ph), vzm+veject*np.cos(th)
    b[tracerindex:].x,b[tracerindex:].y,b[tracerindex:].z = x,y,z
    b[tracerindex:].vx,b[tracerindex:].vy,b[tracerindex:].vz = vx,vy,vz    
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
    
def puff(bref,ntracer, R = 1*km, puffspeed = 0, puffdisp = 60, ang = 30*np.pi/180., rp = 1e-4, mp = 2.7*4*np.pi/3*1e-12):
    bt = np.array([bref]*(ntracer),dtype=b.dtype).view(np.recarray)
    dr = np.array(unisphere(ntracer,radius=R)) if R>0 else 0
    bt.x, bt.y, bt.z = np.array([bt.x,bt.y,bt.z]) + dr
    erad = np.array([bref.x,bref.y,bref.z]*ntracer).reshape((ntracer,3)).T/dist(bref)
    #print(erad.shape)
    evel = np.array([bref.vx,bref.vy,bref.vz]*ntracer).reshape((ntracer,3)).T/speed(bref)
    ezed = np.array([0,0,1]*ntracer).T
    dv = puffspeed * (np.cos(ang)*evel - np.sin(ang)*erad) + np.random.normal(0,puffdisp,(3,ntracer))
    dv[2,:] *= 1
    bt.vx, bt.vy, bt.vz = np.array([bt.vx,bt.vy,bt.vz]) + dv
    bt.r, bt.m = rp, mp
    return bt

def jettrack(dvsp,b0,tracerindex,trun,nt,tlaunch,ntlaunch): # dvsp = veject, del_th, del_phi
    b = b0.copy()
    re = pairsep(b[1],b[0])
    setlaunch(b,dvsp,nb)
    steps(b,tlaunch,ntlaunch)
    nsub = 800
    percentrhill = re * (b[1].m/b[0].m/3)**(1/3) * 0.01 
    istarttrack = 0 # 0 to track very substep
    tsub, ntsub = trun/nsub,nt//nsub
    retval = 0.0
    accval,accvalalt,tmp = [],[],[]
    tx = 0
    attn = 0.0
    for i in range(nsub):
        steps(b,tsub,ntsub)
        x1,y1,z1 = estL1pt(b,0.75)
        x,y,z = b[tracerindex].x,b[tracerindex].y,b[tracerindex].z
        d1 = np.sqrt((x-x1)**2+(y-y1)**2+(z-z1)**2)
        lat, lon = attnmie.skycoords(b,tracerindex=ti)
        ang = np.sqrt(lat**2+lon**2)[0]
        de1 = pairsep(b[tracerindex],b[1])
        if (de1>10*Rearth):
            attn = attnmie.attngeo(10*1e5,ang,d1=de1,asemi=AU)
        else:
            attn = 0.0
        accval.append(attn*tsub/ntsub)
        accvalalt.append(d1)
        tmp.append(attn)
    attn = np.sum(accval)
    if (attn>0):
        retval = -attn
    else:
        retval = np.min(accvalalt)/AU
    print('xxx  np.array([%1.13e, %1.13e, %1.13e])   tx %g yr, d2E final %g AU]  %g '%(dvsp[0],dvsp[1],dvsp[2],tx/yr,pairsep(b[tracerindex],b[1])/AU,retval))
    print('xxxxxx',np.min(tmp),np.max(tmp))
    return retval


def dumpmxv(nm,b):
    for nmi,bi in zip(nm,b):
        print('%s & %13.10e & %13.10f & %13.10f & %13.10f & %13.10f & %13.10f & %13.10f \\\\' % \
              (nmi,bi.m/1e3,bi.x/AU,bi.y/AU,bi.z/AU,bi.vx/km,bi.vy/km,bi.vz/km))

# ---- Main -----


seed = 94123421
do_fit = False
#if len(sys.argv)>1:    
#    if 'fit' in sys.argv[1]:
#        do_fit = True
#        if len(sys.argv)>2:
#            seed = int(sys.argv[2])
#    else:
#        print('usage:',sys.argv[0],'[fit [seed]]')    

parser = argparse.ArgumentParser(description='lunar dust crosses the face of the Sun.')
parser.add_argument('--fit', action='store_true') # const=True, help='triggers fitting algorithm')
parser.add_argument('--seed', metavar='SEED', type=int, nargs=1, default=seed, help='RNG seed, overwrites default')
args = parser.parse_args()
seed = args.seed
do_fit = args.fit

np.random.seed(seed)

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

nm = ['Sun','Earth','Moon','Venus','Mars','Jupiter','Saturn']


ti = tracerindex = nb
bm = b[:nb] # all massive....
comr = np.sum(np.array([bm.m*bm.x,bm.m*bm.y,bm.m*bm.z])/np.sum(bm.m),axis=1)
comv = np.sum(np.array([bm.m*bm.vx,bm.m*bm.vy,bm.m*bm.vz])/np.sum(bm.m),axis=1)
b.x -= comr[0]; b.y -= comr[1]; b.z -= comr[2];
b.vx -= comv[0]; b.vy -= comv[1]; b.vz -= comv[2];

# scattering 
rhop = 2.7
rp = 1e-4
rp = 2e-5
mp = 4*np.pi/3*rhop*rp**3
nolivine = 1.67 - 1e-3j; rhoolivine = 2.7 # D. Fabian, Th. Henning, C. Jäger, H. Mutschke, J. Dorschner, O. Werhan, Astron. Astrophys. 378, 228 (2001).
qext,qsca,g = attnmie.do_mie(rp,nolivine)

b[ti].r = rp
b[ti].m = mp
b[ti].Q = qext-g*qsca
b[ti].eta = 1/3
print('Q,Q_ext:',qext-g*qsca,qext)

veject = 5e5

#dv = np.array([8.2498313870942e+05, 8.6208553525811e+05, 5.1274549795271e+04])
#dv = np.array([8.2498313870942e+03, 8.6208553525811e+03, 5.1274549795271e+03])

# solutions for nominal start date...eject speed veject +  angle corrections to launch dir


if rp == 1e-4:
    dvsp = np.array([veject,-9.5125912654206e-02, -1.5618348247851e-01])  # use this fast ejection if size distribution is an issue.
    dvsp = np.array([2.9001068472507e+05, -3.2550141296260e-01, -4.6695628017039e-01]) # slower eject, better for monodispere....
    dvsp = np.array([2.8462308656802e+05, -3.4076754224856e-01, -5.0785273893168e-01]) # fine tuning on the above
    dvsp = np.array([2.8357100411988e+05, -3.4293781824274e-01, -5.1795151076270e-01]) # cool one of 0.04 years. cum attn of 7 days....
    dvsp = np.array([2.8458305643931e+05, -3.3930196520702e-01, -5.0856864048820e-01]) 
    dvsp = np.array([2.8521533932270e+05, -3.3728893561577e-01, -5.0581976505945e-01])
    
if rp == 2e-5:
    dvsp = np.array([2.8357100411988e+05, -3.4293781824274e-01, -5.1795151076270e-01]) # cool one of 0.04 years. cum attn of 7 days....
    dvsp = np.array([5.6401199226275e+05, -1.0661588793179e-01, -1.8801533775644e-01])
    dvsp = np.array([4.7231562253812e+05, -1.0227176974548e-01, -1.8175993075390e-01]) 

# solutions for other start dates:

tfit = 1.2*0.01*AU/veject
tfit = 0.040*yr
nt = 4800
tlaunch = 4000.0 # integrate at fine res for launch from lunar surface...in units of seconds...
ntlaunch = 400

tadd = 21*24.*3600 # add a week....
tadd = 0.0
nt=2400

if tadd == 21*24.*3600:
    nt=1200
    steps(b,tadd,1000)
    dvsp = np.array([2.3636887702152e+05, -6.7420823231116e-01, 3.4859231567473e-01])
    
setlaunch(b,dvsp,tracerindex=ti)
bstart = b.copy()

print(tfit/yr)
    
if do_fit:
    mytol = 1e-6
    mytest = jettrack(dvsp,b,ti,tfit,nt,tlaunch,ntlaunch)
    print('# my test:',mytest)
    if True: #mytest > mytol:
        smplx = [dvsp]
        for j in range(1,len(dvsp)+1):
            u = np.random.uniform(0.9,1.1,len(dvsp))
            u = np.random.uniform(0.95,1.05,len(dvsp))
            u = np.random.uniform(0.99,1.01,len(dvsp))
            smplx.append(u*dvsp)
        smplx = np.array(smplx)
        maxiter = 400
        res = minimize(jettrack,dvsp,args=(b,ti,tfit,nt,tlaunch,ntlaunch),method='Nelder-Mead',
                       options={'maxiter':maxiter, 'fatol':0.00000001,'initial_simplex':smplx})
        print(res)
        dv = res.x
        setlaunch(bstart,dvsp,nb)
    else:
        print('L1 test metric',mytest)

ntracer = 1

make_movie = False
mode = 'movie'

if 'movie' in mode:
    make_movie = True

npl = 400 # numper of plot calls

b = bstart.copy()
if make_movie:
    ntracer = 10000
    if ntracer>1:
        if rp > 5e-5:
            bp = puff(b[ti],ntracer, R=0, puffdisp=10, rp=b[ti].r, mp = b[ti].m)
            bp.r += np.random.normal(0,0.05*b[ti].r,ntracer)
        else:
            bp = puff(b[ti],ntracer, R=0, puffdisp=20, rp=b[ti].r, mp = b[ti].m)
            bp.r += np.random.normal(0,0.0*b[ti].r,ntracer)
            
        b = np.append(b,bp).view(np.recarray)

bstart = b.copy()
Estart = energy(bstart)

tmax = tfit
if rp == 1e-4:
    tmax *= 1.1


ddt = tmax/nt
dt = tmax/npl
ntsub = int(dt/ddt)
print('ntsubsteps:',ntsub)
tstart = time.time()  
phlis = np.zeros(3)

L = 2.0*np.pi/180
na = 600
ell = np.linspace(-0.5*L,0.5*L,na)
dell = ell[1]-ell[0]
elledge = np.linspace(-0.5*(L+dell),0.5*(L+dell),na+1)
X,Y = np.meshgrid(ell,ell)

if make_movie:
    doframe = 0 # 1 or 2
    framectr = 0
    cmap = 'gnuplot2'
    csky,csun = -0.75,0.96 # color vals working off of whatever cmap
    csky,csun = 0.5,0.96 # color vals working off of whatever cmap
    cmap = 'gist_gray'
    fmovcstr = 'tmp'+str(os.getpid())+'*.png'
    # print(glob.glob(fmovcstr))
    if len(glob.glob(fmovcstr))>0:
        print('# removing files',fmovcstr)
        os.system('rm '+fmovcstr)

moox,mooy = [],[]

inframe = np.array([0,0,0])

nm = nm + ['tracer']*(len(b)-7)

radpress = True

tlatlon = []
cumattn = 0.0
cumtime = 0.0
steps(b,tlaunch,ntlaunch)
for i in range(npl):
    #steps(b,dt,ntsub)
    steps(b,dt,ntsub)
    re = pairsep(b[1],b[0])
    d1 = pairsep(b[ti:],b[1])
    tnow = dt*(i+1)
    lat,lon = attnmie.skycoords(b,ti)
    tlatlon.append([tnow,lat[0],lon[0]])
    ang = np.sqrt(lat**2+(lon*np.cos(lat))**2)
    #print('dist from earth, center of solar disk:',d1[0]/AU,'AU,',ang[0]*180/np.pi,'degrees')
    de1 = pairsep(b[ti:],b[1])
    attn = qext*attnmie.attngeo(rp,ang,d1=de1,asemi=AU)*Mdust/(mp*ntracer)
    cumtime += dt * attnmie.attngeo(rp,ang[0],d1=de1[0],asemi=AU)/attnmie.attngeo(rp,0,d1=de1[0],asemi=AU)
    cumattn += dt*np.sum(attn)
    print('# t=%8.5f yr, %1.4g attn-day %1.4g days at %1.4f au'%(tnow/yr,cumattn/day,cumtime/day,d1[0]/AU))
    if make_movie:
        Z = 0*X + csky
        msk = X**2 + Y**2 < (Rsun/re)**2; Z[msk] = csun
        ZZ,_,_ = np.histogram2d(lat,lon,bins=[elledge,elledge],)
        noccult = np.sum(ZZ[msk])
        ninframe = np.sum(ZZ)
        inframe = np.append([ninframe],inframe[0:-1])
        #print('inframe',inframe)
        if (doframe == 0):
            if np.sum(inframe)>len(inframe):
                doframe = 1
        elif doframe == 1:
            if np.sum(inframe)==0:
                doframe = 2
                break;
        #print('ZZZZZ',doframe)
        if doframe == 1:
            Z = np.where(ZZ>0,-1,Z)
            pl.imshow(Z,extent=(elledge[0],elledge[-1],elledge[0],elledge[-1]),cmap=cmap,vmin=-1,vmax=1)
            #pl.scatter(phti*180/np.pi,thti*180/np.pi,c='#ffffff',s=10)
            pl.figtext(0.6,0.2,'%6.3f d'%(tnow/(24*3600.)),color='#660066')
            out = fmovcstr.replace('*','%04d')%(framectr)
            framectr += 1
            #print('# saving',out)
            pl.savefig(out)
            pl.clf()

if make_movie:
    os.system('convert '+fmovcstr+' ~/public_html/tmp.gif')
    print('# removing files',fmovcstr)
    os.system('rm '+fmovcstr)

if True:
    re = pairsep(b[1],b[0])
    angsun = (b[0].r/re)*180/np.pi
    aa = np.linspace(0,2*np.pi,100)
    xx,yy = angsun*np.cos(aa),angsun*np.sin(aa)
    pl.xlim(-2*angsun,2*angsun)
    pl.ylim(-2*angsun,2*angsun)
    pl.gca().set_aspect('equal')
    pl.plot(xx,yy,'-',c='#999999',zorder=0)
    pl.xlabel(r'$\Delta\phi$ (deg)',size=14)
    pl.ylabel(r'$\Delta\theta$ (deg)',size=14)
    out = 'moonjetjpltraj.pdf'
    ts,th,ph = np.array(tlatlon).T
    hint,pint = interp1d(ts,th),interp1d(ts,ph)
    tsi = np.linspace(ts[0],ts[-1],len(ts)*10)
    thi, phi = hint(tsi), pint(tsi)
    pl.scatter(phi*180/np.pi,thi*180/np.pi,s=0.25,c=tsi,cmap='binary',zorder=1)
    pl.savefig(out)
    os.system('convert '+out+' ~/public_html/tmp.jpg')
        
