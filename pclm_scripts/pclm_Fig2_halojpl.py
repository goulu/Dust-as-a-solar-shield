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
import scatter as sc
import myfig

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

def norm2(x,y,z): return x**2+y**2+z**2

def skyang(b1,b2,bobs): # ang between b1 & b2 from b"obs"'s perspective
    dx1,dy1.dz1=(b1.x-bobs.x),(b1.y-bobs.y),(b1.z-bobs.z)
    dx2,dy2.dz2=(b2.x-bobs.x),(b2.y-bobs.y),(b2.z-bobs.z)
    cosang = (dx1*dx2 + dy1*dy2 + dz1*dz2)/np.sqrt(norm2(dx1,dy1,dz1)*norm2(dx2,dy2,dz2))
    return np.arccos(min(max(cosang,-1),1))

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

def skycoords(b,tracerindex=3,loc='C'):
    # this is overkill. create a basis, ux,uy,uz ux is earth-to-sun, uy is toward earth velocity, z is "up"
    ex,ey,ez = b[1].x,b[1].y,b[1].z # earth pos (loc=='C' for center)
    if loc in 'NSEW':
        if loc == 'N': ez += b[1].r
        elif loc == 'S': ez -= b[1].r
        else:
            sgn = 1 if loc=='W' else -1 
            evx,evy = b[1].vx,b[1].vy; fac = b[1].r/np.sqrt(evx**2+evy**2)
            ex += sgn*evx*fac; ey += sgn*evy*fac; 
    dx,dy,dz = (b[tracerindex:].x-ex),(b[tracerindex:].y-ey),(b[tracerindex:].z-ez)
    rp = np.sqrt(dx**2 + dy**2 + dz**2) 
    upx,upy,upz = dx/rp,dy/rp,dz/rp
    dxs,dys,dzs = (b[0].x-ex),(b[0].y-ey),(b[0].z-ez) # sun pos in earth frame
    re = np.sqrt(dxs**2+dys**2+dzs**2)
    uxx,uxy,uxz = dxs/re,dys/re,dzs/re # unit vec, ux is x-like, dir of sun 
    uzx,uzy,uzz = (uxy*b[1].vz-uxz*b[1].vy),(uxz*b[1].vx-uxx*b[1].vz),(uxx*b[1].vy-uxy*b[1].vx)
    unorm = np.sqrt(uzx**2+uzy**2+uzz**2); uzx,uzy,uzz = uzx/unorm,uzy/unorm,uzz/unorm # uz is z-like, perp to orb plane 
    uyx,uyy,uyz = (uzy*uxz-uzz*uxy),(uzz*uxx-uzx*uxz),(uzx*uxy-uzy*uxx) 
    unorm = np.sqrt(uyx**2+uyy**2+uyz**2); uyx,uyy,uyz = uyx/unorm,uyy/unorm,uyz/unorm # uy is y-like, 
    th = np.pi/2 - np.arccos(upx*uzx + upy*uzy + upz*uzz) # pi/2 - cosine angle, angle in z dir  
    ph = np.pi/2 - np.arccos(upx*uyx + upy*uyy + upz*uyz) #
    # check if not near sun?
    cossp = upx*uxx + upy*uxy + upz*uxz
    th = np.where(cossp<0,np.pi-th,th)
    ph = np.where(cossp<0,np.pi-ph,ph)
    return th,ph

def earthsunframe(b,tracerindex=3):
    # this is overkill. create a basis, ux,uy,uz ux is earth-to-sun, uy is toward earth velocity, z is "up"
    ex,ey,ez = b[1].x,b[1].y,b[1].z # earth pos (loc=='C' for center)
    dx,dy,dz = (b[tracerindex:].x-ex),(b[tracerindex:].y-ey),(b[tracerindex:].z-ez)
    dxs,dys,dzs = (b[0].x-ex),(b[0].y-ey),(b[0].z-ez) # sun pos in earth frame
    re = np.sqrt(dxs**2+dys**2+dzs**2); uxx,uxy,uxz = dxs/re,dys/re,dzs/re # ux, x-like, dir of sun
    uzx,uzy,uzz = (uxy*b[1].vz-uxz*b[1].vy),(uxz*b[1].vx-uxx*b[1].vz),(uxx*b[1].vy-uxy*b[1].vx)
    unorm = np.sqrt(uzx**2+uzy**2+uzz**2); uzx,uzy,uzz = uzx/unorm,uzy/unorm,uzz/unorm # uz, perp to orb plane 
    uyx,uyy,uyz = (uzy*uxz-uzz*uxy),(uzz*uxx-uzx*uxz),(uzx*uxy-uzy*uxx) 
    unorm = np.sqrt(uyx**2+uyy**2+uyz**2); uyx,uyy,uyz = uyx/unorm,uyy/unorm,uyz/unorm # uy is y-like
    x,y,z = uxx*dx+uxy*dy+uxz*dz, uyx*dx+uyy*dy+uyz*dz, uzx*dx+uzy*dy+uzz*dz
    return x,y,z

def cossep(bA,bB,b0):
    xA,yA,zA = (bA.x-b0.x),(bA.y-b0.y),(bA.z-b0.z),
    xB,yB,zB = (bB.x-b0.x),(bB.y-b0.y),(bB.z-b0.z),
    cosp = (xA*xB+yA*yB+zA*zB)/np.sqrt((xA**2+yA**2+zA**2)*(xB**2+yB**2+zB**2))
    return cosp

def overlap(rbig,rsmall,offset): #fraction of big circle that is partially overlapped by little circle
    R,r,d = rbig,rsmall,offset # wolfram mathworld
    drRfac,dRrfac = (d**2+r**2-R**2)/(2*d*r), (d**2+R**2-r**2)/(2*d*R)
    drRfac = np.where(np.abs(drRfac)>1,drRfac/np.abs(drRfac),drRfac)
    dRrfac = np.where(np.abs(dRrfac)>1,dRrfac/np.abs(dRrfac),dRrfac)
    drRyuk = (-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)
    drRyuk = np.where(drRyuk<0,0,drRyuk)
    return (r**2*np.arccos(drRfac)+R**2*np.arccos(dRrfac)-0.5*np.sqrt(drRyuk))/(np.pi*R**2)

def radredux(b):
    des,det,dst = pairsep(b[0],b[1]),pairsep(b,b[1]),pairsep(b,b[0])
    f = np.zeros(len(b))
    msk = (b.m<1e11)&(des>0)&(det>0)&(det<des) # should be all tracers
    re,rs = b[1].r,b[0].r
    rshadow = rs*det[msk]/(des-det[msk])
    rshctr = des*np.sin(np.arccos(cossep(b[msk],b[1],b[0]))) # loc of tracer shadow in plane of earth disk
    f[msk] = np.where((rshadow>re)&(rshctr<(rshadow-re)),re**2/rshadow**2,f[msk])
    f[msk] = np.where((rshadow>re)&(rshctr>(rshadow-re)),overlap(rshadow,re,rshctr),f[msk])
    f[msk] = np.where((rshadow<re)&(rshctr<(re-rshadow)),rshadow**2/re**2,f[msk])
    f[msk] = np.where((rshadow<re)&(rshctr>(re-rshadow)),overlap(re,rshadow,rshctr),f[msk])
    return f

def radreduxMC(b):
    N = 5000; xg = np.linspace(-1,1,N)
    dX,dY = np.meshgrid(xg,xg); msk = dX**2+dY**2 < 1; dX,dY = dX[msk].flatten(), dY[msk].flatten()
    dA = 4/N**2+0*dX
    re,rs = b[1].r,b[0].r
    f = np.zeros(len(b))
    for i in range(len(b)):
        bi = b[i]
        if (bi.m>1e11): continue
        des,det,dst = pairsep(b[0],b[1]),pairsep(bi,b[1]),pairsep(bi,b[0])
        rshadow = rs*det/(des-det)
        rshctr = des*np.sin(np.arccos(cossep(bi,b[1],b[0])))
        xi,yi,ai = dX*rshadow+rshctr,dY*rshadow,dA*rshadow**2
        f[i] = (np.sum(ai[xi**2+yi**2<re**2])/(np.pi*rshadow**2))
    return f

def ovrlap(d,re,rs): # overlapping area of circles rs > re displaced from centers by distance d.
    return re**2*np.arccos((d**2+re**2-rs**2)/(2*d*re))+rs**2*np.arccos((d**2+rs**2-re**2)/(2*d*rs))\
        -0.5*np.sqrt((d+re+rs)*(d-re+rs)*(d+re-rs)*(-d+re+rs))

def attengeo(rp,ang,d1=0.01*AU,asemi=AU):
    global Rearth, Rsun
    rs = Rsun*d1/(asemi-d1)
    re = Rearth
    d = ang*d1*asemi/(asemi-d1)
    if np.isscalar(ang):
        if d<=rs-re:
            fs = re**2/rs**2
        elif (d<rs+re)&(d>rs-re):
            #print('xxx',d/re,rs/re)
            fs = ovrlap(d,re,rs)/(np.pi*rs**2)
        else:
            fs = 0.0
    else:
        fs = np.where(d<=rs-re,re**2/rs**2,0)
        msk = (d<rs+re)&(d>rs-re);
        fs[msk] = ovrlap(d[msk],re,rs)/(np.pi*rs**2)
    return fs*rp**2/d1**2*asemi**2/Rsun**2

def britensca(rp,th,g,nu,d1=0.01*AU,asemi=AU):
    global Rearth, Rsun
    delOm = np.pi*(Rearth/d1)**2
    return rp**2/d1**2*sc.phase_function_D03(th,g,nu)*delOm*asemi**2/Rsun**2

def puff(bref,ntracer, R = 1*km, puffspeed = 0, puffdisp = 60, ang = 30*np.pi/180., rp = 1e-4, rhop = 2.7):
    mp = 4*np.pi*rhop*rp**3
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
    bt.r,bt.m = rp,mp
    return bt

def mycloud(bref,ntracer,rS = 100e2, vs = 0, rp = 1e-4, rhop = 2.7):
    mp = 4*np.pi*rhop*rp**3
    bt = np.array([bref]*(ntracer),dtype=b.dtype).view(np.recarray)
    dr = np.array(unisphere(ntracer,radius=rS))
    bt.x, bt.y, bt.z = np.array([bt.x,bt.y,bt.z]) + dr
    dv = dr*vs/rS
    bt.vx, bt.vy, bt.vz = np.array([bt.vx,bt.vy,bt.vz]) + dv
    bt.r,bt.m = rp,mp
    return bt

def dumpmxv(nm,b):
    for nmi,bi in zip(nm,b):
        print('%s & %13.10e & %13.10f & %13.10f & %13.10f & %13.10f & %13.10f & %13.10f \\\\' % \
              (nmi,bi.m/1e3,bi.x/AU,bi.y/AU,bi.z/AU,bi.vx/km,bi.vy/km,bi.vz/km))



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

nm = ['Sun','Earth','Moon','Venus','Mars','Jupiter','Saturn']

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

do_fit = False
# do_fit = True

if do_fit:
    tfit = 1.0*yr
    mytol = 4e-3
    mytol = 0.004
    mytest = halotrack(drdv,b,ti,tfit,4000)
    print('# my test:',mytest)
    print('# L1 dist: ',pairsep(b[-1],b[1])/pairsep(b[-1],b[0]))
    if mytest > mytol:
        smplx = [drdv]
        for j in range(1,7):
            u = np.random.uniform(0.9,1.1,len(drdv))
            u = np.random.uniform(0.99,1.01,len(drdv))
            smplx.append(u*drdv)
        smplx = np.array(smplx)
        maxiter = 400
        res = minimize(halotrack,drdv,args=(b,ti,1.0*tfit,4000),method='Nelder-Mead',
                       options={'maxiter':maxiter, 'fatol':0.1,'initial_simplex':smplx})
        print(res)
        drdv = res.x
        setorb(b,drdv,nb)
        bstart = b.copy()
    else:
        print('L1 test metric',mytest)

print('# sun-tracer dist/AU:',pairsep(b[ti],b[0])/AU)
print('# earth-tracer dist/AU:',pairsep(b[ti],b[1])/AU)
print('# moon-tracer dist/AU:',pairsep(b[ti],b[2])/AU)
print('# earth-sun dist/AU:',pairsep(b[0],b[1])/AU)
print('# moon-sun dist/AU:',pairsep(b[0],b[2])/AU)
        
ntracer = 1000
gdraine,nudraine = 0.0,0.5

hh = np.linspace(0,np.pi,10000); ddhh = hh[1]-hh[0]
gg = np.sum(sc.phase_function_F19(hh)*np.sin(hh)*np.cos(hh)*2*np.pi*ddhh)


#Rshadow = b[0].r*np.abs(dr0fid)/(pairsep(b[0],b[1])-np.abs(dr0fid))
#print(Rshadow,Rshadow/b[1].r)

make_movie = False
plot_traj = False
do_it = False
plot_beta = False

mode = 'traj'
#mode = 'beta'
#mode = 'movie'

if 'movie' in mode:
    make_movie = True
if 'traj' in mode:
    plot_traj = True
if 'beta' in mode:
    plot_beta = True
if 'do_it' in mode:
    do_it = True

# defaults for plot_traj
ntracer = 1
tmax = 1.0*yr
#ddt = 1*yr/16000
ddt = 1*yr/8000

npl = 400 # numper of plot calls
rhop = 2.7
rp = 1e-4

if make_movie:
    rp = 1e3*1e-1
    ntracer = 20000
    create_cloud = True

if plot_beta:
    print('overiding parms to plot orbits for different beta...')
    ntracer = 6
    tmax = 28*24*3600
    b = np.append(b,mycloud(b[ti],ntracer-1)).view(np.recarray)
    b[ti+1:].r = np.logspace(-3,-1,ntracer-1)
    b[ti+1:].r = np.array([10e-4,32e-4,100e-4,320e-4,0.1])
    b[ti+1:].m = 4*np.pi/3*rhop*b[ti+1:].r**3
    b[ti+1:].Q = 1-gdraine
    b[ti+1:].eta = 1/3.
    tracerdat = np.zeros((npl,ntracer*3))

                         
if do_it:
    print('overiding parms to do do_it...')
    ntracer = 10
    tmax = 0.1*yr
    b = np.append(b,mycloud(b[ti],ntracer-1)).view(np.recarray)
    b[ti+1:].r = np.logspace(-4,0,ntracer-1)
    b[ti+1:].Q = 1
    b[ti+1:].eta = 1/3.
        
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

if make_movie:
    cmap = 'gnuplot2'
    csky,csun = -0.75,0.96 # color vals working off of whatever cmap
    csky,csun = 0.5,0.96 # color vals working off of whatever cmap
    cmap = 'gist_gray'
    fmovcstr = 'tmp'+str(os.getpid())+'*.png'
    # print(glob.glob(fmovcstr))
    if len(glob.glob(fmovcstr))>0:
        print('# removing files',fmovcstr)
        os.system('rm '+fmovcstr)

thph = []
thphN,thphS,thphE,thphW = [],[],[],[]

attnlis = []

radpress = True

nm = nm + ['tracer']*(len(b)-7)
dumpmxv(nm,b)

for i in range(npl):
    steps(b,dt,ntsub)
    print(energy(b),energy(bstart))
    re = pairsep(b[1],b[0])
    tnow = dt*(i+1)
    if make_movie:
        thti,phti = skycoords(b[:ti+1],ti)
        if create_cloud:
            bp = puff(b[ti],ntracer//npl, R=0, puffdisp=30, rp=b[ti].r,rhop=rhop)
            bp.r += np.random.normal(0,0.01*b[ti].r,ntracer//npl)
            b = np.append(b,bp).view(np.recarray)
        Z = 0*X + csky
        msk = X**2 + Y**2 < (Rsun/re)**2; Z[msk] = csun
        th,ph = skycoords(b,ti)
        ZZ,_,_ = np.histogram2d(th,ph,bins=[elledge,elledge],)
        Z = np.where(ZZ>0,-1,Z)
        pl.imshow(Z,extent=(elledge[0],elledge[-1],elledge[0],elledge[-1]),cmap=cmap,vmin=-1,vmax=1)
        #pl.scatter(phti*180/np.pi,thti*180/np.pi,c='#ffffff',s=10)
        pl.figtext(0.6,0.2,'%0.3f'%(tnow/yr),color='#660066')
        out = fmovcstr.replace('*','%04d')%(i)
        print('# saving',out)
        noccult = np.sum(ZZ[msk])
        print('# this time, n tot, n occult:',tnow/yr,len(b),noccult)
        pl.savefig(out)
        pl.clf()
    if plot_beta:
        xt,yt,zt = earthsunframe(b,ti)
        tracerdat[i] = np.concatenate((xt,yt,zt))
        print(tnow/yr,'beta...')
    if plot_traj:
        th, ph = skycoords(b[:ti+1],ti)
        nC = np.sum(np.sqrt(th**2+ph**2)<b[0].r/pairsep(b[1],b[0]))
        #neffMC = np.sum(radreduxMC(b[:ti+1]))
        #neff = np.sum(radredux(b))
        thph.append([tnow,th[0],ph[0]])
        print(tnow/yr,nC)
    if do_it:
        #print(b[ti:])
        lat, lon = skycoords(b,tracerindex=ti)
        ang = np.sqrt(lat**2+lon**2)[0]
        attn = attengeo(1.0,ang,d1=0.01*AU,asemi=AU) 
        bri = britensca(1.0,ang,gdraine,nudraine,d1=0.01*AU,asemi=AU)
        attnlis.append(np.sum(attn-bri))
        afac = (1/Rsun**2*AU**2/(0.01*AU)**2)
        print('# attn,bri,net of L1 tracer:',attn/afac,bri/afac,(attn-bri)/afac)
        ZZ,_,_ = np.histogram2d(lat,lon,bins=[elledge,elledge],)
        nzz = np.sum(ZZ)
        print('zz stats',np.min(ZZ),np.mean(ZZ),np.max(ZZ),np.sum(ZZ))
print("# looptime %s s"%(time.time()-tstart))

Estart = energy(bstart)
Efinal = energy(b)
print('frac energy error:',np.abs((Efinal-Estart)/Estart))


if do_it: 
    attnlis = np.array(attnlis)
    afac = (1/Rsun**2*AU**2/(0.01*AU)**2)
    attnmean,attnmin,attnmax = np.mean(attnlis),np.min(attnlis),np.max(attnlis)
    attnmean /= afac;  attnmin /= afac; attnmax /= afac; 
    print("# attenuation summary: mean min max:",attnmean,attnmin,attnmax)

if make_movie:
    os.system('convert '+fmovcstr+' ~/public_html/tmp.gif')
    print('# removing files',fmovcstr)
    os.system('rm '+fmovcstr)

if plot_beta:
    #pl.gca().set_aspect('equal')
    #aa = np.linspace(0,2*np.pi,100)
    #xx,yy = b[1].r * np.cos(aa) ,b[1].r*np.sin(aa)
    #pl.plot(xx,yy,'-',c='#999999')
    #pl.scatter([0],[0],s=5,c='#999999')
    xmax = 0

    for i in range(1,ntracer):
        xt,yt = (tracerdat[:,i]-tracerdat[:,0])/Rearth,(tracerdat[:,ntracer+i]-tracerdat[:,ntracer])/Rearth
        xt *= -1
        rt = b[ti+i].r
        if 1:
            if rt >= 0.1:
                label = '%1.0f mm'%(rt/1e-1)
            else:
                label = '%1.0f $\mu$m'%(rt/1e-4)                
            pl.plot(xt,yt,'-', c='k',alpha=i/(ntracer-1),label=label)
        else:
            pl.plot(xt,yt,'-', c='k',alpha=i/(ntracer-1))       
        xmax = np.max(xt)
    pl.xlim(-1,20)
    pl.ylim(-4.5,1.5)
    pl.xlabel(r'x (R$_\oplus$)',size=14)
    pl.ylabel(r'y (R$_\oplus$)',size=14)
    pl.legend()
    out = 'halojplbeta.pdf'
    pl.savefig(out)
    os.system('convert '+out+' ~/public_html/tmp.jpg')
    
if plot_traj:

    dumpmxv(nm,b)
    
    re = pairsep(b[1],b[0])
    angsun = (b[0].r/re)*180/np.pi
    aa = np.linspace(0,2*np.pi,100)
    xx,yy = angsun*np.cos(aa),angsun*np.sin(aa)
    pl.gca().set_aspect('equal')
    pl.plot(xx,yy,'-',c='#999999')
    pl.xlabel(r'$\Delta\phi$ (deg)',size=14)
    pl.ylabel(r'$\Delta\theta$ (deg)',size=14)

    out = 'halojpltraj.pdf'
    print('dumping to',out)
    ts,th,ph = np.array(thph).T
    hint,pint = interp1d(ts,th),interp1d(ts,ph)
    tsi = np.linspace(ts[0],ts[-1],len(ts)*10)
    thi, phi = hint(tsi), pint(tsi)
    pl.scatter(phi*180/np.pi,thi*180/np.pi,s=0.25,c=tsi,cmap='binary')
    pl.savefig(out)
    os.system('convert '+out+' ~/public_html/tmp.jpg')
    
    myscript = __file__
    fignm = myscript.split('_')[1]
    print(fignm)
    pl.savefig(fignm+".pdf")

