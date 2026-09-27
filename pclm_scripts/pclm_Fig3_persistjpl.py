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
from constants import *
G = GNewt

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

drdv = np.array([9.7672356395251e+07, 1.0145546625866e+08, 1.0474139184824e+05, 1.0271868687687e+01, 9.7177468883876e+00, 9.7381395186548e+00]) # best quad

setorb(b,drdv,tracerindex=ti)
bstart = b.copy()
        
ntracer = 1000
ntracer = 1
tmax = 1.0*yr
ddt = 1*yr/4000
npl = 400 # numper of plot calls
rhop = 2.7
micron = 1e-4
nm = 1e-7
rp = micron


tmax = 200*24*3600
npl = 2000
dt = tmax/npl
ntsub = int(dt/ddt)+1

radii = np.array([0.001,0.01,0.1,1,10])*micron 
nradii = len(radii)
nbeta = 200
beta = np.logspace(-5,0,nbeta)
ntracer = nbeta
b = np.append(b,mycloud(b[ti],ntracer-1)).view(np.recarray)
bstart = b.copy()

for j,rp in enumerate(radii):
    
    b = bstart.copy()
    b[ti:].r = rp; b[ti:].m = 4*np.pi/3*rhop*rp**3
    b[ti:].Q = 0.0; b[ti:].eta = 1/3.
    col = 0.2 + 0.8*(1-(j+1)/len(radii))
    col = (col,col,col)
    Ap = np.pi*rp**2
    mp = 4*np.pi/3*rhop*rp**3
    
    driftbeta = np.zeros(ntracer)
    startt = 0.0*tmax
    started = False
    attn0 = attengeo(rp,0.0,d1=0.01*AU,asemi=AU) 

    for i in range(npl):
        steps(b,dt,ntsub)
        re = pairsep(b[1],b[0])
        tnow = dt*(i+1)
        if not started:
            if tnow >= startt:
                started = True
                b[ti:].Q = beta/0.213*(rhop/2.7)*rp/micron # beta is 0.213 Q / (rhop/2.7 cgs) / (rp/micron)
                b[ti:].Q = beta/(Lsun*Ap/(4*np.pi*clight*G*Msun*mp))
        else:
            #th, ph = skycoords(b[:],ti)
            #msk = np.sqrt(th**2+ph**2)<b[0].r/pairsep(b[1],b[0])
            #driftbeta[msk] += dt
            lat, lon = skycoords(b,tracerindex=ti)
            ang = np.sqrt(lat**2+lon**2)
            attn = attengeo(rp,ang,d1=0.01*AU,asemi=AU)
            if (rp < 1*micron) and (tnow-startt)>60*(24*3600):
                break
            driftbeta += dt * attn/attn0

    print(' beta   rp (micron)    persist time (day)')
    for bb,pt in zip(beta,driftbeta):
        print('%13.5f %13.2f %13.5g'%(bb,rp/micron,pt/(24*3600)))

    #pl.semilogx(beta,driftbeta/(24*3600),label='%g'%(rp/micron))
    label = r'%g'%(rp/micron) if rp>20*nm else r'%g'%(rp/nm)
    if (rp <= 20*nm): label += r' nm' 
    if (rp >= 20*nm): label += r' $\mu$m' 
    pl.loglog(beta,driftbeta/(24*3600),'-',label=label,c=col,zorder=100-j)

pl.xlim(beta[0],beta[-1])
pl.ylim(0.8,190)
pl.xlabel(r'$\beta$',size=14)
pl.ylabel(r'persistence (days)',size=14)
pl.legend(loc=[0.72,0.6])


myscript = os.path.basename(__file__)
fignm = myscript.split('_')[1]
print(fignm)
pl.savefig(fignm+".pdf")

out = 'persistjpl.pdf'
pl.savefig(out)
os.system('convert '+out+' ~/public_html/tmp.jpg')

