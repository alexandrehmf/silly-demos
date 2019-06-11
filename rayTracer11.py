import matplotlib.pyplot as plt
import numpy as np
import math
import random
import time

tempoInicial = time.time()

size = 128

canvas = np.zeros((size,size,3),dtype=np.uint8)

#camPos = np.array([ 0., 0., 9.])
#camDir = np.array([ 0., 0.,-1.])
camPos = np.array([-9.,4.5, 9.])
camDir = np.array([ 1.,-.5,-1.])
camDir = camDir/np.linalg.norm(camDir)
camUp  = np.array([ 0., 1., 0.])
camRg  = np.cross(camDir,camUp)
scrHig = 0.5
scrWid = 0.5
scrDst = 1.

pxHig = scrHig/size
pxWid = scrWid/size

class Sphere:
  def __init__(self,pos,rd,r,g,b,m):
    self.pos = np.array(pos)
    self.rd = rd
    self.r = r
    self.g = g
    self.b = b
    self.m = m

spheres = []
#spheres.append(Sphere([ 2  , 0  , 0  ],  1,  0,127,  0,0))
spheres.append(Sphere([ 1  , 0  , 0  ],  1,127,255,127,-3))
spheres.append(Sphere([ 0  , 0  , 0  ],  1,255,  0,  0,0))
spheres.append(Sphere([-2  , 0  , 0  ],  1,255,  0,  0,-1))
spheres.append(Sphere([-1  , 0  ,-1  ],  1,  0,  0,255,-5))
#spheres.append(Sphere([-1  , 1  ,-0.5],  1,  0,255,255,0))
#spheres.append(Sphere([-1  , 1  ,-4  ],  1,255,255,  0,0))
#spheres.append(Sphere([ 1.5, 1.5, 2  ], .8,255,  0,255,0))
spheres.append(Sphere([ 0  ,-100,-0  ] ,99, 64,197, 64,0))

def skyColor(direction):
  normalizedDirection = direction/np.linalg.norm(direction)
  upness = np.dot([0,1,0],normalizedDirection)
  #return([97-(upness*upness*63),97-(upness*upness*63),198])
  return([200,200,255])

def castRay(origin,through):
  rayDir = through - origin
  rayDir = rayDir/np.linalg.norm(rayDir)
  collision = []
  sphId = -1
  a = 0
  for i, s in enumerate(spheres):
    #d = (np.sum(s.pos)-np.sum(origin))/np.sum(rayDir)#parametro do ponto mais proximo deduzido derivando a distancia
    #d = (np.sum([x*y for x,y in zip(s.pos,rayDir)])-np.sum([x*y for x,y in zip(origin,rayDir)]))/np.sum([x*x for x in rayDir])#parametro do ponto mais proximo deduzido derivando a distancia
    d = (np.sum(s.pos*rayDir)-np.sum(origin*rayDir))/np.sum(rayDir*rayDir)#parametro do ponto mais proximo deduzido derivando a distancia
    pn = origin+(d*rayDir) #pointnear
    dfrts = np.linalg.norm(s.pos-pn)#distancia da esfera ao raio
    if np.linalg.norm(dfrts)<s.rd:#o ponto mais proximo esta dentro da esfera?
      c = math.sqrt(s.rd*s.rd-dfrts*dfrts)#distancia do ponto do raio mais proximo do centro da esfera ate a casca da esfera
      k = 0
      if d >= c:#então o raio foi disparado de fora da esfera e o+raydir*(d-c) é o ponto da colisão 
        b = origin+rayDir*(d-c)
        k = d-c
      elif d > 0:
        b = origin+rayDir*(d+c)
        k = d+c
      if sphId == -1 and d>0:
        sphId = i
        collision = b
        a = k 
      elif k < a and d>0:
        sphId = i
        collision = b
        a = k
  return(sphId,collision,rayDir)

def sumRays(depth,sphere,position,rayDir):
  n = 50
  s = sphere
  normal = (position - s.pos)
  normal = normal/np.linalg.norm(normal)
  t1 = np.cross(np.array([0,1,0]),normal)
  t1 = t1/np.linalg.norm(t1)
  t2 = np.cross(normal,t1)
  if s.m == 0:
    r = 0
    g = 0
    b = 0
    for i in range(n):
      u1 = random.random()
      u2 = random.random()
      k = math.sqrt(1-(u1*u1))
      a1 = k*math.cos(2*math.pi*u2)
      a2 = k*math.sin(2*math.pi*u2)
      a3 = u2
      rayPt = position+(t1*a1)+(t2*a2)+(normal*a3)
      sI,hitPos,rayD = castRay(position+(normal*0.005),rayPt)
      if sI != -1:
        si = spheres[sI]
        if depth == 1:
          color = skyColor(rayPt-position)
          r = r + color[0]*((si.r)/255)*((s.r)/255)
          g = g + color[1]*((si.g)/255)*((s.g)/255)
          b = b + color[2]*((si.b)/255)*((s.b)/255)
        else:
          suma=sumRays(depth-1,si,hitPos,rayD)
          r = suma[0]
          g = suma[1]
          b = suma[2]
      else:
        color = skyColor(rayPt-position)
        r = r + color[0]*((s.r)/255)
        g = g + color[1]*((s.g)/255)
        b = b + color[2]*((s.b)/255)
    r = int(r/n)
    g = int(g/n)
    b = int(b/n)
    return([r,g,b])
  elif s.m > 0:
    r = 0
    g = 0
    b = 0
    reflected = rayDir-2*normal*np.dot(rayDir,normal)
    i=0
    while i < n:
      x = random.random()-.5
      y = random.random()-.5
      z = random.random()-.5
      if x*x+y*y+z*z < .25:
        i = i+1
        x = x/s.m
        y = y/s.m
        z = z/s.m
        fuzyVec=np.array([x,y,z])
        sI,hitPos,rayD = castRay(position+(normal*0.005),position+reflected+fuzyVec)
        if sI !=-1:
          si = spheres[sI]
          if depth == 1:
            color = skyColor(reflected)
            r = r + color[0]*((si.r)/255)*((s.r)/255)
            g = g + color[1]*((si.g)/255)*((s.g)/255)
            b = b + color[2]*((si.b)/255)*((s.b)/255)
          else:
            suma = sumRays(depth-1,si,hitPos,rayD)
            r = suma[0]
            g = suma[1]
            b = suma[2]
        else:
          color = skyColor(reflected)
          r = r + color[0]*((s.r)/255)
          g = g + color[1]*((s.g)/255)
          b = b + color[2]*((s.b)/255)
    r = int(r/n)
    g = int(g/n)
    b = int(b/n)
    return([r,g,b])
  elif s.m<0:
    r = 0
    g = 0
    b = 0
    reflected = rayDir-2*normal*np.dot(rayDir,normal)
    if np.dot(rayDir,normal) <= 0:
      goingIn = True
      refracted = rayDir-normal*(s.m-1)
      refracted = refracted/np.linalg.norm(refracted)
    else:
      goingIn = False
      refracted = rayDir-normal/(s.m-1)
      refracted = refracted/np.linalg.norm(refracted)
    if goingIn:
      sI,hitPos,rayD = castRay(position-(normal*0.005),position+refracted)
    else:
      sI,hitPos,rayD = castRay(position+(normal*0.005),position+refracted)
    if sI != -1:
      si = spheres[sI]
      if depth == 1:
        color = skyColor(refracted)
        r = r + color[0]*((si.r)/255)*((s.r)/255)
        g = g + color[1]*((si.g)/255)*((s.g)/255)
        b = b + color[2]*((si.b)/255)*((s.b)/255)
      else:
        suma = sumRays(depth,si,hitPos,rayD)
        r = suma[0]
        g = suma[1]
        b = suma[2]
    else:
      color = skyColor(refracted)
      r = r + color[0]*((s.r)/255)
      g = g + color[1]*((s.g)/255)
      b = b + color[2]*((s.b)/255)
    return([r,g,b])

porcentagem = 0
pixeln = 0
pixels = size*size

for Y in range(size):
  y = size-1-Y
  for x in range(size):
    #canvas[Y,x]=[255*(float(x)/size),255*(float(y)/size),0]
    pxPos = camPos+camDir*scrDst+((x-size//2)*pxWid*camRg)+((y-size//2)*pxHig*camUp)
    #quero material(sfera) e normal(posição) tenho que mudar castRay para retornar id da sphere e posição
    #para então eu começar a somar as cores das coisas que estão oticamente ligadas aquele ponto
    sI,hitPos,rayDir = castRay(camPos,pxPos)
    if sI != -1:
      #depth = 1
      s = spheres[sI]
      #canvas[Y,x]=[s.r,s.g,s.b]
      canvas[Y,x]=sumRays(2,s,hitPos,rayDir)
    else:
      canvas[Y,x]=skyColor(pxPos-camPos)
    #tudo que eu fizer aqui acontece para cada pixel
    pixeln = pixeln+1
    print(pixeln/pixels)

tempoFinal = time.time()
tempo = tempoFinal - tempoInicial
print(tempo/60)

plt.imsave('lostCanvas6.png', canvas)