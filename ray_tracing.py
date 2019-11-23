# import numpy as np
# import matplotlib.pyplot as plt
import math
from PIL import Image

MAX_RAY_DEPTH = 5
INFINITY = 1e8

class Vec3:
    '''
    3d vector
    '''
    x,y,z = [0,0,0]

    def __repr__(self):
        return("[%f, %f, %f]"%(self.x, self.y, self.z))
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __init__(self, *xyz):
        if len(xyz)>0:
            if len(xyz)==1:
                self.x, self.y, self.z = xyz[0], xyz[0], xyz[0]
            else:
                self.x, self.y, self.z = xyz

    @property
    def length2(self):
        tmp = self.x**2+self.y**2+self.z**2
        return(tmp)

    @property
    def length(self):
        return(math.sqrt(self.length2))

    def normalize(self):
        nor2 = self.length2
        if(nor2>0):
            invNor = 1.0/math.sqrt(nor2)
            self.x *= invNor; self.y *= invNor; self.z *= invNor
        return self

    def __mul__(self, f):
        if( isinstance(f, Vec3) ):
            return Vec3(self.x*f.x, self.y*f.y, self.z*f.z)
        else:
            return Vec3(self.x*f, self.y*f, self.z*f)

    def dot(self, f):
        return self.x*f.x + self.y*f.y + self.z*f.z

    def __add__(self, f):
        return Vec3(self.x+f.x, self.y+f.y, self.z+f.z)

    def __sub__(self, f):
        return Vec3(self.x-f.x, self.y-f.y, self.z-f.z)

    def __iadd__(self, f):
        self.x += f.x; self.y += f.y; self.z += f.z
        return self

    def __imul__(self, f):
        self.x *= f.x; self.y *= f.y; self.z *= f.z
        return self

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

class Sphere:
    '''
    sphere class
    '''
    center = Vec3()  # position of the sphere
    radius, radius2 = 0.0, 0.0  # sphere radius and radius^2
    surfaceColor = Vec3(); emissionColor = Vec3()
    transparency, reflection = 0.0, 0.0

    def __init__(self, c, r, sc, refl=0.0, transp=0.0, ec=Vec3()):
        self.center = c; self.radius = r; self.radius2 = r**2;
        self.surfaceColor = sc; self.emissionColor = ec
        self.transparency = transp; self.reflection = refl

    # Compute a ray-sphere intersect using the geometric solution
    def intersect(self, rayorig, raydir):
        l = self.center - rayorig
        tca = l.dot(raydir)
        # too large included-angle to intersect
        if(tca < 0):
            return (False, None, None)
        d2 = l.dot(l) - tca*tca
        #
        if(d2 > self.radius2):
            return (False, None, None)
        thc = math.sqrt(self.radius2-d2)
        t0 = tca - thc
        t1 = tca + thc
        return (True, t0, t1)

def mix(a, b, mix):
    return( b*mix + a*(1-mix) )

def trace(rayorig, raydir, spheres, depth):
    '''
    The main trace function
    '''
    tnear = INFINITY
    # nsphere = Sphere()
    # find the closest sphere for each image point
    for sphere in spheres:
        t0, t1 = INFINITY, INFINITY
        b, t0, t1 = sphere.intersect(rayorig, raydir)
        if(b):
            if(t0< 0):
                t0 = t1
            if(t0<tnear):
                tnear = t0
                nsphere = sphere # the nearest sphere

    if( INFINITY==tnear ):
        return Vec3(2)

    # color of the ray/surface of the object intersected by the ray
    surfaceColor = Vec3()
    # point of intersection
    phit = rayorig + raydir*tnear
    # normal at the intersection point
    nhit = phit - nsphere.center
    nhit.normalize()

    bias = 1e-4
    inside = False
    if (raydir.dot(nhit) > 0):
        nhit = -nhit
        inside = True
    if ( (nsphere.transparency>0) or (nsphere.reflection>0) ) \
        and ( depth<MAX_RAY_DEPTH ):
        facingratio = -raydir.dot(nhit)
        # change the mix value to tweak the effect
        fresneleffect = mix( (1-facingratio)**3, 1, 0.1 )
        # compute refleciton direction
        refldir = raydir - nhit*2*raydir.dot(nhit)
        refldir.normalize()
        reflection = trace( phit+nhit*bias, refldir, spheres, depth+1 )
        refraction = Vec3(0.0)
        if( nsphere.transparency ):
            ior = 1.1
            eta = 1 if inside else (1/ior)
            cosi = -nhit.dot(raydir)
            k = 1 - eta * eta * (1 - cosi * cosi)
            refrdir = raydir * eta + nhit * (eta *  cosi - math.sqrt(k))
            refrdir.normalize()
            refraction = trace(phit - nhit * bias, refrdir, spheres, depth + 1)
        # the result is a mix of reflection and refraction (if the sphere is transparent)
        surfaceColor = ( \
            reflection * fresneleffect + \
            refraction * (1 - fresneleffect) * nsphere.transparency) * nsphere.surfaceColor;
    else:
        # it's a diffuse object, no need to raytrace any further
        for i in range(len(spheres)):
            if spheres[i].emissionColor.x > 0:
                # this is a light
                transmission = Vec3(1)
                lightDirection = spheres[i].center - phit
                lightDirection.normalize()
                for j in range(len(spheres)):
                    if(i!=j):
                        [b, t0, t1] = spheres[j].intersect(phit+nhit*bias, lightDirection)
                        transmission = 0
                        break
                surfaceColor += nsphere.surfaceColor * transmission * \
                                max(0.0, nhit.dot(lightDirection)) * spheres[i].emissionColor
    return( surfaceColor+nsphere.emissionColor )


def render(spheres):
    '''
    The main render function
    '''
    width, height = 640, 480
    invWidth, invHeight = 1/width, 1/height
    # image = []# [ [[0,0,0]]*width ]*height
    image = Image.new('RGB', (width, height), 255)
    data = image.load()
    fov = 30
    aspectratio = width/height
    angle = math.tan( math.pi*0.5*fov/180 )
    pixel = Vec3(0)
    # trace rays
    for y in range(height):
        # row = []
        for x in range(width):
            xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio
            yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle
            raydir = Vec3(xx, yy, -1)
            raydir.normalize()
            pixel = trace(Vec3(0), raydir, spheres, 0)

            # pixel = [int(min(1,pixel.x)*255), int(min(1,pixel.y)*255),\
            #         int(min(1,pixel.z)*255)]
            # row.append(pixel)
            data[x,y] = ( int(min(1.0,pixel.x)*255), \
                        int(min(1.0,pixel.y)*255),\
                        int(min(1.0,pixel.z)*255) )
        # image.append(row)
    # plt.imshow(image)
    # plt.show()
    image.show()
    return(image)


if __name__=='__main__':
    spheres = []
    # position, radius, surface color, reflectivity, transparency, emission color
    spheres.append(Sphere(Vec3( 0.0, -10004, -20), 10000, Vec3(0.20, 0.20, 0.20), 0, 0.0));
    spheres.append(Sphere(Vec3( 0.0,      0, -20),     4, Vec3(1.00, 0.32, 0.36), 1, 0.5));
    spheres.append(Sphere(Vec3( 5.0,     -1, -15),     2, Vec3(0.90, 0.76, 0.46), 1, 0.0));
    spheres.append(Sphere(Vec3( 5.0,      0, -25),     3, Vec3(0.65, 0.77, 0.97), 1, 0.0));
    spheres.append(Sphere(Vec3(-5.5,      0, -15),     3, Vec3(0.90, 0.90, 0.90), 1, 0.0));
    # light
    spheres.append(Sphere(Vec3( 0.0,     20, -30),     3, Vec3(0.00, 0.00, 0.00), 0, 0.0, Vec3(3)));
    image = render(spheres)
