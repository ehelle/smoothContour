# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 12:21:03 2014

@author: eh
"""

from osgeo import ogr, gdal
import os
import numpy as np
import numpy.linalg as la
import re
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help="The shapefile to smooth")
parser.add_argument("output", help="Output shapefile")
args = parser.parse_args()

gdal.UseExceptions()

cwd = os.getcwd()

def getindata(infile):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    return driver.Open(infile, 0)

def createoutdata(outfile):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outfile):
        driver.DeleteDataSource(outfile)
    outdata = driver.CreateDataSource(outfile)
    return outdata

def copylayerdefn(inlayer, outlayer):
    inlayerdefn = inlayer.GetLayerDefn()
    for i in range(0, inlayerdefn.GetFieldCount()):
        fielddefn = inlayerdefn.GetFieldDefn(i)
        outlayer.CreateField(fielddefn)
    return outlayer

def circular(arr):
    if max(abs(arr[0,:] - arr[-1,:])) < 0.0001:
        return True
    else:
        return False

def dist2vec(u,v,p):
    v1 = v-u
    l = la.norm(v1)
    if l == 0:
        return la.norm(p-u)
    t = np.dot(p-u, v-u)/l**2
    if t < 0:
        return la.norm(p-u)
    elif t > 1:
        return la.norm(p-v)
    else:
        return la.norm(p-(u+(np.array([t,t])*(v-u))))

def smoothMcMasters(arr, level=5, circular=False):
    l = arr.shape[0]
    cut = int(level/2)
    level = min(level,l)
    if circular:
        arr = np.append(arr, arr[1:cut*2,:], axis=0)
    for j in [0,1]:
        sm = np.convolve(arr[:,j], np.ones(level)/level, mode='same')
        avg = (arr[:,j] + sm)/2

        #test for threshold
        tolerance = 1.0/3600
        diff = arr[:,j]-avg
        for i in range(len(diff)):
            if abs(diff[i]) > tolerance:
                diff[i] = diff[i]*tolerance/abs(diff[i])
        avg = arr[:,j]-diff

        if circular:
            arr[:-2*cut+1,j] = np.append(avg[-2*cut:-cut], avg[cut:-2*cut+1])
        else:
            arr[cut:-cut,j] = avg[cut:-cut]
    if circular:
        return arr[:-2*cut+1,:]
    else:
        return arr

def smoothChaikin(arr, iterations, circular=False):
    l,col = arr.shape
    if circular:
        new_arr = np.ndarray(shape=(l*2-1,col), dtype=float, order='C')
    else:
        new_arr = np.ndarray(shape=(l*2,col), dtype=float, order='C')
    for i in range(0,l-1):
        for j in range(0,col):
            q = 0.75*arr[i,j]+0.25*arr[i+1,j]
            r = 0.25*arr[i,j]+0.75*arr[i+1,j]
            new_arr[i*2+1,j] = q
            new_arr[i*2+2,j] = r
    if circular:
        new_arr[0,:] = new_arr[-1,:]
    else:
        new_arr[[0,l*2-1],:] = arr[[0,l-1],:]
    if iterations > 1:
        return smoothChaikin(new_arr, iterations-1)
    else:

        return new_arr

def simplifyLang(arr, lookahead, tolerance):
    if lookahead <= 1:
        return arr
    l = arr.shape[0]
    if lookahead > l-1 :
        lookahead = l-1
    keep=[0]
    i=1
    while i < l:
        if i+lookahead >= l:
            lookahead = l-i-1
        offset = recursivetolerancebar(arr, i, lookahead, tolerance)
        if offset > 0:
            keep.append(i)
            i += offset-1
        i += 1
    keep.append(l-1)
    return arr[keep,:]

def recursivetolerancebar(arr, i, lookahead, tolerance):
    n = lookahead
    u = arr[i,:]
    v = arr[i+n,:]
    for j in range(1,n+1):
        p = arr[i+j,:]
        dist = dist2vec(u,v,p)
        if dist >= tolerance:
            n -= 1
            if n > 0:
                return recursivetolerancebar(arr,i,n,tolerance)
            else:
                return 0
    return n

def timer(f):
    def inner(*args, **kwargs):
        t1 = time.time()
        result = f(*args, **kwargs)
        t2 = time.time()
        print("Time elapsed: %2.2f sek" % (t2-t1))
        return result
    return inner

@timer
def main(infile, outfile):
    circ=0
    indata = getindata(infile)
    inlayer = indata.GetLayer()
    spatialref = inlayer.GetSpatialRef()
    outdata = createoutdata(outfile)
    inlayername = inlayer.GetName()
    outlayer = outdata.CreateLayer(inlayername, geom_type=ogr.wkbLineString)
    outlayer = copylayerdefn(inlayer, outlayer)
    outlayerdefn = outlayer.GetLayerDefn()

    for i in range(0, inlayer.GetFeatureCount()):
        infeature = inlayer.GetFeature(i)
        outfeature = ogr.Feature(outlayerdefn)
        for j in range(0, outlayerdefn.GetFieldCount()):
            outfeature.SetField(outlayerdefn.GetFieldDefn(j).GetNameRef(), infeature.GetField(j))

        ingeom = infeature.GetGeometryRef()

        geom_type = ingeom.GetGeometryName()

        if geom_type == 'LINESTRING':
            ingeom = [ingeom]

        wkt = ''

        for geo in ingeom:

            if geo.Length() > 0.002:
                lstr = geo.ExportToWkt()
                regexp = r"(-?\d+.\d+) (-?\d+.\d+)"
                lst = re.findall(regexp, lstr)
                if len(lst) == 0:
                    print lstr

                arr = np.array(lst, dtype=np.float)

                circ = circular(arr)

                if geo.Length() > 0.01:
                    arr = simplifyLang(arr, 3, 1.0/3600)

                arr = smoothMcMasters(arr, circular=circ)

                arr = smoothChaikin(arr, 1, circular=circ)

                arr_of_str = arr.astype(np.str)
                newlst = arr_of_str.tolist()

                str_of_coord = ','.join(map(' '.join, newlst))
                if str_of_coord == '':
                    print lstr, lst
                if wkt != '':
                    wkt += ','
                wkt += '(' + str_of_coord + ')'

        if geom_type == 'MULTILINESTRING':
            wkt = geom_type + ' (' + wkt +')'
        elif geom_type == 'LINESTRING':
            wkt = geom_type + ' ' + wkt
        else:
            print "Unknown geometry: %s" % geom_type
            break

        outfeature.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
        outlayer.CreateFeature(outfeature)

    prjfile = open(outfile[:-3]+'prj', 'w')
    prjfile.write(spatialref.ExportToWkt())
    prjfile.close()
    indata.Destroy()
    outdata.Destroy()

if __name__ == '__main__':
    main(args.input, args.output)