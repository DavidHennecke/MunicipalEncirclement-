import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely import Point, Polygon, to_geojson, validation, LineString, distance
from shapely.ops import unary_union
import fiona
import math
from tqdm import tqdm
import rasterio
from rasterstats import point_query
import time

print("Calculate Enclosure:")

#Calculates points along a polygon boundary
def polygonToPoints(locationGeom, SegmentDistance):
    line = locationGeom.boundary
    points = []
    for i in np.arange(0, line.length, SegmentDistance):
        val = line.interpolate(i)
        points.append([val.x, val.y])
    points = np.array(points)
    return points

def lineToPoints(line, SegmentDistance):
    points = []
    for i in np.arange(0, line.length, SegmentDistance):
        val = line.interpolate(i)
        points.append([val.x, val.y])
    return points

def checkLineOfSight (observerGeom, targetGeom, dsmData, transform, hubHeight):
    sight = True
    #get Line of Sight and Points on Line
    segmentSize = transform[0]
    LoS = LineString([observerGeom, targetGeom])
    LoSPoints = lineToPoints(LoS, segmentSize)
    #get Heights and Distance
    observerHeight = point_query(observerGeom.wkt,dsmData, affine=transform)
    targetHeight = point_query(targetGeom.wkt,dsmData, affine=transform)
    distanceBetweenObserverTarget = distance(observerGeom, targetGeom)
    
    #interpolation between two points x = distance; y= height
    distances = [0,distanceBetweenObserverTarget]
    heights = [observerHeight[0] + 1.65, targetHeight[0]+hubHeight] #1.65 m average size of a human being
    
    for point in LoSPoints:
        height = np.interp(segmentSize, distances, heights)
        segmentSize = segmentSize + segmentSize
        pointRealHeight = point_query(Point(point).wkt,dsmData, affine=transform)[0]
        
        if pointRealHeight > height:
            sight = False
            return sight, LoSPoints
        else:
            sight = True
    return sight, LoSPoints

#Calculates Turbines within a location-buffer
def getStock(locationGeom, stockList, bufferRule, stockFeat):
    buffer = locationGeom.buffer(bufferRule)
    points = []
    feats = []
    # checks each turbine whether they are inside the buffer or not
    for i, turbine in enumerate(stockList):
        if Point(turbine[0], turbine[1]).within(buffer):
            points.append([turbine[0], turbine[1]])    
            feats.append(stockFeat[i])
    points = np.array(points)
    return points, feats


def calculateAzimuth(observer, turbines, observerList):
    X = turbines[:,0] - observer[:,0]
    Y = turbines[:,1] - observer[:,1]
    bearingList = np.arctan2(X,Y)*(180/np.pi)
    bearingList[bearingList < 0] += 360
    
    #plots the result
    # fig, axs = plt.subplots()
    # axs.set_aspect('equal', 'datalim')
    # xs = turbines[:,0]  
    # ys = turbines[:,1]
    # xo = observer[:,0]  
    # yo = observer[:,1] 
    
    # xl = observerList[:,0]  
    # yl = observerList[:,1]
    # axs.plot(xl, yl, alpha=1) #Plot Line
    # axs.plot(xo, yo, "o") #Plot Points
    # axs.plot(xs, ys, "o") #Plot Points
    # plt.show()
    
    return bearingList   


def plot(observer, turbines, observerList, area1, area2, area3, area4):
    #plots the result
    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')
    xs = turbines[:,0]  
    ys = turbines[:,1]
    xo = observer[:,0]  
    yo = observer[:,1] 
    
    xlArea1 = area1[:,0]  
    ylArea1 = area1[:,1]
    axs.fill(xlArea1, ylArea1, alpha=0.7, facecolor="seagreen", edgecolor='darkgreen') #Plot allowedAreas
    
    xlArea4 = area4[:,0]  
    ylArea4 = area4[:,1]
    axs.fill(xlArea4, ylArea4, alpha=0.7, facecolor="cornflowerblue", edgecolor='royalblue') #Plot windfarmAreas
    
    xlArea2 = area2[:,0]  
    ylArea2 = area2[:,1]
    axs.fill(xlArea2, ylArea2, alpha=0.7, facecolor="orange", edgecolor='darkorange') #Plot restrictedAreas
    
    xlArea3 = area3[:,0]  
    ylArea3 = area3[:,1]
    axs.fill(xlArea3, ylArea3, alpha=0.7, facecolor="indianred", edgecolor='darkred') #Plot forbiddenAreas
    
    xl = observerList[:,0]  
    yl = observerList[:,1]
    axs.fill(xl, yl, alpha=1, facecolor="slategrey", edgecolor="black") #Plot observerList
    
    #axs.plot(xo, yo, "o") #Plot observer
    axs.plot(xs, ys, ".", color="black", alpha=1) #Plot turbines
    plt.show()

def plotPolygon(observer, turbines, observerList, area1, area2):
    #plots the result
    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')
    xs = turbines[:,0]  
    ys = turbines[:,1]
    xo = observer[:,0]  
    yo = observer[:,1] 
    
    if len(area1[0])>0:
        xlArea1 = area1[0][:,0]  
        ylArea1 = area1[0][:,1]
        axs.fill(xlArea1, ylArea1, alpha=0.8, facecolor="orange", edgecolor='darkorange') #Plot restrictedAreas

    if len(area1)>1:
        xlArea12 = area1[1][:,0]  
        ylArea12 = area1[1][:,1]
        axs.fill(xlArea12, ylArea12, alpha=0.8, facecolor="orange", edgecolor='darkorange') #Plot restrictedAreas

    if len(area2[0])>0:
        xlArea2 = area2[0][:,0]  
        ylArea2 = area2[0][:,1]
        axs.fill(xlArea2, ylArea2, alpha=0.8, facecolor="indianred", edgecolor='orangered') #Plot forbiddenAreas
    
    if len(area2)>1:
        xlArea21 = area2[1][:,0]  
        ylArea21 = area2[1][:,1]
        axs.fill(xlArea21, ylArea21, alpha=0.8, facecolor="indianred", edgecolor='orangered') #Plot forbiddenAreas

    xl = observerList[:,0]  
    yl = observerList[:,1]
    axs.fill(xl, yl, alpha=1, facecolor="slategrey") #Plot observerList
    
    axs.plot(xo, yo, "o", color="red") #Plot observer
    axs.plot(xs, ys, ".", color="black", alpha=1) #Plot turbines
    plt.show()

def generateAngleGroups(bearingList, observer):
    bearingList = np.sort(bearingList)
    groups = []
    for angle in bearingList:
        if len(groups) == 0:
            groups.append([angle])
        else:
            for i, group in enumerate(groups):
                counterVal = 0
                clockVal = 0
                
                # Get counterVal
                if max(group)>=300 and (min(group)<=60 or min(group)>=300):
                    temp = np.array(group)
                    temp = temp[temp>=180]
                    counterVal = temp.min()-60
                elif min(group)<60 and max(group)<=300:
                    temp = np.array(group)
                    temp = temp[temp<=180]
                    clockVal = temp.min()-60+360
                else:
                    counterVal = min(group)-60
                    if counterVal < 0:
                        counterVal = counterVal + 360
                        
                # Get clockVal    
                if min(group)<=60  and (max(group)<=60 or max(group)>=300):
                    temp = np.array(group)
                    temp = temp[temp<=180]
                    clockVal = temp.max()+60
                elif max(group)>=300 and min(group)>=60:
                    temp = np.array(group)
                    temp = temp[temp>=180]
                    clockVal = temp.max()+60-360
                else:
                    clockVal = max(group)+60
                    if clockVal > 360:
                        clockVal = clockVal - 360
                
                #Check if the angle is within the value range
                if angle < clockVal and angle > counterVal:
                    groups[i].append(angle)
                    break
                elif clockVal<counterVal:
                    if angle < clockVal or angle > counterVal:
                        groups[i].append(angle)
                        break
                else:
                    #if angle is not within the value range --> new group
                    groups.append([angle])             
    cleanGroups(groups)
    return groups   

def evaluateGroups(groups, observer, observerStock, observerList):
    #Init Areas
    windfarmAreasGroup = []
    restrictedAreasGroup = []
    forbiddenAreasGroup = []


    # Iteration through groups list
    for i, group in enumerate(groups):
        #Gets min and max value of group
        minG = min(group)
        maxG = max(group)
        #Checks if group value range exceeds angle threshold 
        if minG <= 60 and maxG >= 300:
            
            #Refresh Values
            minG, maxG, minGRange, maxGRange = getAngleValues(group)
            groupRange = minG + 360 - maxG

            
            #Get windfarmAreas
            ring = []
            ring.append((observer[:,0][0], observer[:,1][0]))
            while minG >=0:
                xT = observer[:,0][0] + math.sin(math.radians(minG))*3500
                yT = observer[:,1][0] + math.cos(math.radians(minG))*3500
                ring.append((xT, yT))
                minG = minG -1
            helper =360
            while helper >maxG:
                xT = observer[:,0][0] + math.sin(math.radians(helper))*3500
                yT = observer[:,1][0] + math.cos(math.radians(helper))*3500
                ring.append((xT, yT))
                helper = helper -1
            ring.append((observer[:,0][0], observer[:,1][0]))
            windfarmAreasGroup.append(Polygon(ring))
            ring = np.array(ring)
            #plotPolygon(observer, observerStock, observerList, [[]], [ring])

 
            #Get forbiddenAreas
            #Refresh new values
            minG, maxG, minGRange, maxGRange = getAngleValues(group)
            if groupRange >= 120:
                ringArea = []
                ring = []
                ring.append((observer[:,0][0], observer[:,1][0]))
                while minGRange >=minG:
                    xT = observer[:,0][0] + math.sin(math.radians(minGRange))*3500
                    yT = observer[:,1][0] + math.cos(math.radians(minGRange))*3500
                    ring.append((xT, yT))
                    minGRange = minGRange -1
                ring.append((observer[:,0][0], observer[:,1][0]))
                forbiddenAreasGroup.append(Polygon(ring))
                ring = np.array(ring)
                ringArea.append(ring)
                ring = []
                ring.append((observer[:,0][0], observer[:,1][0]))
                while maxGRange <maxG:
                    xT = observer[:,0][0] + math.sin(math.radians(maxGRange))*3500
                    yT = observer[:,1][0] + math.cos(math.radians(maxGRange))*3500
                    ring.append((xT, yT))
                    maxGRange = maxGRange +1
                ring.append((observer[:,0][0], observer[:,1][0]))
                forbiddenAreasGroup.append(Polygon(ring))
                ring = np.array(ring)
                ringArea.append(ring)
                #plot(observer, observerStock, observerList, ring)
                #plotPolygon(observer, observerStock, observerList, [[]], ringArea)

                
            #Get restrictedAreas
            minG, maxG, minGRange, maxGRange = getAngleValues(group)
            if groupRange < 120:
                ringArea = []
                allowedRangeDifference = 120 - groupRange
                allowedMinG = minG+allowedRangeDifference
                allowedMaxG = maxG-allowedRangeDifference
                ring = []
                ring.append((observer[:,0][0], observer[:,1][0]))
                while minG <=allowedMinG:
                    xT = observer[:,0][0] + math.sin(math.radians(minG))*3500
                    yT = observer[:,1][0] + math.cos(math.radians(minG))*3500
                    ring.append((xT, yT))
                    minG = minG +1
                ring.append((observer[:,0][0], observer[:,1][0]))
                restrictedAreasGroup.append(Polygon(ring))
                ring = np.array(ring)
                ringArea.append(ring)
                ring = []
                ring.append((observer[:,0][0], observer[:,1][0]))
                while maxG >allowedMaxG:
                    xT = observer[:,0][0] + math.sin(math.radians(maxG))*3500
                    yT = observer[:,1][0] + math.cos(math.radians(maxG))*3500
                    ring.append((xT, yT))
                    maxG = maxG -1
                ring.append((observer[:,0][0], observer[:,1][0]))
                restrictedAreasGroup.append(Polygon(ring))
                ring = np.array(ring)
                ringArea.append(ring)
                #plotPolygon(observer, observerStock, observerList, ringArea, [[]])

        else:
            groupRange = maxG - minG
            #Get windfarmAreas
            if maxG != minG:
                ring = []
                ring.append((observer[:,0][0], observer[:,1][0]))
                while minG <=maxG:
                    xT = observer[:,0][0] + math.sin(math.radians(minG))*3500
                    yT = observer[:,1][0] + math.cos(math.radians(minG))*3500
                    ring.append((xT, yT))
                    minG = minG +1
                xT = observer[:,0][0] + math.sin(math.radians(maxG))*3500
                yT = observer[:,1][0] + math.cos(math.radians(maxG))*3500
                ring.append((xT, yT))
                ring.append((observer[:,0][0], observer[:,1][0]))
                windfarmAreasGroup.append(Polygon(ring))
                ring = np.array(ring)
                #plotPolygon(observer, observerStock, observerList, [[]], [ring])
            
            #Get forbiddenAreas
            minG = min(group)
            maxG = max(group)
            if maxG - minG >= 120:
                forbiddenAreasGroup, ring = calcForbiddenOrRestrictedAreas_Normal(minG, maxG, 60, observer, group)
                # forbiddenAreasGroup.append(Polygon(ring))
                # ring = np.array(ring)
                # plot(observer, observerStock, observerList, ring)
                #plotPolygon(observer, observerStock, observerList, [[]], ring)

            #Get restrictedAreas
            minG = min(group)
            maxG = max(group)
            if maxG - minG < 120:
                allowedRangeDifference = 120 - (maxG-minG)
                restrictedAreasGroup, ring = calcForbiddenOrRestrictedAreas_Normal(minG, maxG, allowedRangeDifference, observer, group)               
                # restrictedAreasGroup.append(Polygon(ring))
                #plotPolygon(observer, observerStock, observerList, ring, [[]])
    
    
    windfarmArea = unary_union(windfarmAreasGroup)
    restrictedArea = unary_union(restrictedAreasGroup)
    forbiddenArea = unary_union(forbiddenAreasGroup)
    return windfarmArea, restrictedArea, forbiddenArea  

def calcForbiddenOrRestrictedAreas_Normal(minG, maxG, additionalAngle, observer, group):
    minGRange = minG-additionalAngle
    maxGRange = maxG+additionalAngle
    
    areasGroup = []

    ring = []
    ring.append((observer[:,0][0], observer[:,1][0]))
    
    # If the angle goes beyond 0
    if minGRange < 0:
        ringArea = []
        minGRange = 360 + minGRange
        while minGRange <360:
            xT = observer[:,0][0] + math.sin(math.radians(minGRange))*3500
            yT = observer[:,1][0] + math.cos(math.radians(minGRange))*3500
            ring.append((xT, yT))
            minGRange = minGRange +1
        minGRange = 0
        while minGRange <= minG:
            xT = observer[:,0][0] + math.sin(math.radians(minGRange))*3500
            yT = observer[:,1][0] + math.cos(math.radians(minGRange))*3500
            ring.append((xT, yT))
            minGRange = minGRange +1
        if (additionalAngle!=120):
            ring.append((observer[:,0][0], observer[:,1][0]))
            areasGroup.append(Polygon(ring))
            ring = np.array(ring)
            ringArea.append(ring)
            ring = []
            ring.append((observer[:,0][0], observer[:,1][0]))
        while maxG <=maxGRange:
            xT = observer[:,0][0] + math.sin(math.radians(maxG))*3500
            yT = observer[:,1][0] + math.cos(math.radians(maxG))*3500
            ring.append((xT, yT))
            maxG = maxG +1
        ring.append((observer[:,0][0], observer[:,1][0]))
        areasGroup.append(Polygon(ring))
        ring = np.array(ring)
        ringArea.append(ring)
        return areasGroup, ringArea
    
    # If the angle goes beyond 360
    elif maxGRange > 360:
        ringArea = []
        minG = min(group)
        maxG = max(group)
        minGRange = minG-additionalAngle
        maxGRange = maxG+additionalAngle
        maxGRange = maxGRange -360
        while maxGRange >0:
            xT = observer[:,0][0] + math.sin(math.radians(maxGRange))*3500
            yT = observer[:,1][0] + math.cos(math.radians(maxGRange))*3500
            ring.append((xT, yT))
            maxGRange = maxGRange -1
        maxGRange = 360
        while maxGRange >= maxG:
            xT = observer[:,0][0] + math.sin(math.radians(maxGRange))*3500
            yT = observer[:,1][0] + math.cos(math.radians(maxGRange))*3500
            ring.append((xT, yT))
            maxGRange = maxGRange -1
        if (additionalAngle!=120):
            ring.append((observer[:,0][0], observer[:,1][0]))
            areasGroup.append(Polygon(ring))
            ring = np.array(ring)
            ringArea.append(ring)
            ring = []
            ring.append((observer[:,0][0], observer[:,1][0]))
        while minG >=minGRange:
            xT = observer[:,0][0] + math.sin(math.radians(minG))*3500
            yT = observer[:,1][0] + math.cos(math.radians(minG))*3500
            ring.append((xT, yT))
            minG = minG -1
        ring.append((observer[:,0][0], observer[:,1][0]))
        areasGroup.append(Polygon(ring))
        ring = np.array(ring)
        ringArea.append(ring)
        return areasGroup, ringArea
    
    #Normal state
    else:
        ringArea = []
        minG = min(group)
        maxG = max(group)
        minGRange = minG-additionalAngle
        maxGRange = maxG+additionalAngle
        while minGRange <=minG:
            xT = observer[:,0][0] + math.sin(math.radians(minGRange))*3500
            yT = observer[:,1][0] + math.cos(math.radians(minGRange))*3500
            ring.append((xT, yT))
            minGRange = minGRange +1
        if (additionalAngle!=120):
            ring.append((observer[:,0][0], observer[:,1][0]))
            areasGroup.append(Polygon(ring))
            ring = np.array(ring)
            ringArea.append(ring)
            ring = []
            ring.append((observer[:,0][0], observer[:,1][0]))
        while maxGRange >=maxG:
            xT = observer[:,0][0] + math.sin(math.radians(maxGRange))*3500
            yT = observer[:,1][0] + math.cos(math.radians(maxGRange))*3500
            ring.append((xT, yT))
            maxGRange = maxGRange -1
        ring.append((observer[:,0][0], observer[:,1][0]))
        areasGroup.append(Polygon(ring))
        ring = np.array(ring)
        ringArea.append(ring)
        return areasGroup, ringArea
    
def cleanGroups(groups):
    for i, group in enumerate(groups):
        #Megre Groups if splitted
        for a, groupT in enumerate(groups):
            minG = min(group)
            maxG = max(group)
            minG, maxG, minGRange, maxGRange = getAngleValues(group)
            if a != i:
                minGT = min(groupT)
                maxGT = max(groupT)
                if minGT < minGRange or maxGT > maxGRange:
                    groups[i] = group + groupT
                    del groups[a] 
                    cleanGroups(groups)

def getAngleValues(group):
    minArray = np.array(group)[np.array(group)<=180]
    maxArray = np.array(group)[np.array(group)>180]
    if minArray.size > 0 and maxArray.size > 0:
        minG = minArray.max()
        minGRange = minG + 60
        maxG = maxArray.min()
        maxGRange = maxG - 60
    elif minArray.size > 0:
        minG = min(group)
        minGRange = max(group) + 60
        maxG = max(group)
        maxGRange = min(group) + 360- 60
    elif maxArray.size > 0:
        minG = min(group)
        minGRange = max(group) - 360 + 60
        maxG = max(group)
        maxGRange = min(group) - 60

        
    return minG, maxG, minGRange, maxGRange

# MAIN
def main():
    #PARAMETER
    SegmentDistance = 200
    bufferRule = 3500

    #DATA PATHS
    totalStockPath = r"Testdaten/stock_32633.shp"
    locationsPath = r"Testdaten/Blankenhagen_32633.shp"
    dsmPath = r"Testdaten/SRTM_D_32633.tif"
    
    #load DSM-Raster
    dsm = rasterio.open(dsmPath)
    dsmData = dsm.read(1)
    transform = dsm.transform

    #load Vectordata
    totalStock = fiona.open(totalStockPath)
    locations = fiona.open(locationsPath)
    
    #Calculates enclosure for each location
    for location in locations:
        print("Location ID: ",location["id"])
        
        #Init Outputs
        windfarmAreasList = []
        restrictedAreasList = []
        forbiddenAreasList = []
        
        #Get location geometry and calculate ObserverPoins
        locationGeom = shape(location["geometry"])
        observerList = polygonToPoints(locationGeom, SegmentDistance)
        
        #Get affected turbines by location
        totalStockGeoms = [ shape(feat["geometry"]) for feat in totalStock ]
        totalStockList = [ np.array((geom.xy[0][0], geom.xy[1][0])) for geom in totalStockGeoms ]
        stock, stockFeat = getStock(locationGeom, totalStockList, bufferRule, totalStock)
        
        #ProgressBar Init
        pbar = tqdm(total=observerList.size/2)
        st = time.time()
        #Calculates enclosure for each observer
        for observer in observerList:
            #Get affected turbines by observer
            observerGeom = Point(observer[0], observer[1])
            observerStock, observerStockFeat = getStock(observerGeom, stock, bufferRule, stockFeat)
            #check Line of Sight
            helper = 0 # Number of deletedFeatures
            # for targetId, target in enumerate(observerStockFeat):
            #     targetGeom = Point(shape(target["geometry"]))
            #     hubHeight = target['properties']['Height']
            #     sight, LoSPoints = checkLineOfSight (observerGeom, targetGeom, dsmData, transform, hubHeight)
            #     if sight == False:
            #         observerStock = np.delete(observerStock,targetId-helper,0)
            #         helper = helper + 1
            #Calculation continues if turbines are nearby
            if observerStock.size != 0:
                #change observer to a list
                observer = np.vstack((observer[0], observer[1])).T
                #Calculates angles to turbines of observerStock
                bearingList = calculateAzimuth(observer, observerStock, observerList)
                #Find groups of turbines by angle
                groups = generateAngleGroups(bearingList, observer)
                #Evaluate turbine groups 
                windfarmArea, restrictedArea, forbiddenArea = evaluateGroups(groups, observer, observerStock, observerList)
                
                if windfarmArea.geom_type == 'GeometryCollection':
                    for geom in windfarmArea.geoms:
                        windfarmAreasList.append(geom)
                else:
                    windfarmAreasList.append(windfarmArea)
                
                if restrictedArea.geom_type == 'GeometryCollection':
                    for geom in restrictedArea.geoms:
                        restrictedAreasList.append(geom)
                else:
                    restrictedAreasList.append(restrictedArea)
                    
                if forbiddenArea.geom_type == 'GeometryCollection':
                    for geom in forbiddenArea.geoms:
                        forbiddenAreasList.append(geom)
                else:
                    forbiddenAreasList.append(forbiddenArea)
                
                

            pbar.update(1)
        

        windfarmAreas = unary_union(windfarmAreasList)
        restrictedAreas = unary_union(restrictedAreasList)
        forbiddenAreas = unary_union(forbiddenAreasList)

        allowedAreasRing = []
        for coord in locationGeom.buffer(bufferRule).exterior.coords:
            allowedAreasRing.append(coord)
            
        windfarmAreasRing = []
        for coord in windfarmAreas.exterior.coords:
            windfarmAreasRing.append(coord)
            
        restrictedAreasRing = []
        for coord in restrictedAreas.exterior.coords:
            restrictedAreasRing.append(coord)
            
        forbiddenAreasRing = []
        for coord in forbiddenAreas.exterior.coords:
            forbiddenAreasRing.append(coord)
            
        plot(observerList, stock, observerList, np.array(allowedAreasRing), np.array(restrictedAreasRing), np.array(forbiddenAreasRing), np.array(windfarmAreasRing))
        
        
        
        with open("Ergebnisse/windFarmAreasJSON.geojson", "w") as f:
            f.write(to_geojson(windfarmAreas))
        with open("Ergebnisse/restrictedAreasJSON.geojson", "w") as f:
            f.write(to_geojson(restrictedAreas))
        with open("Ergebnisse/forbiddenAreasJSON.geojson", "w") as f:
            f.write(to_geojson(forbiddenAreas))
        
        pbar.close()     
        et = time.time()   
        elapsedTime = et - st
        print ("Elapsed time: "+ str(elapsedTime))
if __name__ == "__main__":
    main()
    

