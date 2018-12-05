def distances2probability(distances, PARM_truncation, PARM_attenuation):
    
    probabilities = 1 - distances / np.max(distances)  
    probabilities *= (probabilities > PARM_truncation)
    probabilities = pow(probabilities, PARM_attenuation) #attenuate the values
    #check if we didn't truncate everything!
    if np.sum(probabilities) == 0:
        #then just revert it
        probabilities = 1 - distances / np.max(distances) 
        probabilities *= (probabilities > PARM_truncation*np.max(probabilities)) # truncate the values (we want top truncate%)
        probabilities = pow(probabilities, PARM_attenuation)
    probabilities /= np.sum(probabilities) #normalize so they add up to one  
    
    return probabilities

def getBestCandidateCoord(bestCandidateMap, outputSize):
    
    candidate_row = floor(np.argmax(bestCandidateMap) / outputSize[0])
    candidate_col = np.argmax(bestCandidateMap) - candidate_row * outputSize[1]
    
    return candidate_row, candidate_col

def loadExampleMap(exampleMapPath):
    exampleMap = io.imread(exampleMapPath) #returns an MxNx3 array
    exampleMap = exampleMap / 255.0 #normalize
    #make sure it is 3channel RGB
    if (np.shape(exampleMap)[-1] > 3): 
        exampleMap = exampleMap[:,:,:3] #remove Alpha Channel
    elif (len(np.shape(exampleMap)) == 2):
        exampleMap = np.repeat(exampleMap[np.newaxis, :, :], 3, axis=0) #convert from Grayscale to RGB
    return exampleMap

def getNeighbourhood(mapToGetNeighbourhoodFrom, kernelSize, row, col):
    
    halfKernel = floor(kernelSize / 2)
    
    if mapToGetNeighbourhoodFrom.ndim == 3:
        npad = ((halfKernel, halfKernel), (halfKernel, halfKernel), (0, 0))
    elif mapToGetNeighbourhoodFrom.ndim == 2:
        npad = ((halfKernel, halfKernel), (halfKernel, halfKernel))
    else:
        print('ERROR: getNeighbourhood function received a map of invalid dimension!')
        
    paddedMap = np.lib.pad(mapToGetNeighbourhoodFrom, npad, 'constant', constant_values=0)
    
    shifted_row = row + halfKernel
    shifted_col = col + halfKernel
    
    row_start = shifted_row - halfKernel
    row_end = shifted_row + halfKernel + 1
    col_start = shifted_col - halfKernel
    col_end = shifted_col + halfKernel + 1
    
    return paddedMap[row_start:row_end, col_start:col_end]

def updateCandidateMap(bestCandidateMap, filledMap, kernelSize):
    bestCandidateMap *= 1 - filledMap #remove all resolved from the map
    #check if bestCandidateMap is empty
    if np.argmax(bestCandidateMap) == 0:
        #populate from sratch
        for r in range(np.shape(bestCandidateMap)[0]):
            for c in range(np.shape(bestCandidateMap)[1]):
                bestCandidateMap[r, c] = np.sum(getNeighbourhood(filledMap, kernelSize, r, c))

def initCanvas(exampleMap, size):
    
    #get exampleMap dimensions
    imgRows, imgCols, imgChs = np.shape(exampleMap)
    
    #create empty canvas (-1 value means unresolved pixel)
    canvas = np.zeros((size[0], size[1], imgChs)) #inherit number of channels from example map
    filledMap = np.zeros((size[0], size[1])) #map showing which pixels have been resolved
    
    #init a random 3x3 block
    margin = 1
    rand_row = randint(margin, imgRows - margin - 1)
    rand_col = randint(margin, imgCols - margin - 1)
    exampleMap_patch = exampleMap[rand_row-margin:rand_row+margin+1, rand_col-margin:rand_col+margin+1] #need +1 because last element not included
     #plt.pyplot.imshow(exampleMap_patch)
     #print(np.shape(exampleMap_patch))
    
    #put it in the center of our canvas
    center_row = floor(size[0] / 2)
    center_col = floor(size[1] / 2)
    canvas[center_row-margin:center_row+margin+1, center_col-margin:center_col+margin+1] = exampleMap_patch
    filledMap[center_row-margin:center_row+margin+1, center_col-margin:center_col+margin+1] = 1 #mark those pixels as resolved

    return canvas, filledMap

def prepareExamplePatches(exampleMap, searchKernelSize):
    
    #get exampleMap dimensions
    imgRows, imgCols, imgChs = np.shape(exampleMap)
    
    #find out possible steps for a search window to slide along the image
    num_horiz_patches = imgRows - (searchKernelSize-1);
    num_vert_patches = imgCols - (searchKernelSize-1);
    
    #init candidates array
    examplePatches = np.zeros((num_horiz_patches*num_vert_patches, searchKernelSize, searchKernelSize, imgChs))
    
    #populate the array
    for r in range(num_horiz_patches):
        for c in range(num_vert_patches):
            examplePatches[r*num_vert_patches + c] = exampleMap[r:r+searchKernelSize, c:c+searchKernelSize]
            
    return examplePatches

def gkern(kern_x, kern_y, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    """altered copy from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy"""

    # X
    interval = (2*nsig+1.)/(kern_x)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kern_x+1)
    kern1d_x = np.diff(st.norm.cdf(x))
    # Y
    interval = (2*nsig+1.)/(kern_y)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kern_y+1)
    kern1d_y = np.diff(st.norm.cdf(x))
    
    kernel_raw = np.sqrt(np.outer(kern1d_x, kern1d_y))
    kernel = kernel_raw/kernel_raw.sum()
    
    return kernel