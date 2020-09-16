from numba import jit
import numpy as np
import warnings
import cv2

# Ignore Numba warnings
warnings.filterwarnings('ignore')

def wrappedPhase(imgs_x, imgs_y, delta):
    sumIsin_x, sumIcos_x = 0, 0
    sumIsin_y, sumIcos_y = 0, 0
    for k, (nIx, nIy) in enumerate(zip(imgs_x, imgs_y)):
        Ix = cv2.imread(nIx, 0)
        sumIsin_x += Ix*np.sin(delta[k])
        sumIcos_x += Ix*np.cos(delta[k])
        
        Iy = cv2.imread(nIy, 0)
        sumIsin_y += Iy*np.sin(delta[k])
        sumIcos_y += Iy*np.cos(delta[k])
    
    phi_x = -np.arctan2(sumIsin_x, sumIcos_x)
    phi_y = -np.arctan2(sumIsin_y, sumIcos_y)
    
    return phi_x, phi_y

def seedPoint(fn_clx, fn_cly, mask):
    clx = cv2.imread(fn_clx, 0)
    cly = cv2.imread(fn_cly, 0)
    
    # Estimate vertical line
    _, bw = cv2.threshold(np.uint8(clx*mask),0,255,
                          cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    y, x = np.where(bw)
    c = cv2.fitLine(np.c_[x, y], cv2.DIST_L2, 0, 0.01, 0.01)
    l1 = np.cross(np.r_[c[2],c[3],1], np.r_[c[2]-2,-2*c[1]/c[0]+c[3],1])
    
    
    # Estimate horizontal line
    _, bw = cv2.threshold(np.uint8(cly*mask),0,255,
                          cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    y, x = np.where(bw)
    c = cv2.fitLine(np.c_[x, y], cv2.DIST_L2, 0, 0.01, 0.01)
    l2 = np.cross(np.r_[c[2],c[3],1], np.r_[c[2]-2,-2*c[1]/c[0]+c[3],1])
    
    # Estimate intersection point
    p = np.cross(l1,l2)
    return p[:2]/p[-1]

@jit('double[:,:](double[:,:], float32[:], int32[:,:])', cache=True)
def spatialUnwrap(phased, p0, mask):
    # Convert initial point of unwrapping to int32
    p0 = np.round(p0).astype(np.int32)
    
    # Original discontinuous phase shape
    h, w = phased.shape
    
    # Zero-pad array edges and update p0 to this change
    mask = np.pad(mask, ((1,1) , (1,1)), 'constant')
    phased = np.pad(phased, ((1,1) , (1,1)), 'constant')
    p0 += 1

    # Initialize (unwrapped) continuous phase with the wrapped
    phasec = phased.copy()
    
    # Initialize array with the position of unwrapped points
    XY = np.zeros([np.sum(mask),2], np.int32)
    XY[0] = p0 # The first point is p0
    # Remove p0 from the mask
    mask[p0[1],p0[0]] = 0

    # Estimate final continuous phase
    phasec = _8neighbors_unwrap(phased, phasec, mask, p0, XY)
    
    return phasec[1:h+1,1:w+1] # Array without zero-padded edges

@jit('double[:,:](double[:,:], double[:,:], int32[:,:], int32[:], int32[:,:])',
     nopython=True, cache=True)
def _8neighbors_unwrap(phased, phasec, mask, p0, XY):
    xo = [-1,-1,-1,0,0,1,1,1] # x offset
    yo = [-1,0,1,-1,1,-1,0,1] # y offset

    cont, cont1, opct = 0, 0, 1
    while opct:
        # Continuous and discontinuous phase values in p0
        PCI = phasec[p0[1],p0[0]]
        PDI = phased[p0[1],p0[0]]
        
        # Unwrap the 8-neighbors of p0
        for i in range(8):
            # Move p0 with xo and yo
            px = xo[i] + p0[0]
            py = yo[i] + p0[1]
            
            # Check if point is within the mask
            if mask[py,px]:
                # Wrapped phase at the point neighboring to p0
                PDC = phased[py,px]
                D = (PDC-PDI)/(2*np.pi)
                
                # Unwrapp (px, py)
                phasec[py,px] = PCI+2*np.pi*(D-round(D))

                # Save unwrapped point (px, py)
                XY[cont1+1,0] = px
                XY[cont1+1,1] = py
                cont1 += 1 # Count a new point
                
                # Remove (px, py) from the mask
                mask[py,px] = 0
        
        # Count a while loop completed
        cont += 1
        # Check stop criteria
        if cont > cont1:
            opct = 0
        else:
            # Update p0
            p0 = XY[cont,:]
    
    return phasec

def polyfit2D(phasec, mask, n):
    y,x = np.where(mask)
    
    A = np.zeros([len(x), (n+1)*(n+2)//2])
    col = -1
    for i in range(n+1):
        for j in range(i+1):
            col += 1
            A[:, col] = x**(i-j) * y**j
    
    return np.linalg.inv(A.T @ A) @ A.T @ phasec[mask==1]

def polyval2D(x, y, c, n):
    if isinstance(x, np.ndarray):
        A = np.zeros([len(x), (n+1)*(n+2)//2])
    else:
        A = np.zeros([1, (n+1)*(n+2)//2])
        
    col = -1
    for i in range(n+1):
        for j in range(i+1):
            col += 1
            A[:, col] = x**(i-j) * y**j
    
    return A @ c