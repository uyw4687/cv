#legacy
import math
import glob
import numpy as np
from PIL import Image
#NMS, Hough
from math import floor,ceil,sin,cos,pi

# parameters
datadir = './cv/hough'

'''1,2,3'''
sigma=1
threshold=0.2
rhoRes=1
thetaRes=math.pi/270
nLines=15
# '''4'''
# sigma=1.5
# threshold=0.4
# rhoRes=2
# thetaRes=math.pi/90
# nLines=25

cnt=1

'''
Igs     grayscale image
G       filter matrix
        shape (2n-1,2n-1) or ((filter_vertical,filter_horizontal), None) #legacy

Iconv   result
        shape : Igs.shape
'''
def ConvFilter(Igs, G):

    print("convolution")

    #legacy
    separable=False
    Gv=None
    if len(G)==2:
        separable=True
        Gv,G=G[0]

    row_img=len(Igs)
    col_img=len(Igs[0])
    size_filter=len(G)
    size_filter_oneway=size_filter//2+1
    pad_len=size_filter_oneway-1

    padded_img=np.zeros(( row_img+2*pad_len , col_img+2*pad_len )) #gs
    padded_img[ pad_len : row_img+pad_len , pad_len : col_img+pad_len ] = Igs
    
    '''corners'''
    padded_img[ : pad_len , : pad_len ] = np.full( (pad_len,pad_len) , Igs[0][0] )
    padded_img[ : pad_len , col_img+pad_len : col_img+2*pad_len ] = \
        np.full( (pad_len,pad_len) , Igs[0][col_img-1] )
    padded_img[ row_img+pad_len : row_img+2*pad_len , : pad_len ] = \
        np.full( (pad_len,pad_len) , Igs[row_img-1][0] )
    padded_img[ row_img+pad_len : row_img+2*pad_len , col_img+pad_len : col_img+2*pad_len ] = \
        np.full( (pad_len,pad_len) , Igs[row_img-1][col_img-1] )
    
    '''sides'''
    padded_img[ : pad_len , pad_len:col_img+pad_len ] = \
        np.repeat( Igs[np.newaxis, 0], pad_len, axis=0 )
    padded_img[ row_img+pad_len : row_img+2*pad_len , pad_len:col_img+pad_len ] = \
        np.repeat( Igs[np.newaxis, row_img-1], pad_len, axis=0 )
    padded_img[ pad_len : row_img+pad_len , : pad_len ] = \
        np.repeat( Igs[ : , 0:1], pad_len, axis=1 )
    padded_img[ pad_len : row_img+pad_len , col_img+pad_len : col_img+2*pad_len ] = \
        np.repeat( Igs[ : , col_img-1:col_img], pad_len, axis=1 )

    Iconv=np.zeros(( row_img , col_img )) #gs

    for i in range(row_img):
        for j in range(col_img):
            
            if not separable:
                mid=0
                for ki in range(size_filter):
                    for kj in range(size_filter):
                        mid+=(G[ki][kj]*padded_img[i+ki][j+kj])
                Iconv[i][j]=mid

            else: # separable
                mid=[]
                for col in range(size_filter):
                    mid.append(Gv.dot(padded_img[ i:i+size_filter , j+col ]))
                mid = np.array(mid,copy=False)
                Iconv[i][j] = G.dot(mid)

    return Iconv

'''
img     float
        [0,1]
'''
def to_img(img):

    img=(img*255).astype(np.uint8)
    return Image.fromarray(img)

def EdgeDetection(Igs, sigma):

    size_filter=int(round(sigma*6))
    if size_filter%2==0:
        size_filter+=1

    '''apply gaussian filter'''
    dev = size_filter//2
    x = np.linspace(-dev,dev,dev*2+1)
    x = np.exp(-x*x/2/sigma**2)
    x/=x.sum()
    filter_gaussian_sep=(x,x)
    Igauss=ConvFilter(Igs, (filter_gaussian_sep,None))
    
    sobel3x=np.asarray([[1,2,1],[-1,0,1]])
    sobel3y=np.asarray([[1,0,-1],[1,2,1]])

    Ix=ConvFilter(Igauss, (sobel3x,None))
    Iy=ConvFilter(Igauss, (sobel3y,None))
    Im=np.sqrt(Ix*Ix+Iy*Iy)
    Io=np.arctan(-Ix/Iy)

    to_img(Im).save(f'./edges_before_nms{cnt}.png')

    # non maximum suppression
    h,w = Igs.shape
    nmslen = 2
    Im_pad=np.zeros(( h+2*nmslen , w+2*nmslen ))
    Im_pad[ nmslen : h+nmslen , nmslen : w+nmslen ] = Im
    global threshold
    
    for i in range(nmslen,h+nmslen):
        for j in range(nmslen,w+nmslen):
            
            curr=Im_pad[i][j]
            #threshold
            if curr<threshold:
                Im[i-nmslen][j-nmslen]=0
                continue

            m_x=Ix[i-nmslen][j-nmslen]
            m_y=Iy[i-nmslen][j-nmslen]
            ori=-m_x/m_y
            left=right=0

            #vertical
            if abs(ori)>h:
                if curr!=Im_pad[i,j-nmslen:j+nmslen].max():
                    curr=0
            #horizontal
            elif abs(ori)<1/w:
                if curr!=Im_pad[i-nmslen:i+nmslen,j].max():
                    curr=0
            else:
                
                for k in range(1,nmslen+1):

                    #interpolation
                    if abs(m_x)>=abs(m_y):

                        val_bet=i-k/ori
                        floor_p=floor(val_bet)
                        ceil_p=ceil(val_bet)
                        dist_i=abs(val_bet-floor_p)
                        right=(1-dist_i)*Im_pad[floor_p][j+k]+dist_i*Im_pad[ceil_p][j+k]

                        val_bet=i+k/ori
                        floor_n=floor(val_bet)
                        ceil_n=ceil(val_bet)
                        left=dist_i*Im_pad[floor_n][j-k]+(1-dist_i)*Im_pad[ceil_n][j-k]

                    else:
                        
                        val_bet=j-k*ori
                        floor_p=floor(val_bet)
                        ceil_p=ceil(val_bet)
                        dist_i=abs(val_bet-floor_p)
                        right=(1-dist_i)*Im_pad[i+k][floor_p]+dist_i*Im_pad[i+k][ceil_p]

                        val_bet=j+k*ori
                        floor_n=floor(val_bet)
                        ceil_n=ceil(val_bet)
                        left=dist_i*Im_pad[i-k][floor_n]+(1-dist_i)*Im_pad[i-k][ceil_n]

                    if curr!=max(curr,right,left):
                        Im[i-nmslen][j-nmslen]=0
                        break

    # to_img(Igauss).save(f'./gauss{cnt}.png')
    to_img(Im).save(f'./edges{cnt}.png')

    return Im, Io, Ix, Iy

def HoughTransform(Im,threshold,rhoRes,thetaRes):

    if rhoRes*thetaRes==0:
        raise ValueError()

    '''rho=xcos(th)+ysin(th)'''
    h,w = Im.shape
    maxrho=np.sqrt(h**2+w**2)
    numrho=2*int(maxrho/rhoRes)+1
    numtheta=2*int(pi/2/thetaRes)+1
    H=[[0]*numtheta for i in range(numrho)]
    
    rho_mid=numrho//2

    for i in range(h):
        for j in range(w):

            if Im[i][j]<threshold:
                continue

            theta=-pi/2
            thetacnt=0
            thetathres=pi/2
            while True:
                if theta>=thetathres:
                    break
                rho=i*cos(theta)+j*sin(theta)
                H[rho_mid+int(rho/rhoRes)][thetacnt]+=1
                theta+=thetaRes
                thetacnt+=1
    H=np.asarray(H)

    global cnt
    to_img(H/H.max()).save(f'./hough_transform{cnt}.png')

    return H

def HoughLines(H,rhoRes,thetaRes,nLines):

    # non maximum suppression
    h,w = H.shape

    pad_h = max(int(round(h/nLines/10)),1)
    pad_w = max(int(round(w/nLines/10)),1)

    print(pad_h, pad_w)

    H_pad=np.zeros(( h+2*pad_h , w+2*pad_w ))
    H_pad[ pad_h : h+pad_h , pad_w : w+pad_w ] = H

    for i in range(pad_h,h+pad_h):
        for j in range(pad_w,w+pad_w):

            curr=H_pad[i][j]

            #threshold
            if curr<=1:
                continue
            
            if curr!=H_pad[ i-pad_h : i+pad_h , j-pad_w : j+pad_w ].max():
                H[i-pad_h][j-pad_w]=0
    
    mid_h, mid_w = h//2, w//2

    inds_flat=np.argpartition(H.flatten(),-nLines)[-nLines:]
    lRho=[]
    lTheta=[]
    for ind in inds_flat:
        ind_w = ind%w
        ind_h = ind//w
        lRho.append((ind_h-mid_h)*rhoRes)
        lTheta.append((ind_w-mid_w)*thetaRes)

    return lRho,lTheta

def HoughLineSegments(lRho,lTheta,Im,threshold):

    #legacy
    Igs=None
    if type(Im)==type(()):
        Igs,Im=Im
    Ilines = np.copy(Igs)
    Ilinesnob = np.zeros(Im.shape)
    lsintheta = np.sin(lTheta)
    lcostheta = np.cos(lTheta)
    lm = -lcostheta/lsintheta
    lc = lRho/lsintheta
    h,w = Im.shape
    
    chklen = 1

    Im_pad=np.zeros(( h+2*chklen , w+2*chklen ))
    Im_pad[ chklen : h+chklen , chklen : w+chklen ] = Im

    # for performance
    l=[{} for i in range(len(lRho))]
    for i,m in enumerate(lm):
        started=False
        continuous=False

        #vertical
        if abs(m)>h:
            x=int(round(lRho[i]/lcostheta[i]))
            x=min(x,h-1)
            x=max(x,0)
            for y in range(w):
                Ilinesnob[x][y]=Ilines[x][y]=1
                if Im_pad[ x : x+2*chklen , y+chklen ].max() > threshold:
                    if not continuous:
                        continuous=True
                    elif not started:
                        l[i]['start']=(x,y-1)
                        l[i]['end']=(x,y)
                        started=True
                    else:
                        l[i]['end']=(x,y)
                else:
                    continuous=False
        #horizontal
        elif abs(m)<1/w:
            y=int(round(lRho[i]/lsintheta[i]))
            y=min(y,w-1)
            y=max(y,0)
            for x in range(h):
                Ilinesnob[x][y]=Ilines[x][y]=1
                if Im_pad[ x+chklen , y : y+2*chklen ].max() > threshold:
                    if not continuous:
                        continuous=True
                    elif not started:
                        l[i]['start']=(x-1,y)
                        l[i]['end']=(x,y)
                        started=True
                    else:
                        l[i]['end']=(x,y)
                else:
                    continuous=False
        else:

            prev=None
            if abs(m)>=1:
                for y in range(w):
                    x=int(round((y-lc[i])/m))
                    x=min(x,h-1)
                    x=max(x,0)
                    Ilinesnob[x][y]=Ilines[x][y]=1

                    pool=[Im_pad[x+chklen][y+chklen]]
                    for k in range(1,chklen+1):
                        yf=y-k/m
                        pool.append(Im_pad[x+k+chklen][int(round(yf))+chklen])
                        yf=y+k/m
                        pool.append(Im_pad[x-k+chklen][int(round(yf))+chklen])

                    if max(pool)>threshold:
                        if not continuous:
                            continuous=True
                            prev=(x,y)
                        elif not started:
                            l[i]['start']=prev
                            l[i]['end']=(x,y)
                            started=True
                        else:
                            l[i]['end']=(x,y)
                    else:
                        continuous=False
            else:
                for x in range(h):
                    y=int(round(m*x+lc[i]))
                    y=min(y,w-1)
                    y=max(y,0)
                    Ilinesnob[x][y]=Ilines[x][y]=1

                    pool=[Im_pad[x+chklen][y+chklen]]
                    for k in range(1,chklen+1):
                        xf=x-m
                        pool.append(Im_pad[int(round(xf))+chklen][y+k+chklen])
                        xf=x+m
                        pool.append(Im_pad[int(round(xf))+chklen][y-k+chklen])

                    if max(pool)>threshold:
                        if not continuous:
                            continuous=True
                            prev=(x,y)
                        elif not started:
                            l[i]['start']=prev
                            l[i]['end']=(x,y)
                            started=True
                        else:
                            l[i]['end']=(x,y)
                    else:
                        continuous=False

    global cnt
    to_img(Ilines).save(f'./img_houghlines{cnt}.png')
    # to_img(Ilinesnob).save(f'./img_houghlinesnob{cnt}.png')

    Ilinesseg = np.copy(Igs)
    Ilinessegnob = np.zeros(Im.shape)
    for i,m in enumerate(lm):
        if len(l[i])==0:
            continue
        start=l[i]['start']
        end=l[i]['end']
        if abs(m)>h:
            x=int(round(lRho[i]/lcostheta[i]))
            x=min(x,h-1)
            x=max(x,0)
            if start[1]>end[1]:
                start,end = end,start
            for y in range(start[1],end[1]+1):
                Ilinesseg[x][y]=1
                Ilinessegnob[x][y]=1
        elif abs(m)<1/w:
            y=int(round(lRho[i]/lsintheta[i]))
            y=min(y,w-1)
            y=max(y,0)
            if start[0]>end[0]:
                start,end = end,start
            for x in range(start[0],end[0]+1):
                Ilinesseg[x][y]=1
                Ilinessegnob[x][y]=1
        else:
            if abs(m)>=1:
                if start[1]>end[1]:
                    start,end = end,start
                for y in range(start[1],end[1]+1):
                    x=int(round((y-lc[i])/m))
                    x=min(x,h-1)
                    x=max(x,0)
                    Ilinesseg[x][y]=1
                    Ilinessegnob[x][y]=1
            else:
                if start[0]>end[0]:
                    start,end = end,start
                for x in range(start[0],end[0]+1):
                    y=int(round(m*x+lc[i]))
                    y=min(y,w-1)
                    y=max(y,0)
                    Ilinesseg[x][y]=1
                    Ilinessegnob[x][y]=1

    to_img(Ilinesseg).save(f'./hough_line_segs{cnt}.png')
    # to_img(Ilinessegnob).save(f'./hough_line_segsnob{cnt}.png')
    cnt+=1

    return l

def main():

    for img_path in glob.glob(datadir+'/*.png'):
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma)
        H= HoughTransform(Im,threshold, rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)

        l = HoughLineSegments(lRho, lTheta, (Igs,Im), threshold)

def _test(num,all=False):
    Igs=np.asarray([[1,2,3],[4,5,6],[7,8,9]])
    print(ConvFilter(Igs,np.asarray([[1]])))
    print(ConvFilter(Igs,np.asarray([[9,8,7],[4,5,6],[3,2,1]])))
    print(ConvFilter(Igs,np.asarray([[9,8,7,6,5],[2,3,4,5,6],[5,4,3,2,1],[1,2,3,4,5],[5,6,7,8,9]])))
    print(ConvFilter(Igs, (np.asarray([2]),None) ))
    print(ConvFilter(Igs, (np.asarray([1,3,2]),None) ))

if __name__ == '__main__':
    # _test()
    main()