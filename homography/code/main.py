import math
import numpy as np
from PIL import Image
from heapq import heappush,heappop
from math import ceil,floor,pi,sin,cos

datadir='../../data/'
resultdir='../result/'

def get_gaussian_filter_1d(size):

    dev = size//2
    x = np.linspace(-dev,dev,size)
    sigma = size/6
    x = np.exp(-x*x/2/sigma**2)
    return x/x.sum()

ft_4 = get_gaussian_filter_1d(4).reshape((4,1))
ftm_4 = ft_4.dot(ft_4.T)

nr = 10
rl = 2*np.pi/nr

xs = 13
ndots = 30 #per iteration

levels = 7
g_size = 3

sigma=1
threshold=1.5
rhoRes=1
thetaRes=math.pi/60
nLines=4

def stretch(igs,horizontal=True):

    h,w,c = igs.shape
    
    igs_tch = None
    if horizontal:
        igs_tch = np.zeros((h,2*w,c))
    else:
        igs_tch = np.zeros((2*h,w,c))

    print("from",igs.shape)
    print("to  ",igs_tch.shape)

    if horizontal:
        for r in range(h):
            for c in range(w):
                igs_tch[r][2*c]=np.copy(igs[r][c])

        for r in range(h):
            for c in range(w-1):
                igs_tch[r][2*c+1]=(igs_tch[r][2*c]+igs_tch[r][2*c+2])/2
            igs_tch[r][2*w-1]=np.copy(igs_tch[r][2*w-2])
    
    else: #vertical
        for c in range(w):
            for r in range(h):
                igs_tch[2*r][c]=np.copy(igs[r][c])

        for c in range(w):
            for r in range(h-1):
                igs_tch[2*r+1][c]=(igs_tch[2*r][c]+igs_tch[2*r+2][c])/2
            igs_tch[2*h-1][c]=np.copy(igs_tch[2*r-2][c])

    return igs_tch

def compute_h(p1, p2):

    A=np.zeros((2*len(p2),9))
    for i, (x,y) in enumerate(p2):

        xp, yp = p1[i]
        
        A[2*i][0] = x
        A[2*i][1] = y
        A[2*i][2] = 1
        A[2*i][6] = -x*xp
        A[2*i][7] = -y*xp
        A[2*i][8] = -xp

        A[2*i+1][3] = x
        A[2*i+1][4] = y
        A[2*i+1][5] = 1
        A[2*i+1][6] = -x*yp
        A[2*i+1][7] = -y*yp
        A[2*i+1][8] = -yp

    _,_,v = np.linalg.svd(A)
    
    H=v[-1].reshape((3,3))
    return H

def normalize_matrix(p):

    mx,my = np.average(p,axis=0)
    sx,sy = np.std(p,axis=0)
    s=np.sqrt((sx**2+sy**2)/2)

    T = np.zeros((3,3))
    T[0][0] = 1/s
    T[1][1] = 1/s
    T[0][2] = -mx/s
    T[1][2] = -my/s
    T[2][2] = 1

    return T

def compute_h_norm(p1, p2):

    Tp = normalize_matrix(p1)
    T = normalize_matrix(p2)

    newp1 = np.array([Tp.dot([*p,1])[:2] for p in p1])
    newp2 = np.array([T.dot([*p,1])[:2] for p in p2])

    Hm = compute_h(newp1, newp2)
    H = np.linalg.inv(Tp).dot(Hm).dot(T)

    return H

def conv_sep(igs, Gs):

    print("conv")
    G,Gv=Gs
    
    h = w = n = 1
    if len(igs.shape)==2:
        h,w = igs.shape
        igs = igs.reshape(h,w,1)
    else:
        h, w, n = igs.shape

    igs_conv = np.zeros((h,w,n))
    size = len(G)
    pad_len = size//2

    ccd = np.zeros((h+2*pad_len,w,n))
    pdd_igs = np.zeros(( h+2*pad_len , w+2*pad_len, n ))
    pdd_igs[ pad_len : h+pad_len , pad_len : w+pad_len ] = igs
    
    '''corners'''
    pdd_igs[ : pad_len , : pad_len ] = np.full( (pad_len,pad_len,n) , igs[0][0] )
    pdd_igs[ : pad_len , w+pad_len : w+2*pad_len ] = \
        np.full( (pad_len,pad_len,n) , igs[0][w-1] )
    pdd_igs[ h+pad_len : h+2*pad_len , : pad_len ] = \
        np.full( (pad_len,pad_len,n) , igs[h-1][0] )
    pdd_igs[ h+pad_len : h+2*pad_len , w+pad_len : w+2*pad_len ] = \
        np.full( (pad_len,pad_len,n) , igs[h-1][w-1] )
    
    '''sides'''
    pdd_igs[ : pad_len , pad_len:w+pad_len ] = \
        np.repeat( igs[np.newaxis, 0], pad_len, axis=0 )
    pdd_igs[ h+pad_len : h+2*pad_len , pad_len:w+pad_len ] = \
        np.repeat( igs[np.newaxis, h-1], pad_len, axis=0 )
    pdd_igs[ pad_len : h+pad_len , : pad_len ] = \
        np.repeat( igs[ : , 0:1], pad_len, axis=1 )
    pdd_igs[ pad_len : h+pad_len , w+pad_len : w+2*pad_len ] = \
        np.repeat( igs[ : , w-1:w], pad_len, axis=1 )

    for i in range(h):
        for j in range(w):
            ccd[i+pad_len][j] = G.dot(pdd_igs[i+pad_len,j:j+size])
    
    for i in range(h):
        for j in range(w):
            igs_conv[i][j] = Gv.dot(ccd[i:i+size,j])

    if n==1:
        igs_conv = igs_conv.reshape((h,w))

    return igs_conv

def apply_gaussian(igs, size):

    print("appl gauss", size)
    gf1d = get_gaussian_filter_1d(size)
    igs_ga = conv_sep(igs, np.asarray((gf1d,gf1d)))

    return igs_ga

def cors_lap(igs,xs,ndots):

    h,w,_ = igs.shape

    igs = np.linalg.norm(igs,axis=2)
    igs = apply_gaussian(igs, 3)
    al = [igs]
    ps = set()
    
    for s in range(3,xs+1,2):
        al.append(apply_gaussian(igs, s))
        l = np.abs(al[-1]-al[-2])

        th = l.max()*0.8

        pad_l = s//2
        l_pad=np.zeros(( h+2*pad_l , w+2*pad_l ))
        l_pad[ pad_l : h+pad_l , pad_l : w+pad_l ] = l

        for i in range(pad_l,h+pad_l):
            for j in range(pad_l,w+pad_l):
                curr = l_pad[i][j]
                
                if curr<=th or curr!=l_pad[ i-pad_l : i+pad_l+1 , j-pad_l : j+pad_l+1 ].max():
                    l[i-pad_l][j-pad_l]=0

        inds_flat=np.argpartition(l.flatten(),-ndots)[-ndots:]
        for ind in inds_flat:
            x=ind//w
            y=ind%w
            if 20<=x<=h-20 and 20<=y<=w-20:
                ps.add((x,y))

    return list(ps)

def hs(io,sx,sy):

    hs = []
    for i in range(4):
    
        hms = []
        for j in range(4):

            hm = [0]*nr
            for ii in range(4):
                for jj in range(4):
                    co = io[sx+4*i+ii][sy+4*j+jj]
                    hm[co] += ftm_4[ii][jj]
            
            hms.append(hm)

        hs.append(hms)

    return np.array(hs)

def nperm(p,l):

    if p[0]==l-4:
        return None
    
    for i in range(4):
        if p[3-i]!=l-1-i:
            break
    p = p[:3-i]+list(range(p[3-i]+1,p[3-i]+1+i+1))

    return p

def rans(p_in, p_ref):

    lp = len(p_in)
    p_in = np.array(p_in)
    p_ref = np.array(p_ref)

    p = list(range(4))
    hs = []
    while p!=None:
    
        cnt=0
        H = compute_h_norm(p_ref[p], p_in[p])

        for i in range(lp):
            if i in p:
                continue
            a = p_in[i]
            b = p_ref[i]
            x,y,s = H.dot(np.array([[*a,1]]).T)
            ex = (b[0]-x/s)[0]
            ey = (b[1]-y/s)[0]

            if ex*ex+ey*ey <= 8:
                cnt+=1

        heappush(hs,(-cnt,p))
        p=nperm(p,lp)

    _, p = heappop(hs)
    H = compute_h_norm(p_ref[p], p_in[p])

    n_pin = [*p_in[p]]
    n_pref = [*p_ref[p]]

    for i in range(lp):
        if i in p:
            continue
        a = p_in[i]
        b = p_ref[i]
        x,y,s = H.dot(np.array([[*a,1]]).T)
        ex = (b[0]-x/s)[0]
        ey = (b[1]-y/s)[0]
        if ex*ex+ey*ey <= 8:
            n_pin.append(a)
            n_pref.append(b)

    return np.array(n_pin), np.array(n_pref)

def set_cor_mosaic(igs_in, igs_ref):

    ls = cors_lap(igs_in,xs,ndots)
    lsr = cors_lap(igs_ref,xs,ndots)

    ga_size = 3
    in_ga=apply_gaussian(np.linalg.norm(igs_in,axis=2), ga_size)
    ir_ga=apply_gaussian(np.linalg.norm(igs_ref,axis=2), ga_size)

    sobel3x=np.array([[-1,0,1],[1,2,1]])
    sobel3y=np.array([[1,2,1],[1,0,-1]])

    in_x = conv_sep(in_ga, sobel3x)
    in_y = conv_sep(in_ga, sobel3y)
    in_o = np.arctan(-in_x/in_y) + np.pi/2
    in_o[in_y < 0] += np.pi

    ir_x = conv_sep(ir_ga, sobel3x)
    ir_y = conv_sep(ir_ga, sobel3y)
    ir_o = np.arctan(-ir_x/ir_y) + np.pi/2
    ir_o[ir_y < 0] += np.pi

    in_o = (in_o/rl).astype(np.uint8)
    ir_o = (ir_o/rl).astype(np.uint8)

    arm = 8

    pad_len=arm
    h,w = in_o.shape
    pdd_ino = np.zeros(( h+2*pad_len , w+2*pad_len )).astype(np.uint8)
    pdd_ino[ pad_len : h+pad_len , pad_len : w+pad_len ] = in_o
    pdd_iro = np.zeros(( h+2*pad_len , w+2*pad_len )).astype(np.uint8)
    pdd_iro[ pad_len : h+pad_len , pad_len : w+pad_len ] = ir_o

    ft = get_gaussian_filter_1d(16)

    p_in = []
    p_ref = []

    for pn in ls:

        hn = hs(pdd_ino,pn[0],pn[1])
        
        r = ft.dot(pdd_ino[pn[0]:pn[0]+2*arm,pn[1]:pn[1]+2*arm])
        on = ft.dot(r)
        minv=100
        minp=None

        for p in lsr:

            hr = hs(pdd_iro,p[0],p[1])

            r = ft.dot(pdd_iro[p[0]:p[0]+2*arm,p[1]:p[1]+2*arm])
            onr = ft.dot(r)

            if abs(onr-on)>=1:

                go = round(onr-on)
                if go < 0:
                    go += nr
                for i in range(4):
                    for j in range(4):
                        ca = hr[i][j]
                        hr[i][j]=np.concatenate((ca[go:],ca[:go]))

            d = hn-hr
            ds = np.sum(d*d,axis=2)
            e = (ds*ftm_4).sum()

            if minv > e:
                minv = e
                minp = p

        if minv < 0.3:
            p_in.append(pn)
            p_ref.append(minp)

    if len(p_in) < 4:
        raise Exception

    p_in, p_ref = rans(p_in, p_ref)

    print(p_in)
    print(p_ref)

    return p_in, p_ref

def upsample(arr):
    h, w, _ = arr.shape
    image = Image.fromarray(arr)
    image = image.resize((w*2, h*2), Image.ANTIALIAS)
    return np.asarray(image)

def downsample(arr):
    h, w, _ = arr.shape
    image = Image.fromarray(arr)
    image = image.resize((round(w/2), round(h/2)), Image.ANTIALIAS)
    return np.asarray(image)

def build_gaussian_pyramid(pixel_grid, size, levels):
    pyramid = []
    curr_grid=pixel_grid
    for _ in range(levels):
        curr_grid = apply_gaussian(curr_grid, size).astype(np.uint8)
        pyramid.append(curr_grid)
        curr_grid = downsample(curr_grid)
    return pyramid

def build_laplacian_pyramid(gaussian_pyramid):

    num_img = len(gaussian_pyramid)
    
    #np.int16
    laplacian_pyramid = []
    
    for i in range(num_img-1):
        laplacian_pyramid.append(gaussian_pyramid[i].astype(np.int16)-upsample(gaussian_pyramid[i+1]))

    return laplacian_pyramid

def apply_mask(img_left, img_right, mask):
    # mask in [0,1]
    return (1-mask)*img_left + mask*img_right

def blend_images_from_pyramids(gaussian_pyramid_left, gaussian_pyramid_right, laplacian_pyramid_left, laplacian_pyramid_right, gaussian_pyramid_mask):
    depth = len(gaussian_pyramid_left)
    laplacian_pyramid_blended = []
    for i in range(depth-1):
        laplacian_pyramid_blended.append(apply_mask(laplacian_pyramid_left[i], laplacian_pyramid_right[i], gaussian_pyramid_mask[i]/255))

    curr_image = apply_mask(upsample(gaussian_pyramid_left[depth-1]),
                        upsample(gaussian_pyramid_right[depth-1]),
                        gaussian_pyramid_mask[depth-2]/255)
    curr_image = np.clip((curr_image + laplacian_pyramid_blended[depth-2]),0,255).astype(np.uint8)
    
    for i in range(depth-2):
        curr_image = np.clip((upsample(curr_image) + laplacian_pyramid_blended[depth-1-2-i]),0,255).astype(np.uint8)

    return curr_image

def blend_images(left_image, right_image, mask):
    
    print("building pyramids")
    gaussian_pyramid_left = build_gaussian_pyramid(left_image, g_size, levels)
    laplacian_pyramid_left = build_laplacian_pyramid(gaussian_pyramid_left)
    gaussian_pyramid_right = build_gaussian_pyramid(right_image, g_size, levels)
    laplacian_pyramid_right = build_laplacian_pyramid(gaussian_pyramid_right)

    gaussian_pyramid_mask = build_gaussian_pyramid(mask, g_size, levels)

    print('blending images')
    return blend_images_from_pyramids(gaussian_pyramid_left, gaussian_pyramid_right, laplacian_pyramid_left, laplacian_pyramid_right, gaussian_pyramid_mask)

def left(m,p,r):
    xr,yr = r
    xa,ya = p
    xb = m*(ya-yr)+xr
    if xa < xb:
        return True
    return False

def warp_image(igs_in, igs_ref, H):
    
    h,w,n = igs_in.shape

    hi = np.linalg.inv(H)

    cs = [(0,0),(h-1,0),(0,w-1),(h-1,w-1)]
    ws = []
    for x in cs:
        x,y,s=H.dot(np.array([[*x,1]]).T)
        ws.append(((x/s)[0],(y/s)[0]))
    print(ws)
    print()

    ws = np.array(ws)
    xmn=ws[:,0].min()
    xmx=ws[:,0].max()
    ymn=ws[:,1].min()
    ymx=ws[:,1].max()
    print(xmn,xmx)
    print(ymn,ymx)

    wh=int(xmx-xmn)
    ww=int(ymx-ymn)
    igs_warp = np.zeros((wh,ww,n))
    print(igs_warp.shape)

    for i in range(wh):
        for j in range(ww):
            p = (i+xmn,j+ymn)
            x,y,s=hi.dot(np.array([[*p,1]]).T)
            x,y=x/s,y/s
            if 0<=x<=h-1 and 0<=y<=w-1:
                sx,sy=int(x),int(y)
                dx=1-(x-sx)
                dy=1-(y-sy)

                if not (sx+1>h-1 or sy+1>w-1):
                    igs_warp[i][j]=dx*dy*igs_in[sx][sy]
                    igs_warp[i][j]+=(1-dx)*dy*igs_in[sx+1][sy]
                    igs_warp[i][j]+=dx*(1-dy)*igs_in[sx][sy+1]
                    igs_warp[i][j]+=(1-dx)*(1-dy)*igs_in[sx+1][sy+1]
                else:
                    if sx+1>h-1 and sy+1>w-1:
                        igs_warp[i][j]=igs_in[sx][sy]
                    else:
                        if sx+1>h-1:
                            igs_warp[i][j]=dy*igs_in[sx][sy]
                            igs_warp[i][j]+=(1-dy)*igs_in[sx][sy+1]
                        else:
                            igs_warp[i][j]+=dx*igs_in[sx][sy]
                            igs_warp[i][j]+=(1-dx)*igs_in[sx+1][sy]

    h_r,w_r,_ = igs_ref.shape
    xmx = max(xmx,h_r)
    wh=int(xmx-xmn)
    ymx = max(ymx,w_r)
    ww=int(ymx-ymn)

    den = 2**levels
    if wh%den != 0:
        wh=(floor(wh/den)+1)*den
    if ww%den != 0:
        ww=(floor(ww/den)+1)*den

    h_w,w_w,_ = igs_warp.shape
    igs_merge_l = np.zeros((wh,ww,n))
    igs_merge_l[:h_w,:w_w] = igs_warp
    print(igs_merge_l.shape)

    h_m,w_m,_ = igs_merge_l.shape

    si, sj = int(0-xmn), int(0-ymn)

    jt = jb = -1
    for j in range(w_m-1,-1,-1):
        if np.linalg.norm(igs_merge_l[si][j])!=0:
            jt=j
            break
    for j in range(w_m-1,-1,-1):
        if np.linalg.norm(igs_merge_l[si+h_r][j])!=0:
            jb=j
            break
    print('jt',jt,'jb',jb) 

    for i in range(si,int(h_r-xmn)):
        for j in range(sj,min(jt+200,int(w_r-ymn))):
            if np.linalg.norm(igs_merge_l[i][j])==0:
                igs_merge_l[i][j] = igs_ref[i-si][j-sj]

    jv = min(jb+150,w_m)
    igs_merge_l[:,jv:] = np.full((h_m,w_m-jv,n),igs_merge_l[:,jv:jv+1])

    igs_merge_r = np.zeros(igs_merge_l.shape)
    igs_merge_r[:h_w,:w_w] = igs_warp
    igs_merge_r[int(0-xmn):int(h_r-xmn),int(0-ymn):int(w_r-ymn)] = igs_ref
                
    mask = np.zeros(igs_merge_l.shape).astype(np.uint8)
    m = (si-(si+h_r))/(jt-jb)
    r = (si,jt)
    for i in range(h_m):
        for j in range(w_m):
            if not left(m,(i,j),r):
                mask[i,j:] = np.full((1,w_m-j,n),255)
                break

    igs_merge = blend_images(igs_merge_l, igs_merge_r, mask)

    return igs_warp, igs_merge

def EdgeDetection(Igs, sigma):

    size_filter=round(sigma*6)
    if size_filter%2==0:
        size_filter+=1

    dev = size_filter//2	
    x = np.linspace(-dev,dev,dev*2+1)	
    x = np.exp(-x*x/2/sigma**2)	
    x/=x.sum()	
    filter_gaussian_sep=(x,x)	
    Igauss=conv_sep(Igs, filter_gaussian_sep)

    sobel3x=np.asarray([[1,2,1],[-1,0,1]])
    sobel3y=np.asarray([[1,0,-1],[1,2,1]])

    Ix=conv_sep(Igauss, sobel3x)
    Iy=conv_sep(Igauss, sobel3y)
    Im=np.sqrt(Ix*Ix+Iy*Iy)

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

    return Im

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

    return H

def HoughLines(H,rhoRes,thetaRes,nLines):

    # non maximum suppression
    h,w = H.shape

    pad_h = max(round(h/nLines/5),1)
    pad_w = max(round(w/nLines/5),1)
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

def get_votes(lRho,lTheta,Im,threshold):

    lsintheta = np.sin(lTheta)
    lcostheta = np.cos(lTheta)
    lm = -lcostheta/lsintheta
    lc = lRho/lsintheta
    h,w = Im.shape

    votes = np.zeros(Im.shape)
    
    chklen = 1
    Im_pad=np.zeros(( h+2*chklen , w+2*chklen ))
    Im_pad[ chklen : h+chklen , chklen : w+chklen ] = Im

    for i,m in enumerate(lm):

        #vertical
        if abs(m)>h:
            x=round(lRho[i]/lcostheta[i])
            x=min(x,h-1)
            x=max(x,0)
            for y in range(w):
                votes[x][y]+=1
        #horizontal
        elif abs(m)<1/w:
            y=round(lRho[i]/lsintheta[i])
            y=min(y,w-1)
            y=max(y,0)
            for x in range(h):
                votes[x][y]+=1
        else:
            if abs(m)>=1:
                for y in range(w):
                    x=round((y-lc[i])/m)
                    x=min(x,h-1)
                    x=max(x,0)
                    votes[x][y]+=1
            else:
                for x in range(h):
                    y=round(m*x+lc[i])
                    y=min(y,w-1)
                    y=max(y,0)
                    votes[x][y]+=1

    return votes

def set_cor_rec(votes):

    c_in = []
    c_ref = []

    h,w = votes.shape
    for i in range(h):
        for j in range(w):
            if votes[i][j]==2:
                c_in.append((i,j))
                c_ref.append((i,j))
    
    c_ref[1]=(c_ref[0][0],c_ref[1][1])
    c_ref[2]=(c_ref[3][0],c_ref[1][1])

    # k1=c_ref[1][0]
    # k2=c_ref[2][0]
    # c_ref[1]=(c_ref[0][0],c_ref[1][1])
    # c_ref[2]=(c_ref[3][0],c_ref[1][1])
    # c_ref[0]=(k1,c_ref[0][1])
    # c_ref[3]=(k2,c_ref[3][1])

    return c_in, c_ref

def rectify(igs, p1, p2):
    
    h,w,n = igs.shape
    H = compute_h_norm(p2, p1)
    hi = np.linalg.inv(H)

    cs = [(0,0),(h-1,0),(0,w-1),(h-1,w-1)]
    ws = []
    for x in cs:
        x,y,s=H.dot(np.array([[*x,1]]).T)
        ws.append(((x/s)[0],(y/s)[0]))
    print(ws)
    print()

    ws = np.array(ws)
    xmn=ws[:,0].min()
    xmx=ws[:,0].max()
    ymn=ws[:,1].min()
    ymx=ws[:,1].max()
    print(xmn,xmx)
    print(ymn,ymx)

    wh=int(xmx-xmn)
    ww=int(ymx-ymn)
    igs_rec = np.zeros((wh,ww,n))
    print(igs_rec.shape)
    
    for i in range(wh):
        for j in range(ww):
            p = (i+xmn,j+ymn)
            x,y,s=hi.dot(np.array([[*p,1]]).T)
            x,y=x/s,y/s
            if 0<=x<=h-1 and 0<=y<=w-1:
                sx,sy=int(x),int(y)
                dx=1-(x-sx)
                dy=1-(y-sy)

                if not (sx+1>h-1 or sy+1>w-1):
                    igs_rec[i][j]=dx*dy*igs[sx][sy]
                    igs_rec[i][j]+=(1-dx)*dy*igs[sx+1][sy]
                    igs_rec[i][j]+=dx*(1-dy)*igs[sx][sy+1]
                    igs_rec[i][j]+=(1-dx)*(1-dy)*igs[sx+1][sy+1]
                else:
                    if sx+1>h-1 and sy+1>w-1:
                        igs_rec[i][j]=igs[sx][sy]
                    else:
                        if sx+1>h-1:
                            igs_rec[i][j]=dy*igs[sx][sy]
                            igs_rec[i][j]+=(1-dy)*igs[sx][sy+1]
                        else:
                            igs_rec[i][j]+=dx*igs[sx][sy]
                            igs_rec[i][j]+=(1-dx)*igs[sx+1][sy]

    return igs_rec

def main():

    ##############
    # step 0: stretching
    ##############

    # read images
    img_o = Image.open(datadir+'toys.jpg').convert('RGB')
    igs_o = np.array(img_o)

    horizontal = True
    igs_tch = stretch(igs_o, horizontal)
    img_tch = Image.fromarray(igs_tch.astype(np.uint8))

    suffix = 'horizontal' if horizontal else 'vertical'
    img_tch.save(resultdir+'toys_stretched_'+suffix+'.png')

    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open(datadir+'porto1.png').convert('RGB')
    img_ref = Image.open(datadir+'porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic(igs_in, igs_ref)

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)

    ls = 0
    for i, x in enumerate(p_in):
        ans = p_ref[i]
        x,y,s=H.dot(np.array([[*x,1]]).T)
        x,y,s=x[0],y[0],s[0]
        ex,ey = ans[0]-x/s,ans[1]-y/s
        ls+=(ex**2+ey**2)
        print(x/s,y/s,ex,ey)
    print(ls)

    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save(resultdir+'porto1_warped.png')
    img_merge.save(resultdir+'porto_merged.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open(datadir+'iphone.png').convert('L')
    igs_rec = np.array(img_rec)
    Igs = igs_rec / 255.

    Im = EdgeDetection(Igs, sigma)
    H = HoughTransform(Im,threshold, rhoRes, thetaRes)
    lRho,lTheta = HoughLines(H,rhoRes,thetaRes,nLines)
    votes = get_votes(lRho, lTheta, Im, threshold)

    img_rec = Image.open(datadir+'iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)
    c_in, c_ref = set_cor_rec(votes)
    igs_rec = rectify(igs_rec, c_in, c_ref)
    
    img_rec_output = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec_output.save(resultdir+'iphone_rectified.png')

if __name__ == '__main__':
    main()
