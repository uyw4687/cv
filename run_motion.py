import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

ep = 0.5

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.

    dxx,dyx,dxy,dyy,dx,dy = p
    A = np.array( [ [1+dxx,dxy,dx], [dyx,1+dyy,dy], [0,0,1] ] )
    ai = np.linalg.inv(A)

    h,w = img2.shape

    cs = [(0,0),(h-1,0),(0,w-1),(h-1,w-1)]
    ws = []
    for x in cs:
        x,y,_=A.dot(np.array([[*x,1]]).T)
        ws.append((x[0],y[0]))
    print(ws)
    print()

    # ws = np.array(ws)
    # xmn=ws[:,0].min()
    # xmx=ws[:,0].max()
    # ymn=ws[:,1].min()
    # ymx=ws[:,1].max()
    xmn=0
    xmx=h
    ymn=0
    ymx=w
    print(xmn,xmx)
    print(ymn,ymx)

    wh=int(xmx-xmn)
    ww=int(ymx-ymn)
    igs_warp = np.zeros((wh,ww))
    print(igs_warp.shape)
    
    he=0
    sm=np.array([0,0,0,0,0,0],dtype=np.float64)
    for i in range(wh):
        for j in range(ww):
            p = (i+xmn,j+ymn)
            x,y,_=ai.dot(np.array([[*p,1]]).T)

            if 0<=x<=h-1 and 0<=y<=w-1:
                sx,sy=int(x),int(y)
                dx=1-(x-sx)
                dy=1-(y-sy)

                if not (sx+1>h-1 or sy+1>w-1):
                    igs_warp[i][j]=dx*dy*img2[sx][sy]
                    igs_warp[i][j]+=(1-dx)*dy*img2[sx+1][sy]
                    igs_warp[i][j]+=dx*(1-dy)*img2[sx][sy+1]
                    igs_warp[i][j]+=(1-dx)*(1-dy)*img2[sx+1][sy+1]
                else:
                    if sx+1>h-1 and sy+1>w-1:
                        igs_warp[i][j]=img2[sx][sy]
                    else:
                        if sx+1>h-1:
                            igs_warp[i][j]=dy*img2[sx][sy]
                            igs_warp[i][j]+=(1-dy)*img2[sx][sy+1]
                        else:
                            igs_warp[i][j]+=dx*img2[sx][sy]
                            igs_warp[i][j]+=(1-dx)*img2[sx+1][sy]

                # dA = np.array( [ [i,0,j,0,1,0], [0,i,0,j,0,1] ] )
                # gi = np.array([Gx[i][j], Gy[i][j]])
                gx, gy = Gx[i][j], Gy[i][j]
                gidA = np.array([gx*i,gy*i,gx*j,gy*j,gx,gy])
                he += np.sum(gidA*gidA)
                sm += gidA*(img1[i][j]-igs_warp[i][j])

                diff = img1[i][j]-igs_warp[i][j]

    dp=sm/he
    print(sm)
    print(he)
    print(dp)

    # print(img1.min())
    # print(img1.max())
    # print(igs_warp.min())
    # print(igs_warp.max())

    cv2.imwrite('messigray.png',igs_warp)
    err = img1-igs_warp
    cv2.imwrite('messigray_err.png',err)

    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_16S, 1, 0, ksize = 5)
    Gy = cv2.Sobel(I, cv2.CV_16S, 0, 1, ksize = 5)

    print(Gx.min())
    print(Gx.max())
    print(Gy.min())
    print(Gy.max())
    return

    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.

    p = np.zeros((6))
    dp = 1
    while np.linalg.norm(dp) > ep:
        dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        p += dp
        print(p)

    moving_image = np.abs(img2 - img1) # you should delete this

    th_hi = 0.2 * 256 # you can modify this
    th_lo = 0.15 * 256 # you can modify this

    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

data_dir = 'data'
video_path = 'motion.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 150/20, (320, 240))
tmp_path = os.path.join(data_dir, "{}.jpg".format(0))
T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
for i in range(1, 2):#150):
    img_path = os.path.join(data_dir, "{}.jpg".format(i))
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    clone = I.copy()
    moving_img = subtract_dominant_motion(T, I)
    clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
    clone[moving_img, 2] = 255
    out.write(clone)
    T = I
out.release()

