import os
import numpy as np
import cv2
from skimage.filters import apply_hysteresis_threshold

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.

    dxx,dyx,dxy,dyy,dx,dy = p
    A = np.array( [ [1+dxx,dxy,dx], [dyx,1+dyy,dy], [0,0,1] ] )
    ai = np.linalg.inv(A)

    h,w = img2.shape

    xmn=0
    xmx=h-1
    ymn=0
    ymx=w-1
    # print(xmn,xmx)
    # print(ymn,ymx)

    wh=int(xmx-xmn)+1
    ww=int(ymx-ymn)+1

    igs_warp = np.zeros((wh,ww))
    Gx_warp = np.zeros((wh,ww))
    Gy_warp = np.zeros((wh,ww))

    H=np.zeros((6,6))
    # ssq=0
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

                    Gx_warp[i][j]=dx*dy*Gx[sx][sy]
                    Gx_warp[i][j]+=(1-dx)*dy*Gx[sx+1][sy]
                    Gx_warp[i][j]+=dx*(1-dy)*Gx[sx][sy+1]
                    Gx_warp[i][j]+=(1-dx)*(1-dy)*Gx[sx+1][sy+1]

                    Gy_warp[i][j]=dx*dy*Gy[sx][sy]
                    Gy_warp[i][j]+=(1-dx)*dy*Gy[sx+1][sy]
                    Gy_warp[i][j]+=dx*(1-dy)*Gy[sx][sy+1]
                    Gy_warp[i][j]+=(1-dx)*(1-dy)*Gy[sx+1][sy+1]

                else:
                    if sx+1>h-1 and sy+1>w-1:
                        igs_warp[i][j]=img2[sx][sy]
                        Gx_warp[i][j]=Gx[sx][sy]
                        Gy_warp[i][j]=Gy[sx][sy]

                    else:
                        if sx+1>h-1:
                            igs_warp[i][j]=dy*img2[sx][sy]
                            igs_warp[i][j]+=(1-dy)*img2[sx][sy+1]

                            Gx_warp[i][j]=dy*Gx[sx][sy]
                            Gx_warp[i][j]+=(1-dy)*Gx[sx][sy+1]

                            Gy_warp[i][j]=dy*Gy[sx][sy]
                            Gy_warp[i][j]+=(1-dy)*Gy[sx][sy+1]
                        else:
                            igs_warp[i][j]+=dx*img2[sx][sy]
                            igs_warp[i][j]+=(1-dx)*img2[sx+1][sy]

                            Gx_warp[i][j]+=dx*Gx[sx][sy]
                            Gx_warp[i][j]+=(1-dx)*Gx[sx+1][sy]

                            Gy_warp[i][j]+=dx*Gy[sx][sy]
                            Gy_warp[i][j]+=(1-dx)*Gy[sx+1][sy]

                # dA = np.array( [ [i,0,j,0,1,0], [0,i,0,j,0,1] ] )
                # gi = np.array([Gx[i][j], Gy[i][j]])
                gx, gy = Gx_warp[i][j], Gy_warp[i][j]
                gidA = np.array([[gx*i,gy*i,gx*j,gy*j,gx,gy]])
                H += gidA.T.dot(gidA)
                sm += gidA[0]*(img1[i][j]-igs_warp[i][j])
                # ssq += (img1[i][j]-igs_warp[i][j])**2

    dp=np.linalg.inv(H).dot(sm)
    # # print(sm)
    # # print(H)
    # # print(dp)
    # print('ssq',ssq)

    # cv2.imwrite('messigray.png',igs_warp*255)

    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 3)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 3)

    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.

    # print(Gx.min(),Gx.max())
    # print(Gy.min(),Gy.max())
    Gx=Gx/(np.abs(Gx).max()*2)+0.5
    Gy=Gy/(np.abs(Gy).max()*2)+0.5
    # print(Gx.min(),Gx.max())
    # print(Gy.min(),Gy.max())

    p = np.zeros((6))
    img1_n=img1/255
    img2_n=img2/255

    # e = 10**-1
    # while True:

    dp = lucas_kanade_affine(img1_n, img2_n, p, Gx, Gy)
    p += dp
    # print('newp',p)

        # rate = np.linalg.norm(dp)/np.linalg.norm(p)
        # print(rate)
        # if rate < e:
        #     break

    dxx,dyx,dxy,dyy,dx,dy = p
    A = np.array( [ [1+dxx,dxy,dx], [dyx,1+dyy,dy], [0,0,1] ] )
    ai = np.linalg.inv(A)

    h,w = img2.shape

    xmn=0
    xmx=h-1
    ymn=0
    ymx=w-1
    # print(xmn,xmx)
    # print(ymn,ymx)

    wh=int(xmx-xmn)+1
    ww=int(ymx-ymn)+1
    igs_warp = np.zeros((wh,ww))
    # print(igs_warp.shape)
    
    # ssq=0
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
    
                # ssq += (img1[i][j]-igs_warp[i][j])**2

    # print('ssq',ssq)

    moving_image = np.abs(img1-igs_warp)

    th_hi = 0.2 * 256 # you can modify this
    th_lo = 0.08 * 256 # you can modify this

    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

data_dir = 'data'
video_path = 'motion.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 150/20, (320, 240))
tmp_path = os.path.join(data_dir, "{}.jpg".format(0))
T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
for i in range(1, 150):
    print("#",i)
    img_path = os.path.join(data_dir, "{}.jpg".format(i))
    I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    clone = I.copy()
    moving_img = subtract_dominant_motion(T, I)
    clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
    clone[moving_img, 2] = 255
    cv2.imwrite(f"g{i}.png",clone)
    out.write(clone)
    T = I

out.release()

