import math
import shutil

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
plt.rcParams['axes.grid'] = True


def cuboid_data(o, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0),size=(1,1,1),ax=None,**kwargs):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos, size )
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', vmin=-1., vmax=0.,**kwargs)

def stepwise_transform(a, points,ndim, rot,nsteps=10):
    '''
    code based on https://dododas.github.io/linear-algebra-with-python/posts/16-12-29-2d-transformations.html
    Generate a series of intermediate transform for the matrix multiplication
      np.dot(a, points) # matrix multiplication
    starting with the identity matrix, where
      a: 2-by-2 matrix
      points: 2-by-n array of coordinates in x-y space

    Returns a (nsteps + 1)-by-2-by-n array
    '''

    # create empty array of the right size
    transgrid = np.zeros((nsteps+1,) + np.shape(points))
    if rot:
        #keep the length of vector unchanged for rotation matrices
        for j in range(nsteps + 1):
            intermediate = np.eye(ndim) + j / nsteps * (a - np.eye(ndim))
            transgrid[j] = np.dot(intermediate, points)*calcNorm(points)/calcNorm(np.dot(intermediate, points))  # apply intermediate matrix transformation
    else:
        for j in range(nsteps+1):
            intermediate = np.eye(ndim) + j/nsteps*(a - np.eye(ndim))
            transgrid[j] = np.dot(intermediate, points) # apply intermediate matrix transformation
    print(transgrid)
    return transgrid

def prepareCoordSys3D(minimum, maximum, basis):
    #creates 3d coordniate system with basis vectors (optional)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #set limits of axis
    ax.set_zlim(minimum,maximum)
    ax.set_xlim(minimum,maximum)
    ax.set_ylim(minimum,maximum)
    if basis:
        basisVectors=np.eye(3)
        [plotVector(e, ax) for e in basisVectors]
    #show x,y,z axis by dashed lines
    x_values = [[minimum, maximum], [0, 0], [0, 0]]
    y_values = [[0, 0], [minimum, maximum], [0, 0]]
    z_values = [[0, 0], [0, 0], [minimum, maximum]]
    labels='xyz'
    for i in range(len(x_values)):
        plt.plot(x_values[i], y_values[i],z_values[i],'--', c='grey', linewidth=0.6, label=labels[i])
        ax.text(x_values[i][1], y_values[i][1],z_values[i][1], s=labels[i],c='grey')
    return fig

def plotVector(x, axes,name='',cube=True,**kwargs):
    #adds 3D vector represented by the coordinates of its tail to the figure, cube for the sake of
    #better visualisation
    z=0
    if len(x)==3:
        z=x[2]
    axes.quiver(0,0,0,x[0],x[1],z,arrow_length_ratio=0.1,**kwargs) # all vectors start in [0,0,0]
    if cube:
        plotCubeAt((0, 0, 0), size=[x[0],x[1],z], ax=axes, alpha=0.2)
    axes.text(x[0], x[1], z, s=name)

def plotVec2d(vec, ax, **kwargs):
    # adds 3D vector represented by the coordinates of its tail to the figure
    ax.arrow(0, 0, vec[0], vec[1],width= 0.1, length_includes_head=True,**kwargs)
    ax.text(vec[0] - 1, vec[1] + 0.2, s='(' + str(round(vec[0],1))+','+str(round(vec[1],1))+')')

def calcNorm(vec):
    #calculate euclidean norm of 3d vector 'manualy' instead of linalg.norm
    if len(vec) ==3:
        length=np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
    else:
        length= np.linalg.norm(vec)
    return length

def createAnimation(matr, vector, name, label,rot, min,max):
    #animates a sequence of matrix vector multiplications for the SVD slide
    folderName = 'Temp' + name
    os.makedirs(folderName)
    for j in range(len(matr)):
        interm = stepwise_transform(matr[j], vector[j], matr[j].shape[0], rot[j],20)
        i = 0
        for vec in interm:
            fig = prepareCoordSys3D(min, max, False)
            plt.title(label=label[j])
            ax = fig.axes[0]
            [plotVector(v,ax) for v in vector[j+1:len(vector)]]
            plotVector(vec, ax)
            z=0
            lengthVec=calcNorm(vec)
            if vec.shape[0]==3:
                z=vec[2]
            ax.text(vec[0]+1,vec[1]+1,z,s='||x||='+str(np.round(lengthVec,2)))
            outfile = os.path.join(folderName, "frame-" + str(j + 1).zfill(2) +'.'+ str(i + 1).zfill(2) + ".png")
            fig.savefig(outfile)
            i += 1
            plt.close(fig)

    from subprocess import call

    call("cd "+folderName+" && convert -delay 20 frame-*.png ../" + name + "-animation.gif", shell=True)

def createAnimationEV(matr, vector, name, label,rot, minimum,maximum,text):
    # animates a sequence of matrix vector multiplications for the Eigenvector slide
    folderName = 'Temp' + name
    os.makedirs(folderName)
    alwaysThere=np.dot(matr[0],vector[0])
    for j in range(len(matr)):
        interm = stepwise_transform(matr[j], vector[j], matr[j].shape[0], rot[j],20)
        i = 0
        for vec in interm:
            fig = prepareCoordSys3D(minimum, maximum,False)
            plt.title(label=label[j])
            ax = fig.axes[0]
            plotVector(vector[1],ax)
            ax.text(vector[1][0]+4, vector[1][1],2, s='v')
            if j>0:
                plotVector(alwaysThere, ax)
                ax.text(alwaysThere[0] -3, alwaysThere[1]-3, alwaysThere[2], s='A*v' )
            plotVector(vec, ax)
            z=0
            if vec.shape[0]==3:
                z=vec[2]
            ax.text(vec[0]+2,vec[1]+2,z+2,s=text[j])
            #plt.show()
            outfile = os.path.join(folderName, "frame-" + str(j + 1).zfill(3) +'.'+ str(i + 1).zfill(3) + ".png")
            fig.savefig(outfile)
            i += 1
            plt.close(fig)

    from subprocess import call

    call("cd "+folderName+" && convert -delay 20 frame-*.png ../" + name + "-animation.gif", shell=True)


def animationRotateSpace(ax,fig,name):
    #creates animation to show data cloud from all angles
    folderName = 'Temp' + name
    os.makedirs(folderName)
    j=0
    for i in range(0,90,1):
        ax.view_init(azim=i, elev=i)
        #plt.show()
        outfile = os.path.join(folderName, "frame-" + str(j + 1).zfill(3) + '.' + str(i + 1).zfill(3) + ".png")
        fig.savefig(outfile)
        plt.close(fig)

    from subprocess import call

    call("cd "+folderName+" && convert -delay 20 frame-*.png ../" + name + "-animation.gif", shell=True)
    #shutil.rmtree('Temp')

def slideSVD3d():
    A=np.asarray([[-1,-2],[1,-2],[1,-3]])
    x=[4,5]
    min=-20
    max=20
    newX=np.dot(A,x)

    #SVD part
    U,S,V=np.linalg.svd(A)

    #needed for animation
    SigmaSq=np.diag(S)

    #Extend Sigma for dot product
    Sigma=np.concatenate((SigmaSq,np.array([[0,0]])))

    #intermediate transformations of vector x
    rot1=np.dot(V,x)
    stretched=np.dot(Sigma,rot1)
    rot2=np.dot(U,stretched)

    fig2 = prepareCoordSys3D(min, max, True)
    ax2 = fig2.axes[0]
    plotVector(x,ax2,'x')
    plotVector(newX,ax2,'newX')
    plotVector(rot1,ax2,'rot1')
    plotVector(stretched,ax2,'stretched')
    plotVector(rot2,ax2,'rot2')
    #plt.show()

    A2=np.concatenate((A,np.asarray([[0],[0],[0]])),axis=1)
    x2=np.concatenate((x,np.asarray([0])))

    #direct Ax + SVDx
    createAnimation([A2,V,SigmaSq,U],[x2,x,rot1,stretched,rot2],"SVD",['A*x','Rotation 1', 'Streckung','Rotation 2'],[False,True,False,True], min, max)



#Eigenvalue decomposition
def slideEVD():
    B=np.asarray([[-1,0],[0.5,-1]]) #any matrix
    SPD=np.dot(B.T,B) #symmetric positive definite matrix

    anyVec=np.array([-1,2.5])

    #Apply matrix directly
    anyVecTransformed=np.dot(SPD,anyVec)

    #EVD
    eigvals, eigvecT= np.linalg.eig(SPD)
    eigvec= eigvecT.T

    #Apply matrixes from EVD one after another
    anyVecRot1=np.dot(eigvec,anyVec)
    anyVecStretched= np.dot(np.diag(eigvals),anyVecRot1)
    anyVecRot2=np.dot(eigvecT,anyVecStretched)

    #Make plot
    figEig, axEig = plt.subplots(2,2)
    print(eigvals, eigvecT)
    min=-4
    max=4
    plt.setp(axEig, xlim=(min,max), ylim=(min,max), aspect='equal')
    [ax.xaxis.set_ticks(np.arange(min,max, 1)) for axs in axEig for ax in axs]
    [ax.yaxis.set_ticks(np.arange(min,max, 1)) for axs in axEig for ax in axs]
    plt.subplots_adjust(wspace=0.25, hspace=0.8)
    plotVec2d(anyVec,axEig[0,0], color='red') #left up
    plotVec2d(anyVecRot1,axEig[1,0], color='red')#left down
    plotVec2d(anyVecStretched,axEig[1,1],color='red') #right down
    #check whether results are the same: plot 2 vectors
    plotVec2d(anyVecTransformed,axEig[0,1],color='red') #right up
    plotVec2d(anyVecRot2,axEig[0,1],color='red')  #right up

    plt.savefig('EVD.png')

def slideSVD2d():

    SPD= np.array([[1,2],[-2,3]]) #any matrix, disregard the naming

    anyVec=np.array([-1,2.5])

    #Apply matrix directly
    anyVecTransformed=np.dot(SPD,anyVec)

    #SVD
    singVecL,singvals, singVecRT= np.linalg.svd(SPD)
    singVecR= singVecRT.T

    #Apply matrixes from EVD one after another
    anyVecRot1=np.dot(singVecR,anyVec)
    anyVecStretched= np.dot(np.diag(singvals),anyVecRot1)
    anyVecRot2=np.dot(singVecL,anyVecStretched)

    #Make plot
    figSig, axSig = plt.subplots(2,2)
    min=-4
    max=12
    plt.setp(axSig, xlim=(min,max), ylim=(min,max), aspect='equal')
    [ax.xaxis.set_ticks(np.arange(min,max, 2)) for axs in axSig for ax in axs]
    [ax.yaxis.set_ticks(np.arange(min,max, 2)) for axs in axSig for ax in axs]
    plt.subplots_adjust(wspace=0.25, hspace=0.8)
    plotVec2d(anyVec,axSig[0,0], color='red') #left up
    plotVec2d(anyVecRot1,axSig[1,0], color='red')#left down
    plotVec2d(anyVecStretched,axSig[1,1],color='red') #right down
    #check whether results are the same: plot 2 vectors
    #plotVec2d(anyVecTransformed,axSig[0,1],color='red') #right up
    plotVec2d(anyVecRot2,axSig[0,1],color='red')  #right up

    plt.savefig('SVD.png')


def slideEVnonSquareMatr():
    A = np.asarray([[-1,-2],[1,-2],[1,-3]])
    x = [4, 5]
    minimum = -20
    maximum = 20
    Back=np.array([[-3, 0], [0, -3]])
    Forth=np.array([[3, 0], [0, 3]])

    #for animation: square matrices needed
    A2 = np.concatenate((A, np.asarray([[0], [0], [0]])), axis=1)
    x2 = np.concatenate((x, np.asarray([0])))
    createAnimationEV([A2,Forth,Back],[x2,x,x],"EVnonSq",['A*v', 'l*v, l>0','l*v, l<0'],[False,False,False],minimum,maximum,['','l*v','l*v'])

def findMinMaxCoords(matr):
    maxIndex = np.zeros(3)
    maxCoord = np.zeros(3)
    minIndex = np.zeros(3)
    minCoord = np.zeros(3)
    for i in range(3):
        maxPoint = max(matr, key=lambda x: (x[i]))
        maxIndex[i] = int(np.argwhere(matr == maxPoint)[i][0])
        maxCoord[i] = maxPoint[i]
        minPoint = min(matr, key=lambda x: (x[i]))
        minIndex[i] = int(np.argwhere(matr == minPoint)[i][0])
        minCoord[i] = minPoint[i]
    minimum = (min(minCoord)) - 3
    maximum = max(maxCoord) + 3
    return minimum,maximum,minIndex,maxIndex

#PCA part
def slidePCA():
    #create dataset
    nsamples=50
    dataset1= np.random.randn(nsamples,3)+1.5*np.ones((nsamples,3)) #random numbers in 3d
    stdev=[5,3,1]
    dataset2= np.dot(dataset1,np.diag(stdev)) #dataset with much bigger variance alongside one axis

    r=R.from_euler('xzy', [45,45,0], degrees=True)
    rotationMatr=r.as_matrix()
    dataset=np.dot(dataset2,rotationMatr)

    averDataset = [np.average(dataset[:, 0]), np.average(dataset[:, 1]), np.average(dataset[:, 2])]
    matr3centered = dataset - averDataset

    minimum,maximum,minIndex,maxIndex=findMinMaxCoords(matr3centered)

    fig3 = prepareCoordSys3D(minimum, maximum, False)
    ax3 = fig3.axes[0]
    ax3.scatter(matr3centered[:,0],matr3centered[:,1],matr3centered[:,2])

    #perform PCA
    pca = PCA(n_components=3)
    pca.fit(matr3centered)
    pcsTemp= pca.components_

    #avoid flipping of the data: new basis vectors do not show in the opposite direction to old ones
    standardBasis=np.eye(3)
    vectors=zip(pcsTemp,standardBasis)
    pcs=np.asarray([pc*(-1) if np.dot(e,pc)<0 else pc for pc,e in vectors])

    #colors for data points
    col=np.ones(nsamples)
    print(maxIndex)
    print(minIndex)
    for i in maxIndex:
        col[int(i)]=0
    for i in minIndex:
        col[int(i)]=0
    colors=col.reshape(nsamples,1)
    matr3centered= np.concatenate((matr3centered,colors), axis=1)

    fig4 = prepareCoordSys3D(minimum, maximum, False)
    ax4 = fig4.axes[0]


    # find angle between (0,0,1) and projection of pc3 onto yz plane
    pc1proj = pcs[2].copy()
    pc1proj[0] = 0
    cosOfangle = np.dot(pc1proj, np.asarray([ 0, 0,1])) / calcNorm(pc1proj)
    angle2 = np.degrees(np.arccos(cosOfangle))+90

    # find angle between (0,0,1) and projection of pc3 onto yz plane
    pc1proj = pcs[0].copy()
    pc1proj[2] = 0
    cosOfangle = np.dot(pc1proj, np.asarray([1, 0, 0])) / calcNorm(pc1proj)
    angle3 = np.degrees(np.arccos(cosOfangle))

    ax4.scatter(matr3centered[:,0],matr3centered[:,1],matr3centered[:,2],c=matr3centered[:,3], cmap='viridis', alpha=0.8)
    animationRotateSpace(ax4, fig4, 'rotateDataset')

    #add PCs to Animation
    figPCsAnim = prepareCoordSys3D(minimum, maximum, False)
    axPCsAnim = figPCsAnim.axes[0]
    axPCsAnim.scatter(matr3centered[:, 0], matr3centered[:, 1], matr3centered[:, 2], c=matr3centered[:, 3], cmap='viridis',
                alpha=0.8)
    names=['pc1','pc2','pc3']
    colorsPC=['green','orange','red']
    inp=zip(pcs,names,stdev,colorsPC)
    [plotVector(vec*2*stdev,axPCsAnim,name,False,color=col) for vec,name,stdev,col in inp]
    axPCsAnim.view_init(azim=-90, elev=angle2)
    plt.savefig('pca.png')

    basisChanged = np.asarray([np.dot(pcs, point) for point in matr3centered[:,0:3]])
    basisChanged= np.concatenate((basisChanged,colors), axis=1)
    figBasisChanged = prepareCoordSys3D(minimum, maximum, False)
    ax5 = figBasisChanged.axes[0]
    ax5.view_init(azim=-angle3, elev=90)
    ax5.scatter(basisChanged[:, 0], basisChanged[:, 1], basisChanged[:, 2],c=basisChanged[:,3], cmap='viridis',alpha=0.8)
    plt.savefig('newBasisRot.png')
    ax5.view_init(azim=0, elev=0)
    plt.savefig('newBasisYZ.png')
    ax5.view_init(azim=-90, elev=89)
    plt.savefig('newBasisXY.png')

    reducedPCs=pcs.copy()
    reducedPCs[2]=np.array([0,0,0])
    basisReduced = np.asarray([np.dot(reducedPCs, point) for point in matr3centered[:, 0:3]])
    print(basisReduced)
    basisReduced = np.concatenate((basisReduced, colors), axis=1)
    figbasisReduced = prepareCoordSys3D(minimum, maximum, False)
    axBasisReduced = figbasisReduced.axes[0]
    axBasisReduced.scatter(basisReduced[:, 0], basisReduced[:, 1], basisReduced[:, 2], c=basisReduced[:, 3], cmap='viridis',
                alpha=0.8)

    #plt.show()

    animationRotateSpace(axPCsAnim,figPCsAnim,'rotatePCA')
    animationRotateSpace(axBasisReduced,figbasisReduced, 'rotateAfterPCA')

#callthe functions for slides
slideEVD()
slideEVnonSquareMatr()
slideSVD2d()
slideSVD3d()
slidePCA()

