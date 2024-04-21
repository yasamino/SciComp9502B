#instructions:
#change path line 24
#add c180_pentahexa.csv to folder
#add celldiv.py to folder
#create the file array Files with the names of the files you want to analyze

import csv
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcol
#import seaborn
from scipy.stats import gaussian_kde
import pandas as pd


#seaborn.set()
# on coltrane
sys.path.append("/home/yasaminmodabber/CellSim3D/bin")
import celldiv


argv = sys.argv

if "--" not in argv:
    print("ERROR: No arguments provided to script")
    sys.exit(80)
else:
    a = argv.index("--")
    argv = argv[a + 1:]

helpString = """
Run as:
blender --background --python %s --
[options]
""" % __file__

parser = argparse.ArgumentParser(description=helpString)

parser.add_argument("trajPath", type=str,
                    help="Trajectory path. Absolute or relative.")

parser.add_argument("-s", "--smooth", action='store_true',
                    help="Do smoothing (really expensive and doesn't look as good)")

parser.add_argument("-k", "--skip", type=int, required=False,
                    help="Trajectory frame skip rate. E.g. SKIP=10 will only \
                    render every 10th frame.",
                    default=1)

parser.add_argument("-nc", "--noclear", type=bool, required=False,
                    help="specifying this will not clear the destination directory\
                    and restart rendering.",
                    default=False)

parser.add_argument("--min-cells", type=int, required=False,
                    help='Start rendering when system has at least this many cells',
                    default=1)

parser.add_argument("--inds", type=int, required=False, nargs='+',
                    help="Only render cells with these indices",
                    default=[])

parser.add_argument("-nf", "--num-frames", type=int, required=False,
                    help="Only render this many frames.",
                    default=sys.maxsize)

parser.add_argument("-r", "--res", type=int, default=1, required=False,
                    help='Renders images with resolution RES*1080p. RES>=1. \
                    Use 2 for 4k. A high number will devour your RAM.')

parser.add_argument("-cc", "--cell-color", type=int, nargs=3, required=False,
                    default=[72, 38, 153],
                    help="RGB values of cell color. From 0 to 255")

parser.add_argument("-bc", "--background-color", type=int, nargs=3,
                    required=False, default=[255,255,255],
                    help="RGB values of cell color. From 0 to 255")

parser.add_argument("-si", "--specular-intensity", type=float, required=False,
                    default = 0.0,
                    help="Set specular-intensity (shininess). From 0.0 to 1.0")

args = parser.parse_args(argv)

imageindex = 0
firstfaces = []


doSmooth = args.smooth
if doSmooth:
    print("Doing smoothing. Consider avoiding this feature...")


if (args.res < 1):
    print("ERROR: invalid resolution factor")
    sys.exit()

with open('C180_pentahexa.csv', newline='') as g:
    readerfaces = csv.reader(g, delimiter=',')
    for row in readerfaces:
        firstfaces.append([int(v) for v in row])


filename = os.path.realpath(args.trajPath)
basename = os.path.splitext(filename)[0] + "/images/CellDiv_"

nSkip = args.skip

if nSkip > 1:
    print("Rendering every %dth" % nSkip, "frame...")


noClear = args.noclear

sPath = os.path.splitext(filename)[0] + "/images/"

if not noClear and os.path.exists(sPath):
    for f in os.listdir(sPath):
        os.remove(sPath+f)

cellInds = []
minInd = args.min_cells - 1
if len(args.inds) > 0:
    minInd = max(minInd, min(args.inds))

stopAt = args.num_frames


# Data extraction
first_data=1
last_data=100
var= "Stiff"
Files=[]
for i in range(first_data,last_data+1):
    #check to see if file named var+"{}".format(i)+".csv" exists
    if os.path.exists(var+"{}".format(i)+".csv"):
        Files.append(var+"{}".format(i)+".csv")
print(Files)


def Plot_cell_distribution(cell_com,h):
    xs=[ i[0] for i in cell_com ]
    ys=[ i[1] for i in cell_com ]
    coords= np.array(cell_com)[: , 0:2]
    #print(coords)
    vor = Voronoi(coords)
    fig = voronoi_plot_2d(vor, show_vertices=False, point_size=2)
    plt.savefig("voronoi{}.png".format(h))
    plt.show()


def save_g3_disrtibution(com_coordinates,cut_off_radius,filename,header,h):
    r=[]
    cos_theta=[]
    for B in com_coordinates:
        #an array with all coordinates except the current one
        other_coordinates = [i for i in com_coordinates if not np.all(i == B)]
        dist = [np.linalg.norm(np.array(B) - np.array(i)) for i in other_coordinates]
        #find min index
        min_index = dist.index(min(dist))
        A = other_coordinates[min_index]
        
        other_other_coordinates= [i for i in other_coordinates if not np.all(i == A)]
        #find the atom C in the cutoff range
        C = [i for i in other_other_coordinates if np.linalg.norm(np.array(B) - np.array(i)) < cut_off_radius]
        if C:
            #calculate the angle between the vectors AB and AC
            AB = np.array(B) - np.array(A)
            AC = np.array(B) - np.array(C)
            for i in range(len(AC)):
                r.append(np.linalg.norm(AC[i]))
                cos_theta.append(np.dot(AB,AC[i])/(np.linalg.norm(AB)*np.linalg.norm(AC[i])))
    r = np.array(r)
    cos_theta = np.array(cos_theta)
    bins_r=401
    bins_theta=201
    k = gaussian_kde([r,cos_theta]) # this creates a kernel density estimator
    xi, yi = np.mgrid[0:cut_off_r:bins_r*1j, -1:1:bins_theta*1j] # this creates a grid of points
    zi = k(np.vstack([xi.flatten(), yi.flatten()])) # this evaluates the KDE on the grid
    #Now to add the data to a csv file
    #we create the file if it does not exist
    if os.path.exists(filename+".csv"):
        df = pd.read_csv(filename+".csv")
    else:
        df = pd.DataFrame()
    df[header+'{}'.format(h)] = zi
    df.to_csv(filename+".csv", index=False)
    # plot and save the results
    cmap = cm.get_cmap('jet')
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
    plt.colorbar()
    #plt.title('Density plot of g3 distribution')
    plt.xlabel('r/$\\sigma$')
    plt.ylabel('$cos(\\theta)$')
    plt.savefig("g3_dist{}.png".format(h))


cut_off_r=2

file_number=1
for filename in Files:
    with celldiv.TrajHandle(filename) as th:
        frameCount = 1
        try:
            for i in range(int(th.maxFrames/nSkip)):
                cell_com=[]
                frameCount += 1
                if frameCount > args.num_frames:
                    break
                f=th.ReadFrame(inc=nSkip )
                if len(args.inds) > 0:
                    f = [f[a] for a in args.inds]
                f = np.vstack(f)
                for mi in range(int(len(f)/192)):
                    cell_com.append(np.mean(f[mi*192:(mi+1)*192],axis=0))
                coords= np.array(cell_com)[: , 0:2] #2D coordinates
                if frameCount==int(th.maxFrames/nSkip): 
                    save_g3_disrtibution(coords,cut_off_r,"g3_dist",var,file_number)
                    Plot_cell_distribution(coords)
        except celldiv.IncompleteTrajectoryError:
            print ("Stopping...")   
    file_number+=1 #
plt.show()                