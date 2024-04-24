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

sys.path.append("C:\\Users\\lenovo\\Documents\\physics\\voronoi\\decfirst")
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

parser.add_argument("-k", "--skip", type=int, required=False,
                    help="Trajectory frame skip rate. E.g. SKIP=10 will only \
                    render every 10th frame.",
                    default=1)

parser.add_argument("--min-cells", type=int, required=False,
                    help='Start rendering when system has at least this many cells',
                    default=1)

parser.add_argument("--inds", type=int, required=False, nargs='+',
                    help="Only render cells with these indices",
                    default=[])

parser.add_argument("-nf", "--num-frames", type=int, required=False,
                    help="Only render this many frames.",
                    default=sys.maxsize)

args = parser.parse_args(argv)

imageindex = 0
firstfaces = []


with open('C180_pentahexa.csv', newline='') as g:
    readerfaces = csv.reader(g, delimiter=',')
    for row in readerfaces:
        firstfaces.append([int(v) for v in row])


filename = os.path.realpath(args.trajPath)
basename = os.path.splitext(filename)[0] + "/images/CellDiv_"

nSkip = args.skip

if nSkip > 1:
    print("Rendering every %dth" % nSkip, "frame...")

cellInds = []
minInd = args.min_cells - 1
if len(args.inds) > 0:
    minInd = max(minInd, min(args.inds))

stopAt = args.num_frames

# Set material color

#filename="inp.xyz"
#Files=["inp2.xyz" , "inp5.xyz" , "inp10.xyz"]

Files=["inp{}.xyz".format(i) for i in range(1,2)]
#cell_com=[]


def Plot_NN_disrtibution(vor):
    max=14
    N=np.zeros(max)
    a=np.arange(1,max+1)
    #vor.regions : list of indices of vertices for each voronoi cell
    # we can find how many neighbors each cell has by looking at the length of each list
    for j in vor.regions:
        N[len(j)-1]+=1
    #Dros=[0,0,0,0.03,0.28,0.46,0.2,0.03,0,0,0,0]
    #xenopus=[0,0,0.01,0.04,0.285,0.43,0.18,0.05,0.02,0,0,0]
    #plt.bar(a- 0.2,Dros,0.2, edgecolor='black',label="Drosophila(2171)")
    #plt.bar(a ,xenopus,0.2, edgecolor='black',label="Xenopus(1051)")
    #plt.bar(a + 0.2,N/sum(N),0.2, edgecolor='black',label="simulation(1783)")
    plt.bar(a,N/sum(N),0.2, edgecolor='black',label="simulation")

    
    plt.title("Number of Nearest Neighbors")
    plt.xlabel("$N_n$")
    plt.ylabel("fraction")
    #plt.legend()
    #cmap = plt.get_cmap('hot')
    #plt.set_cmap(cmap)
    plt.savefig("NN_plot{}.png".format(i))
    #plt.show()

def Plot_cell_distribution(cell_com):
    xs=[ i[0] for i in cell_com ]
    ys=[ i[1] for i in cell_com ]
    coords= np.array(cell_com)[: , 0:2]
    #print(coords)
    vor = Voronoi(coords)
    fig = voronoi_plot_2d(vor, show_vertices=False, point_size=2)
    plt.savefig("voronoi.png")
    plt.show()

def Calculte_area_distribution(vor):
    s=[]
    for p in range(len(vor.points)):
        index_of_vor_region_for_point = vor.point_region[p]
        if -1 not in vor.regions[index_of_vor_region_for_point]:
            index_of_vertice = vor.regions[index_of_vor_region_for_point]
            S_seg=0
            for v in range(len(index_of_vertice)): 
                p1=vor.vertices[index_of_vertice[v]]
                p2=vor.vertices[index_of_vertice[v-1]]
                p3=vor.points[p]
                S_seg += 1/2*np.abs(( p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) ))
                # plt.show()
                # plt.scatter(vor.vertices[ver][0],vor.vertices[ver][1])
                # print(vor.vertices[ver])
            if S_seg<=(np.pi):
                s.append(S_seg)
            S_seg=0
    return s

def Plot_area_distribution(vor,param):
    s=Calculte_area_distribution(vor)
    plt.hist(s,bins=200,label="Simulation")
    #plt.legend()
    plt.xlabel("Area")
    plt.ylabel("Count")
    plt.title("Area distribution for stiffnes {}".format(param))
    plt.savefig("Area_distribution{}.png".format(i))
    #plt.title("Area distribution for a simulated epithelium")
    #svg
    #print(s)        
        #plt.scatter(vor.points[p][0], vor.points[p][1] , marker=".")
         
def g3_distribution(com_coordinates,cut_off_radius): 
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
    #plot density plot
    r = np.array(r)
    cos_theta = np.array(cos_theta)
    bins_r=401
    bins_theta=201
    k = gaussian_kde([r,cos_theta]) # this creates a kernel density estimator
    xi, yi = np.mgrid[r.min():r.max():bins_r*1j, cos_theta.min():cos_theta.max():bins_theta*1j] # this creates a grid of points
    zi = k(np.vstack([xi.flatten(), yi.flatten()])) # this evaluates the KDE on the grid
    # Plotting
    cmap = cm.get_cmap('jet')
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
    plt.colorbar()
    #plt.title('$\\gamma_{ext}$='+'{}'.format(1))
    plt.xlabel('r/$\\sigma$')
    plt.ylabel('$cos(\\theta)$')
    plt.savefig("g3_dist{}.png".format(cut_off_radius))
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
    xi, yi = np.mgrid[0:cut_off_radius:bins_r*1j, -1:1:bins_theta*1j] # this creates a grid of points
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
    #clear previous plot
    plt.clf()
    
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
    plt.colorbar()
    #plt.title('Density plot of g3 distribution')
    plt.xlabel('r/$\\sigma$')
    plt.ylabel('$cos(\\theta)$')
    #save photo in file photos
    
    plt.savefig("g3_dist{}.png".format(h))

def Color_area_distribution(coords,scan_theta,scan_radius): # (coords,[theta_min,theta_max],[r_min,r_max])
    vor=Voronoi(coords)
    fig, ax = plt.subplots()
    s=Calculte_area_distribution(vor)
    Maximum=scan_theta[1] #largest theta 
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cnorm = mcol.Normalize(vmin=0,vmax=np.pi)
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick.set_array([])

    for p in range(len(vor.points)):
        index_of_vor_region_for_point = vor.point_region[p]
        if -1 not in vor.regions[index_of_vor_region_for_point]:
            index_of_vertice = vor.regions[index_of_vor_region_for_point]
            #calculte the area 
            S_seg=0
            A_seg=0
            for v in range(len(index_of_vertice)): 
                p1=vor.vertices[index_of_vertice[v]]
                p2=vor.vertices[index_of_vertice[v-1]]
                p3=vor.points[p]
                S_seg += 1/2*np.abs(( p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]) ))
                A_seg += np.abs((p1[0]-p2[0]) + (p1[1]-p2[1])*1j)

            #g3
            B=vor.points[p]
            com_coordinates = coords
            other_coordinates = [i for i in com_coordinates if not np.all(i == B)]
            dist = [np.linalg.norm(np.array(B) - np.array(i)) for i in other_coordinates]
            #find min index
            min_index = dist.index(min(dist))
            A = other_coordinates[min_index]
            other_other_coordinates= [i for i in other_coordinates if not np.all(i == A)]
            #find the atom C in the cutoff range
            C = [i for i in other_other_coordinates if np.linalg.norm(np.array(B) - np.array(i)) < scan_radius[1]]
            if C:
                #calculate the angle between the vectors AB and AC
                AB = np.array(B) - np.array(A)
                AC = np.array(B) - np.array(C)
                r=[]
                cos_theta=[]
                for i in range(len(AC)):
                    r.append(np.linalg.norm(AC[i]))
                    cos_theta.append(np.dot(AB,AC[i])/(np.linalg.norm(AB)*np.linalg.norm(AC[i])))
                
                #if a cos(theta) is in the range of the scan
                if any(scan_theta[0]<i<scan_theta[1] for i in cos_theta) and any(scan_radius[0]<i<scan_radius[1] for i in r) and S_seg<=(np.pi):
                    p = []
                    for ind in index_of_vertice:
                        p.append(vor.vertices[ind])
                    patches = []
                    patches.append(Polygon(p, closed=True))
                    p = PatchCollection(patches,edgecolor="r",alpha=0.8)
                    p.set_color(cpick.to_rgba(S_seg))
                    ax.add_collection(p)
            elif S_seg<=(np.pi): 
                p = []
                for ind in index_of_vertice:
                    p.append(vor.vertices[ind])
                patches = []
                patches.append(Polygon(p, closed=True))
                p = PatchCollection(patches,edgecolor="b",alpha=0.4)
                p.set_color("black")
                ax.add_collection(p)
    ax.set_xlim([-5,45])
    ax.set_ylim([-5,45])
    plt.colorbar(cpick,label="Area",ax=plt.gca())
    plt.savefig("Area_plot{}.png".format(i))
    plt.close()


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
                    #save_g3_disrtibution(coords,2,"g3_dist","visc",file_number)
                    print(file_number)
                    #g3_distribution(coords,2)
                    #Plot_cell_distribution(coords)
                    Color_area_distribution(coords,[-1,1],[0,1.0])

        except celldiv.IncompleteTrajectoryError:
            print ("Stopping...")   
    file_number+=1 #
plt.show()                
