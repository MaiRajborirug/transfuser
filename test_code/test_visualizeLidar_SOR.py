import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import open3d as o3d

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the time duration for the animation (10 seconds) and number of frames (500)
fps = 30

# Function to load the point cloud from two npy files
def load_point_cloud(frame):
    # Calculate the indices of the two files to be combined
    idx1 = (frame * 2) + 1   # lidar_1.npy, lidar_3.npy, etc.
    idx2 = (frame * 2) + 2   # lidar_2.npy, lidar_4.npy, etc.
    print(idx1, idx2)

    pc1 = np.load(f'{choosing_route}/lidar_{idx1}.npy')
    pc2 = np.load(f'{choosing_route}/lidar_{idx2}.npy')
    
    # Combine the two point clouds into one array
    return np.vstack((pc1, pc2))

# Initialize the scatter plot
scatter = None

# Function to update the plot for each frame
def update(frame):
    global scatter
    ax.cla()  # Clear the previous plot

    # Load the current frame's combined point cloud data
    pcn = load_point_cloud(frame)
    
    #---------
    # Preprocess the point cloud
    # 1: limit the height
    pcn = pcn[(pcn[:, 2] > 0.3 -2.5) & (pcn[:, 2] < 0.5-2.5)]
    
    # 2: Statistical Outlier Removal (SOR):
        # convert to Open3D point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcn[:,:3])
        # Apply statistical outlier removal ~ k = n/100
    pcd_SOR, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.5)
    #     Convert back to numpy array (filtered X, Y, Z)
    # xyz_SOR = np.asarray(pcd_SOR.points)
    #     # Retain intensity from the original point cloud for the filtered points
    # intensity_SOR = pcn[ind, 3]
    # pointcloud_SOR = np.hstack((xyz_SOR, intensity_SOR.reshape(-1, 1)))
    
    # 3: voxel Grid Down Sampling
    voxel_size = 0.2
    pcd_downsampled = pcd_SOR.voxel_down_sample(voxel_size)
    
        # Convert back to numpy array (downsampled X, Y, Z)
    xyz_downsampled = np.asarray(pcd_downsampled.points)

        # Downsample intensity: choose the nearest point intensity or take the average
        # For simplicity, here we will take the average intensity for each voxel
    intensity_downsampled = []
    for point in xyz_downsampled:
        # Find the points in the original point cloud that fall into this voxel
        distances = np.linalg.norm(pcn[:, :3] - point, axis=1)
        mask = distances < voxel_size
        if np.any(mask):
            intensity_downsampled.append(np.mean(pcn[mask, 3]))
        else:
            intensity_downsampled.append(0)  # Handle cases with no intensity (optional)

    # Combine downsampled X, Y, Z with downsampled intensity
    intensity_downsampled = np.array(intensity_downsampled)
    pcn_downsampled = np.hstack((xyz_downsampled, intensity_downsampled.reshape(-1, 1)))
    
    #----------
    # print the number of points in each point cloud
    print(len(pcn), len(pcd_SOR.points), len(pcn_downsampled))
    
    #---- CBF-----
    # h(x,y,r) = x^2 + y^2+r^2
    v_i = 2 # m/s
    R_i = 2 # m
    dt = 0.05
    gamma = 0.03

    # random vehicle state
    v_e = 5
    # random vehicle control
    a_e = 2
    R_e = 4    
    #-------------
    

    # Extract X, Y, Z, and intensity from the point cloud
    # X+ left Y+ forward Z+ up (-2.2 to -1.5)
    X = pcn_downsampled[:, 0]
    Y = pcn_downsampled[:, 1]
    Z = pcn_downsampled[:, 2]
    intensity = pcn_downsampled[:, 3]
    
    #-------------
    # discrete CBF
    def _h(X, Y, R_i):
        return np.sqrt(X**2 + Y**2) - R_i
    def _h_t(X, Y, R_i, v_i, dt, gamma, v_e, a_e, R_e):
        arc = (v_e + 1/2 * a_e * dt)* dt
        angle = arc/R_e
        dy = - np.sin(angle) * R_e - v_i * dt * Y / np.sqrt(X**2 + Y**2)
        dx = (np.cos(angle)-1) * R_e - v_i * dt * X / np.sqrt(X**2 + Y**2)
        dtheta = arc/angle
        return np.sqrt((X+dx)**2 + (Y+dy)**2) - R_i
    
    # control action from discrete CBF
    certify = (_h_t(X, Y, R_i, v_i, dt, gamma, v_e, a_e, R_e) - (1*gamma)*_h(X, Y, R_i))
    
    #-------------

    # If scatter is None, initialize the scatter plot and color bar
    scatter = ax.scatter(X, Y, Z, c=certify, cmap='viridis', alpha=0.8, s=0.05)
    # if scatter is None:
    #     scatter = ax.scatter(X, Y, Z, c=certify, cmap='viridis', alpha=0.8, s=0.05)
    #     # color_bar = fig.colorbar(scatter, ax=ax, label='Color scale')
    # else:
    #     scatter._offsets3d = (X, Y, Z)
    #     # scatter.set_array(certify)

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the axis limits
    ax.set_xlim([-50, 50])  # X-axis range
    ax.set_ylim([-50, 50])   # Y-axis range
    ax.set_zlim([-25, 30])    # Z-axis range

    # Set the view angle (optional, adjust as needed)
    ax.view_init(elev=-90, azim=0)

# Load the point clouds from both files
route7 = '/media/haoming/970EVO/Pharuj/cdc_eval/240912_tfcbf_noise0.0_rep1_4' # num_frame=363
route13 = '/media/haoming/970EVO/Pharuj/cdc_eval/240912_tfcbf_noise0.0_rep1_5' # num_frame=312
route29 = '/media/haoming/970EVO/Pharuj/cdc_eval/240912_tfcbf_noise0.0_rep1_1' # num_frame=1206

choosing_route = route29

# Create the animation using FuncAnimation
num_frames = 1206//2
ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)

# Save the animation as a video file (optional)
# ani.save('lidar_animation.mp4', writer='ffmpeg', fps=fps)

# Show the animation
plt.show()