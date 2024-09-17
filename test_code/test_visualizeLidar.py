import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
    point_cloud = load_point_cloud(frame)
    point_cloud = point_cloud[(point_cloud[:, 2] > 0.2 -2.5) & (point_cloud[:, 2] < 0.5-2.5)]

    print(len(point_cloud))

    # Extract X, Y, Z, and intensity from the point cloud
    # X+ left Y+ forward Z+ up (-2.2 to -1.5)
    X = point_cloud[:, 0]
    Y = point_cloud[:, 1]
    Z = point_cloud[:, 2]
    intensity = point_cloud[:, 3]

    # Create the scatter plot (use 'viridis' color map for intensity)
    scatter = ax.scatter(X, Y, Z, c=intensity, cmap='viridis', alpha=0.8, s = 0.05)

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the axis limits
    ax.set_xlim([-50, 50])  # X-axis range
    ax.set_ylim([-50, 50])   # Y-axis range
    ax.set_zlim([-25, 30])    # Z-axis range

    # Set the view angle (optional, adjust as needed)
    ax.view_init(elev=90, azim=0)

# Load the point clouds from both files
route7 = '/media/haoming/970EVO/pharuj/cdc_eval/240912_tfcbf_noise0.0_rep1_4' # num_frame=363
route13 = '/media/haoming/970EVO/pharuj/cdc_eval/240912_tfcbf_noise0.0_rep1_5' # num_frame=312
route29 = '/media/haoming/970EVO/pharuj/cdc_eval/240912_tfcbf_noise0.0_rep1_1' # num_frame=1206

choosing_route = route13

# Create the animation using FuncAnimation
num_frames = 310//2
ani = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps)

# Save the animation as a video file (optional)
# ani.save('lidar_animation.mp4', writer='ffmpeg', fps=fps)

# Show the animation
plt.show()