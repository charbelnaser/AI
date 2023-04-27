# Task 1
This Task Is used to Write to read, write and visualize the point cloud

To access the data used in this project, click [here] (https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip)


```python
import open3d as o3d
import numpy as np
import glob
import copy
import os


# Get a list of all binary files in the directory
file_list = glob.glob('velodyne_points/data/*.bin')

# Load each file and concatenate the data into a single array
data = np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, 4) for f in file_list])
# Reshape the data to a 2D array of shape (N, 4), where N is the number of points
data = data.reshape(-1, 4)

# Write the segmented point cloud data to a new binary file
with open("segmented_point_cloud.bin", mode="wb") as f:
    data[:, :3].astype(np.float32).tofile(f)

# Read the segmented point cloud data from the new binary file
with open("segmented_point_cloud.bin", mode="rb") as f:
    data_new = np.fromfile(f, dtype=np.float32).reshape(-1, 3)
pcd_new = o3d.geometry.PointCloud()
pcd_new.points = o3d.utility.Vector3dVector(data_new)
pcd_new.colors = o3d.utility.Vector3dVector(np.zeros((data_new.shape[0], 3)))

# Visualize the segmented point cloud data
o3d.visualization.draw_geometries([pcd_new])

```


There are various methods for removing dynamic objects from point clouds. Here are three commonly used techniques:

Octree-based filtering: This method divides the point cloud into octree voxels and analyzes the motion characteristics of each voxel. If a voxel's motion exceeds a certain threshold, it is marked as a dynamic object and filtered out from the point cloud. This technique can handle different types of dynamic objects and is computationally efficient.

Moving least squares (MLS) filter: The MLS filter is a widely used method for point cloud filtering. It fits a local polynomial to a set of points and replaces each point with its projection onto the polynomial. This method can be used to filter out the dynamic objects by analyzing the local motion characteristics of each point.

Time-of-Flight (ToF) camera-based filtering: ToF cameras capture the time it takes for a light pulse to travel from the camera to the object and back. By analyzing the time delay, the distance to the object can be calculated. ToF cameras can be used to distinguish dynamic objects from static objects based on the change in distance over time.

It's worth noting that the performance of each method heavily depends on the specific application and the characteristics of the point cloud. Therefore, it's important to evaluate and compare different techniques to choose the most suitable method for a specific task


```python
import unittest
import numpy as np
import open3d as o3d

class TestPointCloudSegmentation(unittest.TestCase):
    
    def test_segmented_point_cloud(self):
        # Get a list of all binary files in the directory
        file_list = glob.glob('velodyne_points/data/*.bin')

        # Load each file and concatenate the data into a single array
        data = np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, 4) for f in file_list])
        # Reshape the data to a 2D array of shape (N, 4), where N is the number of points
        data = data.reshape(-1, 4)

        # Write the segmented point cloud data to a new binary file
        with open("segmented_point_cloud.bin", mode="wb") as f:
            data[:, :3].astype(np.float32).tofile(f)

        # Read the segmented point cloud data from the new binary file
        with open("segmented_point_cloud.bin", mode="rb") as f:
            data_new = np.fromfile(f, dtype=np.float32).reshape(-1, 3)
        
        # Check if the number of points in the segmented point cloud is correct
        self.assertEqual(data_new.shape[0], data.shape[0])
        
        # Check if the loaded point cloud has the correct number of points and colors
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(data_new)
        pcd_new.colors = o3d.utility.Vector3dVector(np.zeros((data_new.shape[0], 3)))
        self.assertEqual(len(pcd_new.points), data_new.shape[0])
        self.assertEqual(len(pcd_new.colors), data_new.shape[0])

if __name__ == '__main__':
    unittest.main()
```

## Requirements

This project requires Python 3.6 or later, along with the following packages:

- numpy==1.21.6
- open3d==0.17.0

To install the required packages, run the following command