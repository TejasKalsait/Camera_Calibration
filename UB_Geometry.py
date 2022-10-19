import numpy as np
from typing import List, Tuple
import cv2
import matplotlib.pyplot as plt

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    
    # Converting to radians for numpy library
    
    #alpha = (alpha * np.pi) / 180
    #beta = (beta * np.pi) / 180
    #gamma = (gamma * np.pi) / 180
    alpha, beta, gamma = converttoradians(alpha, beta, gamma)
    
    rot_xyz2XYZ = Rz(gamma) * Rx(beta) * Rz(alpha)
    #rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    
    # Converting to radians for numpy library
    
    #alpha = (alpha * np.pi) / 180
    #beta = (beta * np.pi) / 180
    #gamma = (gamma * np.pi) / 180
    
    alpha, beta, gamma = converttoradians(alpha, beta, gamma)
    
    
    temp = Rz(gamma) * Rx(beta) * Rz(alpha)
    rot_XYZ2xyz = np.linalg.inv(temp)
    
    
    #rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation
    
    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1

def Rx(angle):
  return np.matrix([[ 1, 0 , 0],
                   [ 0, np.cos(angle), -np.sin(angle)],
                   [ 0, np.sin(angle), np.cos(angle)]])
  
def Ry(angle):
  return np.matrix([[ np.cos(angle), 0, np.sin(angle)],
                   [ 0, 1, 0 ],
                   [-np.sin(angle), 0, np.cos(angle)]])
  
def Rz(angle):
  return np.matrix([[ np.cos(angle), -np.sin(angle), 0 ],
                   [ np.sin(angle), np.cos(angle) , 0 ],
                   [ 0, 0, 1]])

def converttoradians(alpha, beta, gamma):
    
    alpha = (alpha * np.pi) / 180
    beta = (beta * np.pi) / 180
    gamma = (gamma * np.pi) / 180
    
    return alpha, beta, gamma



#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(imagee: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    #img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation
    
    #dimensions = imagee.shape
    #print(dimensions)
    
    # Cropping
    plt.imshow(imagee)
    crop_image_1 = imagee[500:, 0:1000]
    crop_image_2 = imagee[500:, 1000:]
    
    # Grayscale
    
    gray = cv2.cvtColor(imagee, cv2.COLOR_BGR2GRAY)
    gray_1 = cv2.cvtColor(crop_image_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(crop_image_2, cv2.COLOR_BGR2GRAY)
    
    # Gettingcorners
    
    retval_1, corners_1 = cv2.findChessboardCorners(image = gray_1, patternSize = (4,4), flags = None)
    retval_2, corners_2 = cv2.findChessboardCorners(image = gray_2, patternSize = (4,4), flags = None)
    
    #Criteria for subpix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Offset for left image

    if retval_1 == True and retval_2 == True:
        for i in range(len(corners_1)):
            corners_1[i][0][1] = corners_1[i][0][1] + 500
            
        # Offset for image right
        
        for i in range(len(corners_2)):
            corners_2[i][0][1] = corners_2[i][0][1] + 500
            corners_2[i][0][0] = corners_2[i][0][0] + 1000
            
        # Precise corners
        new_corners_1 = cv2.cornerSubPix(image = gray, corners = corners_1, winSize = (4, 4), zeroZone = (-1, -1), criteria = criteria)
        new_corners_2 = cv2.cornerSubPix(image = gray, corners = corners_2, winSize = (4, 4), zeroZone = (-1, -1), criteria = criteria)
        
        # Reshaping
        new_corners_1 = new_corners_1.reshape(-1, 2)
        new_corners_2 = new_corners_2.reshape(-1, 2)
        
        # Combining the corners of two images
        img_coord = np.append(new_corners_1, new_corners_2, axis = 0)
        
        

    
    
    
    
    

    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation
    
    world = [[0, 40, 40],
         [0, 30, 40],
         [0, 20, 40],
         [0, 10, 40],
         [0, 40, 30],
         [0, 30, 30],
         [0, 20, 30],
         [0, 10, 30],
         [0, 40, 20],
         [0, 30, 20],
         [0, 20, 20],
         [0, 10, 20],
         [0, 40, 10],
         [0, 30, 10],
         [0, 20, 10],
         [0, 10, 10],
         [40, 0, 40],
         [40, 0, 30],
         [40, 0, 20],
         [40, 0, 10],
         [30, 0, 40],
         [30, 0, 30],
         [30, 0, 20],
         [30, 0, 10],
         [20, 0, 40],
         [20, 0, 30],
         [20, 0, 20],
         [20, 0, 10],
         [10, 0, 40],
         [10, 0, 30],
         [10, 0, 20],
         [10, 0, 10],]
    
    world_coord = np.array([world], dtype = float).reshape(32, 3)

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation       # world_coord   # img_coord
    
    A = np.zeros((64, 12))
    
    counteri = 0
    for i in range(0, 64, 2):
        
        A[i][0] = world_coord[counteri][0] 
        A[i][1] = world_coord[counteri][1] 
        A[i][2] = world_coord[counteri][2]
        A[i][3] = 1
        A[i][4] = 0
        A[i][5] = 0
        A[i][6] = 0
        A[i][7] = 0
        A[i][8] = -(np.dot(img_coord[counteri][0], world_coord[counteri][0]))
        A[i][9] = -(np.dot(img_coord[counteri][0], world_coord[counteri][1]))
        A[i][10] = -(np.dot(img_coord[counteri][0], world_coord[counteri][2]))
        A[i][11] = -(img_coord[counteri][0])
        
        A[i+1][0] = 0
        A[i+1][1] = 0
        A[i+1][2] = 0
        A[i+1][3] = 0
        A[i+1][4] = world_coord[counteri][0]
        A[i+1][5] = world_coord[counteri][1]
        A[i+1][6] = world_coord[counteri][2]
        A[i+1][7] = 1
        A[i+1][8] = -(np.dot(img_coord[counteri][1], world_coord[counteri][0]))
        A[i+1][9] = -(np.dot(img_coord[counteri][1], world_coord[counteri][1]))
        A[i+1][10] = -(np.dot(img_coord[counteri][1], world_coord[counteri][2]))
        A[i+1][11] = -(img_coord[counteri][1])
        
        counteri += 1
        
    # Eigen vectors method
    A_ = np.matmul(A.T, A)
    e_vals, e_vecs = np.linalg.eig(A_)
    
    # Eigen value minimum
    mini = e_vecs[:, 11]
    projection_matrix = mini.reshape(3, 4)
    
    # Calculating
    m1 = projection_matrix[0, 0:3 ].reshape(3, 1)
    m2 = projection_matrix[1, 0:3 ].reshape(3, 1)
    m3 = projection_matrix[2, 0:3 ].reshape(3, 1)

    ox = np.dot(m1.T, m3)
    oy = np.dot(m2.T, m3)
    fx = np.sqrt((np.dot(m1.T, m1)) - ox**2)
    fy = np.sqrt((np.dot(m2.T, m2)) - oy**2)

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation
    
    A = np.zeros((64, 12))
    
    counteri = 0
    for i in range(0, 64, 2):
        
        A[i][0] = world_coord[counteri][0] 
        A[i][1] = world_coord[counteri][1] 
        A[i][2] = world_coord[counteri][2]
        A[i][3] = 1
        A[i][4] = 0
        A[i][5] = 0
        A[i][6] = 0
        A[i][7] = 0
        A[i][8] = -(np.dot(img_coord[counteri][0], world_coord[counteri][0]))
        A[i][9] = -(np.dot(img_coord[counteri][0], world_coord[counteri][1]))
        A[i][10] = -(np.dot(img_coord[counteri][0], world_coord[counteri][2]))
        A[i][11] = -(img_coord[counteri][0])
        
        A[i+1][0] = 0
        A[i+1][1] = 0
        A[i+1][2] = 0
        A[i+1][3] = 0
        A[i+1][4] = world_coord[counteri][0]
        A[i+1][5] = world_coord[counteri][1]
        A[i+1][6] = world_coord[counteri][2]
        A[i+1][7] = 1
        A[i+1][8] = -(np.dot(img_coord[counteri][1], world_coord[counteri][0]))
        A[i+1][9] = -(np.dot(img_coord[counteri][1], world_coord[counteri][1]))
        A[i+1][10] = -(np.dot(img_coord[counteri][1], world_coord[counteri][2]))
        A[i+1][11] = -(img_coord[counteri][1])
        
        counteri += 1
        
    # Eigen vectors method
    A_ = np.matmul(A.T, A)
    e_vals, e_vecs = np.linalg.eig(A_)
    
    # Eigen value minimum
    mini = e_vecs[:, 11]
    projection_matrix = mini.reshape(3, 4)
    
    
    decomposed = np.linalg.qr(projection_matrix)
    R = decomposed[0]
    T = decomposed[1][:, -1]

    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2
  





#---------------------------------------------------------------------------------------------------------------------