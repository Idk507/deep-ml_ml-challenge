"""
Implement the PCA color distortion technique used in AlexNet for data augmentation. This method applies PCA to the RGB pixel values of natural images, then adds multiples of the principal components with magnitudes proportional to the corresponding eigenvalues times a random variable. Given an RGB image (H, W, 3), compute the principal components of RGB values and apply the color distortion with specified alpha values.

Example:
Input:
np.random.seed(42)
image = np.random.randint(0, 256, (2, 2, 3)).astype(np.uint8)
alpha = np.array([0.1, -0.05, 0.03])
result = pca_color_augmentation(image, alpha)
print(result)
Output:
[[[ 97  98  99]
  [144 145 146]]

 [[243 244 245]
  [ 32  33  34]]]
Reasoning:
The function computes PCA on the RGB pixel values, then applies a color distortion based on the principal components weighted by alpha values. The distortion is added to all pixels and the result is clamped to [0, 255].

Learn About topic
Understanding AlexNet PCA Color Augmentation
PCA Color Augmentation is a data augmentation technique introduced in AlexNet that performs sophisticated color space transformations to increase dataset diversity and improve model robustness.

Mathematical Foundation
Step 1: Extract RGB pixel values For an RGB image of shape 
(
H
,
W
,
3
)
(H,W,3), reshape to get all pixel values as a matrix 
X
X of shape 
(
H
×
W
,
3
)
(H×W,3) where each row represents an RGB pixel.

Step 2: Compute mean-centered data 
X
ˉ
=
X
−
μ
X
ˉ
 =X−μ where 
μ
μ is the mean RGB value across all pixels.

Step 3: Compute covariance matrix 
C
=
1
n
−
1
X
ˉ
T
X
ˉ
C= 
n−1
1
​
  
X
ˉ
  
T
  
X
ˉ
  where 
n
=
H
×
W
n=H×W is the number of pixels.

Step 4: Eigen decomposition 
C
=
P
Λ
P
T
C=PΛP 
T
  where:

P
P = eigenvectors (principal components) of shape 
(
3
,
3
)
(3,3)
Λ
Λ = diagonal matrix of eigenvalues 
[
λ
1
,
λ
2
,
λ
3
]
[λ 
1
​
 ,λ 
2
​
 ,λ 
3
​
 ]
Step 5: Generate color distortion 
distortion
=
P
⋅
(
α
⊙
Λ
)
distortion=P⋅(α⊙ 
Λ
​
 ) where:

α
=
[
α
1
,
α
2
,
α
3
]
α=[α 
1
​
 ,α 
2
​
 ,α 
3
​
 ] are random coefficients (typically sampled from 
N
(
0
,
0.1
)
N(0,0.1))
⊙
⊙ denotes element-wise multiplication
Λ
=
[
λ
1
,
λ
2
,
λ
3
]
Λ
​
 =[ 
λ 
1
​
 
​
 , 
λ 
2
​
 
​
 , 
λ 
3
​
 
​
 ]
Step 6: Apply augmentation 
X
augmented
=
X
+
distortion
X 
augmented
​
 =X+distortion

Why PCA Color Augmentation?
Natural Color Variations: Captures natural lighting variations in real images
Illumination Invariance: Makes models robust to different lighting conditions
Reduced Overfitting: Increases effective dataset size through meaningful augmentations
Preserves Object Identity: Color changes are realistic and don't alter object semantics
Key Properties
Principal Component 1: Usually captures overall brightness/intensity
Principal Component 2: Often represents color temperature (warm vs cool)
Principal Component 3: Captures saturation and specific color shifts
AlexNet Results
In the original AlexNet paper, this technique:

Reduced top-1 error rate by over 1%
Made the network more robust to lighting variations
Helped achieve state-of-the-art ImageNet performance
Implementation Notes
Apply only during training, not inference
Typical 
α
α values: 
α
i
∼
N
(
0
,
0.1
)
α 
i
​
 ∼N(0,0.1)
Can be combined with other augmentations (rotation, cropping, etc.)
Clamp final pixel values to valid range [0, 255]
Modern Usage
While newer augmentation techniques exist (AutoAugment, RandAugment), PCA color augmentation remains:

A standard baseline in computer vision
Effective for medical imaging where color variations matter
Useful in domains with limited training data


"""

import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image.
    
    Args:
        image: RGB image of shape (H, W, 3) with values in [0, 255]
        alpha: Array of 3 random coefficients for principal components
    
    Returns:
        Augmented image of shape (H, W, 3) with values clamped to [0, 255]
    """
    # Store original shape
    orig_shape = image.shape
    H, W, C = orig_shape
    
    # Reshape image to (H*W, 3) - each row is an RGB pixel
    pixels = image.reshape(-1, 3).astype(np.float64)
    
    # Compute mean of each channel
    mean = np.mean(pixels, axis=0)
    
    # Center the data
    centered = pixels - mean
    
    # Compute covariance matrix (3x3)
    # C = (1/(n-1)) * X^T * X
    cov_matrix = np.cov(centered.T)
    
    # Perform eigendecomposition
    # eigenvalues: array of 3 values
    # eigenvectors: matrix of shape (3, 3), each column is an eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    # (eigh returns them in ascending order)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute color distortion
    # distortion = P * (alpha ⊙ sqrt(λ))
    # where P is the eigenvector matrix, λ are eigenvalues
    distortion = eigenvectors @ (alpha * np.sqrt(np.maximum(eigenvalues, 0)))
    
    # Apply distortion to all pixels
    augmented_pixels = pixels + distortion
    
    # Reshape back to original image shape
    augmented_image = augmented_pixels.reshape(orig_shape)
    
    # Clamp values to [0, 255] and convert to uint8
    augmented_image = np.clip(augmented_image, 0, 255)
    
    return augmented_image
