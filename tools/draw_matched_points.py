import sys
import numpy as np
import cv2

sys.path.append('.')

from perspective_transform.calibration import calibration_position

# Load the two images and the matching point pairs
image1 = cv2.imread('path/to/cam_example.jpg')
image2 = cv2.imread('path/to/map.png')
points1 = np.array(calibration_position["S020"][0]["cam_position"])  # calibration_position["space_name"][cam_index]
points2 = np.array(calibration_position["S020"][0]["map_position"])

# Compute the maximum height of the two images
max_height = max(image1.shape[0], image2.shape[0])

# Create a new blank image with the combined width and maximum height
combined_image = np.zeros((max_height, image1.shape[1]+image2.shape[1], 3), dtype=np.uint8)

# Copy the first image onto the left half of the combined image
combined_image[0:image1.shape[0], 0:image1.shape[1], :] = image1

# Copy the second image onto the right half of the combined image
combined_image[0:image2.shape[0], image1.shape[1]:, :] = image2
combined_image_h = combined_image.copy()

# Shift the coordinates of points2 to account for the offset of the second image
points2_origin = points2.copy()
points2[:, 0] += image1.shape[1]

# Draw lines connecting the matching points
for i in range(len(points1)):
    color = tuple(np.random.randint(0, 255, 3).tolist())  # Choose a random color for each line
    pt1 = tuple(map(int, points1[i]))
    pt2 = tuple(map(int, points2[i]))
    cv2.line(combined_image, pt1, pt2, color, 2)
    cv2.circle(combined_image, pt1, 7, color, -1)
    cv2.circle(combined_image, pt2, 7, color, -1)

# Save the result
cv2.imwrite('tools/outputs/match_point.jpg', combined_image)

H, mask = cv2.findHomography(points1, points2_origin, cv2.USAC_MAGSAC, 10.0, maxIters=100000)  # USAC_ACCURATE, USAC_MAGSAC, RANSAC / threshold / maxIters
print("Homography Matrix: ", H)
print(f"condition number: {np.linalg.cond(H)}")
print(f"inliers number: {mask.sum()} / {len(mask)}")

pts = np.float32([ [1920, 1080] ]).reshape(-1,1,2)  # predict map's coordinate using above Homography Matrix
dst = cv2.perspectiveTransform(pts, H)
print(f"predicted points of {pts}: {dst}")

mask = [m[0]==1 for m in mask]
print(f"Before Homography\nCam: {points1.tolist()}\nMap: {points2_origin.tolist()}")
points1 = points1[mask]
points2 = points2[mask]
points2_origin = points2_origin[mask]
print(f"After Homography\nCam: {points1.tolist()}\nMap: {points2_origin.tolist()}")

# Draw lines connecting the matching points
for i in range(len(points1)):
    color = tuple(np.random.randint(0, 255, 3).tolist())  # Choose a random color for each line
    pt1 = tuple(map(int, points1[i]))
    pt2 = tuple(map(int, points2[i]))
    cv2.line(combined_image_h, pt1, pt2, color, 2)
    cv2.circle(combined_image_h, pt1, 7, color, -1)
    cv2.circle(combined_image_h, pt2, 7, color, -1)

cv2.imwrite('tools/outputs/match_point_filterd.jpg', combined_image_h)


imageA = image1.copy()
imageB = image2.copy()

# Get the dimensions of image A
heightA, widthA, channelsA = imageA.shape

# Create a mask for image A
maskA = np.ones((heightA, widthA), dtype=np.uint8) * 255

# Warp image A onto image B
warpedA = cv2.warpPerspective(imageA, H, (imageB.shape[1], imageB.shape[0]))

# Warp the mask of image A onto image B
warpedMaskA = cv2.warpPerspective(maskA, H, (imageB.shape[1], imageB.shape[0]))

# Create a mask for image B
maskB = np.ones_like(warpedMaskA) * 255

# Combine the warped image A and the original image B using the warped mask of image A
# mergedImage = cv2.bitwise_and(warpedA, warpedA, mask=warpedMaskA)*0.5 + \
#               cv2.bitwise_and(imageB, imageB, mask=cv2.bitwise_not(warpedMaskA))
mergedImage = cv2.bitwise_and(warpedA, warpedA, mask=warpedMaskA)*1 + imageB*0.5
mergedImage.astype(np.uint8)

# Display the result
cv2.imwrite('tools/outputs/merged_image.jpg', mergedImage)