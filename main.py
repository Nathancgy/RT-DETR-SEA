import cv2
from ultralytics import RTDETR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import csv

# Initialize the model
model = RTDETR("weights/vid.pt")

# Define class names
class_names = ['can', 'carton', 'p-bag', 'p-bottle', 'p-con', 'styrofoam', 'tire']

# Initialize dictionaries to store positions, class IDs, and uncertainties
positions = {}                # Stores the relative positions of debris keyed by track IDs
class_ids = {}                # Maps track IDs to class IDs
covariances = {}              # Stores covariance matrices for each debris object
previous_frame_centers = {}   # Stores centers of debris in the previous frame
previous_frame_covariances = {}  # Stores covariances of debris in the previous frame

# Scaling factor for uncertainty estimation
k = 50  # Temporary value; adjust based on your data

# Mahalanobis distance threshold for clustering
tau = 15  # Corresponds to 99.7% confidence interval in a Gaussian distribution

# Open the video file
input_video = 'vid.mp4'
cap = cv2.VideoCapture(input_video)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    current_frame_centers = {}
    current_frame_covariances = {}
    frame_displacements = []

    # Process detections in the current frame
    if hasattr(results[0], 'boxes'):
        for obj in results[0].boxes:
            class_id = int(obj.cls)
            class_name = class_names[class_id]
            track_id = int(obj.id)
            confidence = float(obj.conf)
            bbox = obj.xyxy.tolist()[0]
            x1, y1, x2, y2 = map(int, bbox)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center = np.array([center_x, center_y], dtype=np.float64)
            current_frame_centers[track_id] = center
            class_ids[track_id] = class_id

            # Estimate covariance matrix based on detection confidence
            sigma_sq = k / confidence
            covariance = np.array([[sigma_sq, 0], [0, sigma_sq]])
            current_frame_covariances[track_id] = covariance

    if not current_frame_centers:
        frame_count += 1
        continue

    # Initialize positions in the first frame
    if frame_count == 0:
        # Randomly select one object to assign position (0, 0)
        track_ids = list(current_frame_centers.keys())
        random_track_id = random.choice(track_ids)
        positions[random_track_id] = np.array([0.0, 0.0], dtype=np.float64)
        covariances[random_track_id] = current_frame_covariances[random_track_id]
        first_center = current_frame_centers[random_track_id]
        
        # Compute positions for other objects relative to the selected one
        for track_id, curr_center in current_frame_centers.items():
            if track_id != random_track_id:
                displacement = curr_center - first_center
                positions[track_id] = positions[random_track_id] + displacement
                covariances[track_id] = current_frame_covariances[track_id]
        previous_frame_centers = current_frame_centers.copy()
        previous_frame_covariances = current_frame_covariances.copy()
        frame_count += 1
        continue

    # Compute displacements for known debris to estimate camera motion
    for track_id in current_frame_centers:
        if track_id in previous_frame_centers:
            prev_center = previous_frame_centers[track_id]
            curr_center = current_frame_centers[track_id]
            displacement = curr_center - prev_center
            frame_displacements.append(displacement)

    # Estimate camera motion
    if len(frame_displacements) > 0:
        camera_motion = np.mean(frame_displacements, axis=0)
    else:
        camera_motion = np.array([0.0, 0.0], dtype=np.float64)

    # Update positions of debris
    known_ids_in_frame = set(positions.keys()).intersection(current_frame_centers.keys())

    for track_id, curr_center in current_frame_centers.items():
        if track_id in positions:
            # Known debris: adjust position for camera motion
            positions[track_id] -= camera_motion
            # Covariance remains the same for this step
        else:
            # New debris: compute position relative to known debris
            if known_ids_in_frame:
                estimated_positions = []
                estimated_covariances = []
                for known_id in known_ids_in_frame:
                    known_center = current_frame_centers[known_id]
                    displacement = curr_center - known_center
                    estimated_position = positions[known_id] + displacement - camera_motion
                    estimated_positions.append(estimated_position)
                    estimated_covariances.append(covariances[known_id] + current_frame_covariances[track_id])
                # Bayesian fusion of estimates
                cov_inv_sum = np.zeros((2, 2))
                weighted_pos_sum = np.zeros(2)
                for i in range(len(estimated_positions)):
                    try:
                        cov_inv = np.linalg.inv(estimated_covariances[i])
                    except np.linalg.LinAlgError:
                        # If covariance matrix is singular, skip this estimate
                        continue
                    cov_inv_sum += cov_inv
                    weighted_pos_sum += cov_inv @ estimated_positions[i]
                if cov_inv_sum.any():
                    fused_covariance = np.linalg.inv(cov_inv_sum)
                    fused_position = fused_covariance @ weighted_pos_sum
                    positions[track_id] = fused_position
                    covariances[track_id] = fused_covariance
                else:
                    # If all covariance inversions failed, assign based on camera motion
                    positions[track_id] = np.array([0.0, 0.0], dtype=np.float64)
                    covariances[track_id] = current_frame_covariances[track_id]
            else:
                # Assign (0,0) if no known debris is available
                positions[track_id] = np.array([0.0, 0.0], dtype=np.float64)
                covariances[track_id] = current_frame_covariances[track_id]
        # Ensure class_id is recorded
        if track_id not in class_ids:
            class_ids[track_id] = -1  # Unknown class ID

    previous_frame_centers = current_frame_centers.copy()
    previous_frame_covariances = current_frame_covariances.copy()
    frame_count += 1

    print("Frame Count:", frame_count)

cap.release()

# Prepare data for clustering
positions_list = []
for track_id in positions.keys():
    position = positions[track_id]
    covariance = covariances[track_id]
    class_id = class_ids.get(track_id, -1)
    positions_list.append({'position': position, 'covariance': covariance, 'class_id': class_id, 'track_id': track_id})

# Clustering with uncertainty integration
final_positions = []
used_indices = set()

for i, p1 in enumerate(positions_list):
    if i in used_indices:
        continue
    # Removed class_id consideration for clustering
    pos1 = p1['position']
    cov1 = p1['covariance']
    track_ids_cluster = {p1['track_id']}  # Initialize with the first track ID
    cluster_positions = [pos1]
    cluster_covariances = [cov1]
    for j, p2 in enumerate(positions_list):
        if j <= i or j in used_indices:
            continue
        pos2 = p2['position']
        cov2 = p2['covariance']
        # Compute Mahalanobis distance
        delta = pos1 - pos2
        cov_sum = cov1 + cov2
        try:
            inv_cov_sum = np.linalg.inv(cov_sum)
        except np.linalg.LinAlgError:
            # If covariance matrix is singular, skip this pair
            continue
        distance = np.sqrt(delta.T @ inv_cov_sum @ delta)
        if distance < tau:
            cluster_positions.append(pos2)
            cluster_covariances.append(cov2)
            track_ids_cluster.add(p2['track_id'])
            used_indices.add(j)
    # Bayesian fusion of cluster estimates
    cov_inv_sum = np.zeros((2, 2))
    weighted_pos_sum = np.zeros(2)
    for idx in range(len(cluster_positions)):
        try:
            cov_inv = np.linalg.inv(cluster_covariances[idx])
        except np.linalg.LinAlgError:
            # If covariance matrix is singular, skip this position
            continue
        cov_inv_sum += cov_inv
        weighted_pos_sum += cov_inv @ cluster_positions[idx]
    if cov_inv_sum.any():
        fused_covariance = np.linalg.inv(cov_inv_sum)
        fused_position = fused_covariance @ weighted_pos_sum
    else:
        fused_covariance = np.array([[0.0, 0.0], [0.0, 0.0]])
        fused_position = np.array([0.0, 0.0])
    # Concatenate track IDs into a string
    if len(track_ids_cluster) == 1:
        ids_str = f"{list(track_ids_cluster)[0]}"
    else:
        ids_str = "{" + ", ".join(map(str, sorted(track_ids_cluster))) + "}"
    # Determine class_id for the cluster
    # Option 1: Assign the class_id of the first track in the cluster
    # Option 2: Assign the most frequent class_id in the cluster
    # Here, we'll use Option 2 for better accuracy
    cluster_class_ids = [p['class_id'] for p in positions_list if p['track_id'] in track_ids_cluster]
    if cluster_class_ids:
        unique, counts = np.unique(cluster_class_ids, return_counts=True)
        dominant_class_id = unique[np.argmax(counts)]
    else:
        dominant_class_id = -1  # Unknown
    final_positions.append({'position': fused_position, 'covariance': fused_covariance, 'class_id': dominant_class_id, 'ids': ids_str})

# Save final points and other information to a CSV file
with open('debris_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['x_position', 'y_position', 'covariance_xx', 'covariance_xy', 'covariance_yy', 'class_id', 'class_name', 'track_ids']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in final_positions:
        position = item['position']
        covariance = item['covariance']
        class_id = item['class_id']
        class_name = class_names[class_id] if class_id != -1 else 'Unknown'
        ids_str = item['ids']

        writer.writerow({
            'x_position': position[0],
            'y_position': position[1],
            'covariance_xx': covariance[0, 0],
            'covariance_xy': covariance[0, 1],
            'covariance_yy': covariance[1, 1],
            'class_id': class_id,
            'class_name': class_name,
            'track_ids': ids_str
        })

print("Data saved to debris_data.csv")

# Plot the 2D map of debris with uncertainty ellipses and IDs
# Updated color palette excluding 'yellow' and adding 'magenta'
colors = ['red', 'green', 'blue', 'magenta', 'purple', 'orange', 'cyan']

plt.figure(figsize=(12, 10))

# To avoid duplicate labels in the legend, keep track of already plotted class_ids
plotted_classes = set()

for item in final_positions:
    position = item['position']
    covariance = item['covariance']
    class_id = item['class_id']
    ids_str = item['ids']
    color = colors[class_id % len(colors)] if class_id != -1 else 'black'
    class_label = class_names[class_id] if class_id != -1 else 'Unknown'
    # Plot only one instance per class for the legend
    if class_label not in plotted_classes:
        plt.scatter(position[0], -position[1], color=color, s=100, label=class_label)
        plotted_classes.add(class_label)
    else:
        plt.scatter(position[0], -position[1], color=color, s=100)

    # Plot covariance ellipse if covariance matrix is positive definite
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if np.all(eigenvalues > 0):
            angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
            width, height = 2 * tau * np.sqrt(eigenvalues)
            ellipse = patches.Ellipse(xy=(position[0], -position[1]), width=width, height=height,
                                      angle=-angle, edgecolor=color, fc='None', lw=2)
            plt.gca().add_patch(ellipse)
    except np.linalg.LinAlgError:
        # Skip plotting ellipse if covariance matrix is not valid
        pass

    # Annotate the point with its ID(s)
    plt.text(position[0] + 5, -position[1] + 5, ids_str, fontsize=9, color=color)

# Create a custom legend to avoid duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('2D Map of Marine Debris with Uncertainty and IDs')
plt.grid(True)
plt.show()
