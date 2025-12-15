import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------
# 1. SETUP: Define your file paths here
# ---------------------------------------------------------
# Replace these strings with the actual paths to your JSON files
FILE_FOG = "./fog-removal/fog_dataset/labels.json"
FILE_RAIN = "./rain_annotations.json"
FILE_LOW_LIGHT = "./low_light_annotations.json"

# The 10 standard BDD100K Object Detection classes
CLASSES = [
    "Pedestrian",
    "Rider",
    "Car",
    "Truck",
    "Bus",
    "Train",
    "Motorcycle",
    "Bicycle",
    "Traffic Light",
    "Traffic Sign",
]


def get_class_counts(json_path):
    """
    Parses a BDD100K JSON file and counts occurrences of each class.
    Returns a dictionary of counts and the total number of annotations.
    """
    counts = defaultdict(int)
    total_annotations = 0

    with open(json_path, "r") as f:
        data = json.load(f)

    for annotation in data["annotations"]:
        category = CLASSES[annotation["category_id"] - 1]
        counts[category] += 1
        total_annotations += 1

    return counts, total_annotations


# ---------------------------------------------------------
# 2. PROCESS DATA
# ---------------------------------------------------------
print("Processing Fog dataset...")
fog_counts, fog_total = get_class_counts(FILE_FOG)

print("Processing Rain dataset...")
rain_counts, rain_total = get_class_counts(FILE_RAIN)

print("Processing Low Light dataset...")
light_counts, light_total = get_class_counts(FILE_LOW_LIGHT)

# Calculate Percentages
# We use a list comprehension to ensure the order matches 'CLASSES'
fog_pct = [(fog_counts[c] / fog_total) * 100 for c in CLASSES]
rain_pct = [(rain_counts[c] / rain_total) * 100 for c in CLASSES]
light_pct = [(light_counts[c] / light_total) * 100 for c in CLASSES]

# ---------------------------------------------------------
# 3. PLOTTING
# ---------------------------------------------------------
x = np.arange(len(CLASSES))  # Label locations
width = 0.25  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Create the three bars per class
# We offset the x position for each group
rects1 = ax.bar(
    x - width, fog_pct, width, label="Fog", color="#b0bec5"
)  # Grey-blue for fog
rects2 = ax.bar(x, rain_pct, width, label="Rain", color="#1e88e5")  # Blue for rain
rects3 = ax.bar(
    x + width, light_pct, width, label="Low Light", color="#5e35b1"
)  # Purple for low light

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Percentage of Total Annotations (%)")
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add a grid for easier reading
ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)

# Layout adjustment to prevent label clipping
plt.tight_layout()

# Show (or save) the plot
plt.savefig("class_distributions.png")
