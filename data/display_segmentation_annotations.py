import json
from PIL import Image
from matplotlib.patches import Polygon, Rectangle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# displays the segmentation annotation 

data_path = "./train.json"

with open(data_path, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

image_annotations = {}
for annotation in annotations:
    image_id = annotation["image_id"]
    if image_id not in image_annotations:
        image_annotations[image_id] = []
    image_annotations[image_id].append(annotation)

cmap = cm.get_cmap('tab20')

for image_data in images:
    image_id = image_data["id"]
    file_name = image_data["file_name"]
    image = Image.open(file_name)

    if image_id in image_annotations:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')

        annotations = image_annotations[image_id]
        unique_category_ids = list({annotation["category_id"] for annotation in annotations})
        color_map = {unique_id: cmap(i / len(unique_category_ids)) for i, unique_id in enumerate(unique_category_ids)}

        category_to_segmentation = {}
        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id not in category_to_segmentation:
                category_to_segmentation[category_id] = []
            category_to_segmentation[category_id].append(annotation["segmentation"])

        for category_id, segmentations in category_to_segmentation.items():
            color = color_map[category_id]

            all_x = []
            all_y = []
            for segmentation in segmentations:

                if isinstance(segmentation[0], list):
                    segmentation = [point for sublist in segmentation for point in sublist]
                try:
                    segmentation = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
                except TypeError:
                    continue  
                poly = Polygon(segmentation, edgecolor=color, facecolor='none')
                plt.gca().add_patch(poly)

                all_x.extend([p[0] for p in segmentation])
                all_y.extend([p[1] for p in segmentation])

        for annotation in annotations:
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]
            color = color_map[category_id]

            x, y, w, h = bbox
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)

            plt.text(x, y, f"x: {x:.1f}, y: {y:.1f}", color=color, fontsize=12, verticalalignment='top', horizontalalignment='left')
            plt.text(x + w, y + h, f"w: {w:.1f}, h: {h:.1f}", color=color, fontsize=12, verticalalignment='bottom', horizontalalignment='right')

        plt.savefig(f'xywh/seg_{image_id}.png', bbox_inches='tight')
        plt.close()