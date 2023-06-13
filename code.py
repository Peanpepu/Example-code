from PIL import Image
from ultralytics import YOLO
import cv2

# Load the pre-trained model
# model_path = os.path.join("/root/TFG/results/train_rgb/weights/best.pt") # File .pt path obtained after the training
model = YOLO("/root/TFG/results/train_rgb/weights/best.pt")

# Load the image
image_path = "/root/TFG/data/dataset_rgb/images/test/frame2157.jpg"  # Image path for processing
image = cv2.imread(image_path)

# Perform detections on the image
results = model.predict(source=image, save=True, save_txt=True, save_conf=True)

# Function to check the intersection of two bounding boxes
def check_intersection(detected_objects):
    for i, ((x, y, w, h), class_id) in enumerate(detected_objects):
        # Calculate the area of the current object
        area_current = w * h
        for j, ((prev_x, prev_y, prev_w, prev_h), prev_class_id) in enumerate(detected_objects[i+1:], i+1):
            # Calculate the area of the last object
            area_prev = prev_w * prev_h
        
           # Calculate the intersection between the objects
        intersection_x = max(x, prev_x)
        intersection_y = max(y, prev_y)
        intersection_w = min(x+w, prev_x+prev_w) - intersection_x
        intersection_h = min(y+h, prev_y+prev_h) - intersection_y
    
        # Calculate the area of the intersection
        intersection_area = max(0, intersection_w) * max(0, intersection_h)
    
        # Calculate the overlap percentage
        overlap_percentage = intersection_area / area_current * 100
    
        if overlap_percentage >= 70:
            # Display the overlap percentage
            print(f"Object {i+1} and Object {j+1}: {overlap_percentage}% overlap")

# Below is a function to draw bounding boxes on the original image based on the txt file
def draw_bounding_boxes(image_path, annotations_file, output_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Get the dimensions of the image
    height, width, _ = img.shape

    # Declare classes with their corresponding indices
    classes = {
        0: 'emergency vehicle',
        1: 'non-emergency vehicle',
        2: 'first responder',
        3: 'non-first responder'
    }

    # Define colors for each class
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    # Read the annotations from the YOLO annotations file
    with open(annotations_file, 'r') as f:
        annotations = f.read().splitlines()

    # List to store the location and area of each detected object
    detected_objects = []

    for annotation in annotations:

        # Get the information for each annotation
        class_id, x_center, y_center, bbox_width, bbox_height, conf = map(float, annotation.split())
        conf = round(conf, 2) # Round the confidence to 2 decimal places

        # Calculate the coordinates of the bounding box
        x = int((x_center - bbox_width/2) * width) # Multiply because bbox coordinates are relative to its 
        # width and height
        y = int((y_center - bbox_height/2) * height)
        w = int(bbox_width * width)
        h = int(bbox_height * height)
    
        # Store the location and area of the detected object in the list
        detected_objects.append(((x, y, w, h), class_id))

        # Get the color corresponding to each class
        color = colors[int(class_id)]
    
        # Draw the bounding box on the image
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # Get the text and text color
        text = classes[int(class_id)] + " " + str(conf)
        text_color = (255, 255, 255)  # White color

        # Calculate the text size
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

        # Calculate the coordinates for the text rectangle
        rect_x = x
        rect_y = y - text_size[1] - 10
        rect_width = text_size[0] + 10
        rect_height = text_size[1] + 10

        # Draw the text rectangle
        cv2.rectangle(img, (rect_x, rect_y), (rect_x+rect_width, rect_y+rect_height), color, -1)

        # Draw the text with the corresponding color
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    check_intersection(detected_objects)

    # Save the resulting image with the bounding boxes
    cv2.imwrite(output_path, img)


# Now call the function
annotation_path = "/root/ultralytics/runs/detect/predict/labels/frame2157.txt"
output_path = "/root/TFG/results/pruebas/pred_frame2157.jpg"
im = image_path

draw_bounding_boxes(im, annotation_path, output_path)
