from django.core.cache import cache
# Clear all cache keys
cache.clear()

from django.shortcuts import render

from django.conf import settings
import os

from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response



import requests
from urllib.parse import urlparse
from pathlib import Path


import pyrebase
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

#NOTE - Firebase
firebase = pyrebase.initialize_app(
    {
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
    }
)
storage = firebase.storage()

api_key = os.getenv("FIREBASE_API_KEY")
print("API_KEY:", api_key)

import shutil
# shutil.rmtree('run')

from ultralytics import YOLO
from torchvision import torch, models, transforms
import torch.nn as nn
from PIL import Image

#FIXME - DONT TOUCH THIS
# class serve_image_and_label(APIView):
    
#     def post(request, *args, **kwargs):

#         if not request.image_path:
#             return Response({"error": "Path not provided"}, status=status.HTTP_400_BAD_REQUEST)

#         if not os.path.exists(image_path):
#             return Response({"error": "Detected image file not found"}, status=status.HTTP_404_NOT_FOUND)

#         # Derive the text file path (same name as image, different extension)
#         base_name = os.path.splitext(os.path.basename(image_path))[0]
#         parent_path = os.path.dirname(os.path.dirname(image_path))
#         label_filename = f"{base_name}.txt"
#         label_path = f"{parent_path}\labels\{label_filename}"   
#         print ('label_path: %s' % label_path)
#         print ('image_path: %s' % image_path)
#         # Check if the text file exists
#         if not os.path.exists(label_path):
#             return Response({"error": "Label file not found"}, status=status.HTTP_404_NOT_FOUND)

#         # Prepare the file responses
#         # image_url = request.build_absolute_uri(image_instance.images.url)
#         label_url = request.build_absolute_uri(os.path.join(settings.MEDIA_URL, 'labels', label_filename).replace('\\', '/'))

#         # return Response({
#         #     "images": image_url,
#         #     "labels": label_url
#         # })


def save_file_from_FB(image_url):

        try:
            # Download image
            response = requests.get(image_url)
            if response.status_code != 200:
                return Response({"error": "Failed to download image"}, status=400)

            # Save image temporarily
            parsed_url = urlparse(image_url)
            image_name = os.path.basename(parsed_url.path)

            #NOTE - Firebase pure image container

            dir_to_remove = "pure_images%2F"
            image_name = image_name.replace(dir_to_remove,"")


            # Define the local path to save the image
            media_path = Path(settings.MEDIA_ROOT) / "img_to_detect" / image_name

            # Write the image content to the media folder
            with open(media_path, 'wb') as file:
                file.write(response.content)

            # Return a success response with the saved image path
            return media_path

        except Exception as e:
            return Response({"error": str(e)}, status=500)
def upload_to_firebase(file_path):

    file_path = file_path 
    image_name = os.path.basename(file_path)
    
    #NOTE - Firebase detected images container

    storage.child(f"result/detected_img/{image_name}").put(file_path)
    image_url = storage.child(f"result/detected_img/{image_name}").get_url(None)
    
    return image_url

import os

def count_crops_in_acne_classes(base_directory):
    result = {}
    for root, dirs, files in os.walk(base_directory):
        # os.walk(base_directory) duyệt qua cây thư mục từ base và trả về 3 tuples:
        #    root là thư mục hiện tại
        #    dirs là danh sách các thư mục con của thư mục hiện tại
        #    files là danh sách các tập tin trong thư mục hiện tại

        # nếu root hiện tại không có dirs (không có con => thư mục acne classes)
        if not dirs:
            folder_name = os.path.basename(root)  # Lấy tên thư mục
            file_count = len(files)  # Đếm số lượng file trong thư mục đó
            result[folder_name] = file_count  # Lưu kết quả vào dictionary
    return result

#SECTION - Key for sort function, use to sort vectors (crop images)
import re

def extract_last_numeric_value_key(s):
    # Extract the part of the string after the '/'
    file_name = s.split('/')[-1]
    
    
    # Extract the last numeric value from this part
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0


def extract_first_numeric_value(list):
    result = []
    for s in list:
    # Extract the last numeric value from this part
        match = re.search(r'\d+', s)
        result.append(int(match.group()) if match else 0)
    return result
# # List of strings to be sorted
# strings = ['1xx/yy2', '2xx/yy1']

# # Sort the list using the custom key function
# sorted_strings = sorted(strings, key=extract_last_numeric_value)


def list_crops_with_classes(root_dir):
    crop_list = []
    
    for root, dirs, files in os.walk(root_dir):

        for file in files:
            # Get the parent directory name
            parent_dir = os.path.basename(root)
            # Form the string as 'parent_dir/filename'
            file_with_parent = f'{parent_dir}/{file}'
            crop_list.append(file_with_parent)
    
    return crop_list


def replace_crop_labels(file_path, label):

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    if len(lines) != len(label):
        raise ValueError("The length of class array must match the number of vectors in label file.")
    
    updated_lines = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) > 1:  # Ensure there's at least one vector component
            coordinates = " ".join(parts[1:])  # Keep the vector part unchanged
            updated_line = f"{label[i]} {coordinates}"
            updated_lines.append(updated_line)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write("\n".join(updated_lines))


def copy_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"File copied from '{source_path}' to '{destination_path}'.")
    except FileNotFoundError:
        print(f"File '{source_path}' not found.")
    except PermissionError:
        print(f"Permission denied while copying '{source_path}' to '{destination_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

import cv2

def draw_bounding_boxes(image_path, label_path, output_path):
    # Check if paths are valid
    if not image_path:
        print("Can't find image path to draw bounding boxes")
        return
    if not label_path:
        print("Can't find label path to draw bounding boxes")
        return
    if not os.path.exists(os.path.dirname(output_path)):
        print("Can't find output directory. Creating directory.")
        os.makedirs(os.path.dirname(output_path))
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image '{image_path}'.")
    
    h, w, _ = image.shape
    
    # Read label file
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label file '{label_path}' not found.")
    
    with open(label_path, 'r') as file:
        labels = file.readlines()

    # Process each label
    for label in labels:
        label = label.strip().split()
        if len(label) != 6:
            print(f"Skipping invalid label: {label}")
            continue
        
        try:
            class_id, x_center, y_center, bbox_width, bbox_height, _ = map(float, label)
        except ValueError as e:
            print(f"Error processing label {label}: {e}")
            continue
        
        # Convert from YOLO format to pixel coordinates
        x1 = int((x_center - bbox_width / 2) * w)
        y1 = int((y_center - bbox_height / 2) * h)
        x2 = int((x_center + bbox_width / 2) * w)
        y2 = int((y_center + bbox_height / 2) * h)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw class ID text
        class_id_text = f'{int(class_id)}'
        text_size = cv2.getTextSize(class_id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = x1
        text_y = y1 - 5 if y1 - 5 > 5 else y1 + 5
        cv2.putText(image, class_id_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save or display the result
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Failed to save image to '{output_path}'")
    
    # Optionally, verify if the file was saved and is not empty
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise IOError(f"Image file '{output_path}' is not saved or is empty.")


class AcneDetectionView(APIView):

    os.makedirs(os.path.join(settings.MEDIA_ROOT,"result_img"), exist_ok=True)

    def post(self, request, *args, **kwargs):
        
        image_url = request.data.get('image_url')

        if not image_url:
            return Response({'error': 'No image url provided.'}, status=status.HTTP_400_BAD_REQUEST)
        
        image_path = save_file_from_FB(image_url)
        
        if os.path.isfile(image_path):
            try:   
                Image.open(image_path)
            except IOError:
                return Response({'error': 'Invalid file format! File is not an image!'}, status=status.HTTP_400_BAD_REQUEST)
        else:    
            return Response({'error': 'No file found at the provided path!'}, status=status.HTTP_400_BAD_REQUEST)
            
        
        yolo_model = YOLO('model/yolo_63Acc.pt')

        yolo_results = yolo_model(image_path, save=True, save_conf=True, save_txt=True, save_crop=True, show_labels=False, show_conf=False, project='run/detect', name='yolo_predict', exist_ok=True)
        
        resnet_model = models.resnet50(pretrained=True)
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)  # Adjust to match the original model's output units
        resnet_model.load_state_dict(torch.load('model/resnet50_Acc_78.pth'))
        resnet_model.eval()    

        crops_path = "run\\detect\\yolo_predict\\crops\\acne"
        crop_dir = os.listdir(crops_path)

        class_names = ['0_blackhead', '1_whitehead', "2_nodule", "3_pustule", "4_papule"]
            
        classify_base_dir = "run\\classify"
            
        for class_name in class_names:
            class_dir = os.path.join(classify_base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        for crop_img in crop_dir:
            crop_img_path = os.path.join(crops_path, crop_img)
            image = Image.open(crop_img_path)
            preprocess = transforms.Compose([
                transforms.Resize(100),
                transforms.CenterCrop(80),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)  

            # Perform inference
            with torch.no_grad():
                output = resnet_model(input_batch)

            # Get the predicted class
            _, predicted_class = output.max(1)

            # Map the predicted class to the class name
            predicted_class_name = class_names[predicted_class.item()]

            # Determine the output directory for this image
            output_dir = os.path.join(classify_base_dir, predicted_class_name)
            
            # Save the image in the appropriate directory
            output_path = os.path.join(output_dir, crop_img)
            image.save(output_path)

            print(f'Saved {crop_img} to {output_path}')

        # detected_img = "run/detect/yolo_predict/" + os.path.basename(image_path)

        # detection_url = upload_to_firebase(detected_img)
        # print(detection_url)

        #SECTION - Return crops quantity based on classes
        number_of_crops = count_crops_in_acne_classes("run/classify/")
        # print(number_of_crops)

        #SECTION - Update yolo label with classes and draw bounding boxes
        crop_list = list_crops_with_classes("run/classify")
        sorted_crops = sorted(crop_list, key=extract_last_numeric_value_key)
        # print(sorted_crops)
        crop_classes = extract_first_numeric_value(sorted_crops)

        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)

        result_img_file = name + ".jpg"

        filename = name + ".txt"
        label_path = os.path.join("run/detect/yolo_predict/labels", filename)
        final_path = os.path.join("media/result_img", result_img_file)
        shutil.copy(label_path, final_path)

        replace_crop_labels(label_path, crop_classes)

        draw_bounding_boxes(image_path, label_path, final_path)

        final_url = upload_to_firebase(final_path)
        # remove data from local
        shutil.rmtree('run')
        try:
            os.remove(final_path)
        except Exception as e:
            print(f"Error occurred while removing file '{final_path}': {e}")


        return Response({"status" : "status.HTTP_200_OK", "data" : {"detected_image_url": final_url, "number_of_acnes": number_of_crops}})
