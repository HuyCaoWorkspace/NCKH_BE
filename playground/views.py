from django.core.cache import cache

# Clear all cache keys
cache.clear()

from django.shortcuts import render

from django.conf import settings
import os

from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response

import shutil

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


class AcneDetectionView(APIView):
        
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

        detected_img = "run/detect/yolo_predict/" + os.path.basename(image_path)

        detection_url = upload_to_firebase(detected_img)
        print(detection_url)

        number_of_crops = count_crops_in_acne_classes("run/classify/")
        print(number_of_crops)

        # remove data from local
        # shutil.rmtree('run')

        return Response({"status" : "status.HTTP_200_OK", "data" : {"detected_image_url": detection_url, "number_of_acnes": number_of_crops}})
