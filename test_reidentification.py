import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
from deepface import DeepFace

print("-----------------------------------------")

# warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf_major_version = int(tf.__version__.split(".")[0])

# if tf_major_version == 2:
# 	import logging
# 	tf.get_logger().setLevel(logging.ERROR)

# print("Running unit tests for TF ", tf.__version__)

print("-----------------------------------------")

test_threshold = 97
num_cases = 0
succeed_cases = 0

def evaluate(condition):

	global num_cases, succeed_cases

	if condition is True:
		succeed_cases += 1
	
	num_cases += 1

# ------------------------------------------------

detectors = ['opencv', 'mtcnn', 'retinaface']
models = ['VGG-Face', 'Facenet', 'Facenet512', 'ArcFace', 'SFace']
metrics = ['cosine', 'euclidean', 'euclidean_l2']

dataset = [
	['dataset/img1.jpg', 'dataset/img2.jpg', True],
	['dataset/img5.jpg', 'dataset/img6.jpg', True],
	['dataset/img6.jpg', 'dataset/img7.jpg', True],
	['dataset/img8.jpg', 'dataset/img9.jpg', True],
	['dataset/img1.jpg', 'dataset/img11.jpg', True],
	['dataset/img2.jpg', 'dataset/img11.jpg', True],

	['dataset/img1.jpg', 'dataset/img3.jpg', False],
	['dataset/img2.jpg', 'dataset/img3.jpg', False],
	['dataset/img6.jpg', 'dataset/img8.jpg', False],
	['dataset/img6.jpg', 'dataset/img9.jpg', False],
]

print("-----------------------------------------")

def test_cases():
    print("find function")
    df = DeepFace.find(img_path = "dataset/img1.jpg", db_path = "dataset")
    print(df.head())
    evaluate(df.shape[0]>0)
    
    print("Face recognition tests")

    passed_tests = 0; test_cases = 0
    
    model = 'ArcFace'
    

    for metric in metrics:
        for instance in dataset:
            img1 = instance[0]
            img2 = instance[1]
            result = instance[2]

            resp_obj = DeepFace.verify(img1, img2
						, model_name = model
						#, model = prebuilt_model
						, distance_metric = metric)

            prediction = resp_obj["verified"]
            distance = round(resp_obj["distance"], 2)
            threshold = resp_obj["threshold"]

            evaluate( prediction == result )

            test_result_label = "failed"
            if prediction == result:
                passed_tests = passed_tests + 1
                test_result_label = "passed"

            if prediction == True:
                classified_label = "verified"
            else:
                classified_label = "unverified"

            test_cases = test_cases + 1

            print(img1.split("/")[-1], "-", img2.split("/")[-1], classified_label, "as same person based on", model,"and",metric,". Distance:",distance,", Threshold:", threshold,"(",test_result_label,")")

        print("--------------------------")

	#-----------------------------------------

    print("Passed unit tests: ",passed_tests," / ",test_cases)

    min_score = 70

    accuracy = 100 * passed_tests / test_cases
    accuracy = round(accuracy, 2)

    print("--------------------------")
    
    
test_cases()