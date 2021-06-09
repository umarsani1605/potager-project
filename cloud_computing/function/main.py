import tensorflow 
from google.cloud import storage
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from PIL import Image
import numpy

# We keep model as global variable so we don't have to reload it in case of warm invocations
model = None

BUCKET_NAME = 'capstone_project_cap0464'

class CustomModel(Model):
  def _init_(self):
    super(CustomModel, self)._init_()
    self.conv1 = Conv2D(32, (3,3), strides=(1,1), activation="relu", input_shape=(256,256,3))
    self.maxpool1 = MaxPooling2D(pool_size=(2,2))
    self.conv2 = Conv2D(64, (3,3), activation="relu")
    self.maxpool2 = MaxPooling2D(pool_size=(2,2))
    self.conv3 = Conv2D(64, (3,3), activation="relu")
    self.maxpool3 = MaxPooling2D(pool_size=(2,2))
    self.flatten = Flatten()
    self.d1 = Dense(512, activation='relu')
    self.d2 = Dense(34, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


def download_blob(bucket_name, source_blob_name, destination_file_name):
  """Downloads a blob from the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(source_blob_name)

  blob.download_to_filename(destination_file_name)

  print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))


def handler(request):
  global model
  class_names = [
    'Bean__healthy', 'Potato___Early_blight', 'Bean__Angular_leaf_spot', 'Strawberry___Leaf_scorch', 
    'Bean__Bean_rust', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Blueberry___healthy', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,bell__healthy', 'Strawberry___healthy', 
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape__Esca(Black_Measles)', 'Potato___Late_blight', 
    'Corn_(maize)__Common_rust', 'Squash___Powdery_mildew', 'Tomato___healthy', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Tomato_mosaic_virus', 'Tomato___Bacterial_spot', 'Potato___healthy', 
    'Soybean___healthy', 'Cucumber__healthy', 'Tomato___Target_Spot', 'Pepper,bell__Bacterial_spot', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Early_blight', 'Cucumber__Ill', 'Raspberry___healthy', 'Tomato___Late_blight', 'Grape___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Tomato___Leaf_Mold'
   ]

  # Model load which only happens during cold starts
  if model is None:
    download_blob(BUCKET_NAME, 'tensorflow/variables.index',
                  '/tmp/variables.index')
    download_blob(BUCKET_NAME, 'tensorflow/variables.data-00000-of-00001',
                  '/tmp/variables.data-00000-of-00001')
    model = CustomModel()
    #model = tensorflow.keras.models.load_model('/tmp/model')
    #model.load_weights('/tmp/saved_model.pb')

  download_blob(BUCKET_NAME, 'tensorflow/test.png', '/tmp/test.png')
  image = Image.open('/tmp/test.png')
  input_np = (numpy.array(image).astype('float32') / 255)[numpy.newaxis, :, :, numpy.newaxis]
  predictions = model.call(input_np)
  print(predictions)
  print('Image is ' + class_names[numpy.argmax(predictions)])

  return class_names[numpy.argmax(predictions)]


