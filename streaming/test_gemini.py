from gemini import GeminiModel
from PIL import Image
import io
import numpy as np

image = Image.open('/home/u-ril/project-tracey/streaming/baseofcoffee.png')
#imgByteArr = io.BytesIO()
#image.save(imgByteArr, format=image.format)
#imgByteArr = imgByteArr.getvalue()
image = np.asanyarray(image)
gemini_model = GeminiModel()
gemini_model.inference3D(image)