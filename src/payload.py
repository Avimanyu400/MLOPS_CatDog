import requests
import numpy as np

# Assuming your image size is 64
img_size = 64

# Create a dummy image array matching [3, 64, 64] with float values
dummy_image = np.random.rand(3, img_size, img_size).astype(np.float32).tolist()

#print(dummy_image)
# Send the request to your FastAPI server
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"image": dummy_image}
)

print(response.json())