# KEYPOINT: It lets you inspect installed Python packages directly from your code.

# You can ask: “Which version of this library is installed?” or “What metadata does this package have?”


import importlib.metadata as metadata

# Check package versions
numpy_version = metadata.version("numpy")
requests_version = metadata.version("requests")

print("Numpy version:", numpy_version)
print("Requests version:", requests_version)

