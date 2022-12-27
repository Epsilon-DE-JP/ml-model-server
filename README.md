# ml-model-server
A code for server that serve the AI model

# Setup & Running
1. Make sure you have [Python 3](https://www.python.org) installed.
2. Download `weights` folder from [this link](https://drive.google.com/file/d/1tGlmF7Om5aUfBh-Jsu1jw_JQ-AcAuv7P/view) and extract it in the same place at the same level as `main.py`. Please see an example in the image below.
<img width="166" alt="image" src="https://user-images.githubusercontent.com/11158905/209627066-098a4521-95c9-407c-8e84-329dd8f7fa73.png">

3. Run a command to install all packages:

```bash
pip3 install -r requirements.txt
```

Some machine may require this command instead:

```bash
pip install -r requirements.txt
```

4. Start running the server by this command:

```bash
python3 main.py --port 5500
```

Some machine may require this command instead:

```bash
python main.py --port 5500
```

# API Documentation
## GET `/v1/image/<exp>/<filename>`
To get an image with bounding box. Image must existed on the server.

### Request Body
None.

### Response
An image file.

### Example:
#### Request
GET `/v1/image/exp1/df66aacc-e2bd-4775-8f44-6cc693c5baa5.jpeg`

#### Response
Image file.

## POST `/v1/object-detection`
To predict all classes belong to this image. Send it as `multipart/form-data`.

### Request Body
"file": `BINARY` image file of a target image that you want to get predicted

"id": `STRING` of an image id. Please note that it must be compatible with OS filename rules

### Response
```json
{
  "id": ID of an image sent in a request,
  "classes": A list of unique class ID,
  "class_names": A list of unique class name,
  "resultImage": A path of result image with bounding box,
  "predictions": [{
    "class": An integer of class ID,
    "name": A class name,
    "x0": An integer of a coordinate (X_begin),
    "y0": An integer of a coordinate (Y_begin),
    "x1": An integer of a coordinate (X_end),
    "y1": An integer of a coordinate (Y_end)
  }]
}
```

### Example:
#### Request
POST `/v1/object-detection`

```
enctype='multipart/form-data'

"file": <IMAGE_BINARY>
"id": "df66aacc-e2bd-4775-8f44-6cc693c5baa5"
```

#### Response
```json
{
  "id": "df66aacc-e2bd-4775-8f44-6cc693c5baa5",
  "classes": [3],
  "class_names": ["Potholes"],
  "resultImage": "exp3/df66aacc-e2bd-4775-8f44-6cc693c5baa5.jpeg",
  "predictions": [{
    "class": 3,
    "name": "Potholes",
    "x0": 123,
    "y0": 123,
    "x1": 234,
    "y1": 234
  },
  {
    "class": 3,
    "name": "Potholes",
    "x0": 234,
    "y0": 234,
    "x1": 345,
    "y1": 345
  },
  {
    "class": 3,
    "name": "Potholes",
    "x0": 345,
    "y0": 345,
    "x1": 456,
    "y1": 456
  }]
}
```
