# Medical APIs

This repository contains the Medical APIs.The APIs are developed using OpenCV, Tensorflow and FastAPI.
To run the server,
Install the dependencies using:
```bash
pip install -r requirements.txt
```
run 

```python
uvicorn server:app --port <PORT NUMBER>
```
 
The endpoints defined so far are:

- `/predict_pcos:`
    - paramaters:
        - `image`: An image of PCOS.
    - returns:
        - `prediction`: The prediction of the image. `healthy` for no PCOS and `unhealthy` for PCOS.