import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-5-early.yaml')
    
    # Train the model
    model.train(data=R'ultralytics/cfg/datasets/base.yaml',
                cache=False,
                imgsz=256,
                epochs=1,
                batch=16,
                close_mosaic=10,
                workers=2,
                device='0',
                project='rgbd_pap',
                name='rgbd_test',
                )
    
    # Export the model to ONNX format after training
    model.export(format='edgetpu', data='ultralytics/cfg/datasets/cali.yaml', fraction=0.2)  # Set dynamic=True to allow dynamic shapes for ONNX
