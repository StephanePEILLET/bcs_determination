from ultralytics import YOLO

def train_pose_model(data_path="dog-pose.yaml", model_path="checkpoints/pose/yolo26n-pose.pt", epochs=20, imgsz=640):
    # Load a model
    model = YOLO(model_path)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=data_path, epochs=epochs, imgsz=imgsz)
    return results

if __name__ == "__main__":
    train_pose_model()