from roboflow import Roboflow

rf = Roboflow(api_key="Um9MRFtCAcphTltvvqIz")
project = rf.workspace("usmanchaudhry622-gmail-com").project("traffic-and-road-signs")
version = project.version(1)
dataset = version.download("yolov8")