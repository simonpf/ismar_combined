import ciprdr
import datetime

t0 = datetime.datetime(2016, 10, 14, 10, 30)
t1 = datetime.datetime(2016, 10, 14, 11, 15)
images = ciprdr.ImageFolder("/home/simonpf/src/joint_flight/data/20161014/20161014072702")
images.extract_images(t0, t1, output_path = "/home/simonpf/src/joint_flight/data/particle_images_15", regexp = ".*4CIP.*")
images = ciprdr.ImageFolder("/home/simonpf/src/joint_flight/data/20161014/20161014072702")
images.extract_images(t0, t1, output_path = "/home/simonpf/src/joint_flight/data/particle_images_100", regexp = ".*1CIP.*")
