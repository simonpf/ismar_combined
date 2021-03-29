import ciprdr
import datetime

t0 = datetime.datetime(2016, 10, 14, 10, 30)
t1 = datetime.datetime(2016, 10, 14, 11, 15)
images = ciprdr.ImageFolder("/home/simonpf/src/joint_flight/data/20161014/20161014072702")
images.minimum_size = 5
images.extract_single_images(t0,
                             t1,
                             output_path = "/home/simonpf/src/joint_flight/data/shape_images_test.nc",
                             regexp = ".*1CIP.*",
                             output_size = (64, 64),
                             remove_out_of_focus = True)
#images = ciprdr.ImageFolder("/home/simonpf/src/joint_flight/data/20161014/20161014072702")
#images.extract_combined_images(t0, t1, output_path = "/home/simonpf/src/joint_flight/data/particle_images_100", regexp = ".*1CIP.*")
