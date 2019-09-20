import os

if "JOINT_FLIGHT_PATH" in os.environ:
    path = os.environ["JOINT_FLIGHT_PATH"]
else:
    path = os.path.join(os.path.dirname(__file__), "..")
    print("No environment variable JOINT_FLIGHT_PATH found. Will look in {} "
          " for the joint flight data.".format(path))
