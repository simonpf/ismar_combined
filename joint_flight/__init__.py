import os
from pathlib import Path

if "JOINT_FLIGHT_PATH" in os.environ:
    path = Path(os.environ["JOINT_FLIGHT_PATH"])
else:
    path = Path(__file__).parent.parent
    print("No environment variable JOINT_FLIGHT_PATH found. Will look in {} "
          " for the joint flight data.".format(path))
