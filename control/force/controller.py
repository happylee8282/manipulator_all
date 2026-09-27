"""Explicit unsupported entry point for future force-control integration."""


def main(args=None):
    raise RuntimeError(
        "Force control is not commissioned. Select controller:=position. "
        "A calibrated force sensor, frame transforms, robot command interface, "
        "timed control loop, and contact stop validation are required first."
    )
