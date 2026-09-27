"""Explicit unsupported entry point for future impedance-control integration."""


def main(args=None):
    raise RuntimeError(
        "Impedance control is not commissioned. Select controller:=position. "
        "Torque-interface support, dynamics and gravity compensation, frame-consistent "
        "feedback, loop timing, and contact stability must be validated first. "
        "A position-only driver needs a separately designed admittance controller."
    )
