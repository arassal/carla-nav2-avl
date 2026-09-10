"""Pure command-source selection for the actuator control loop."""


def select_command_source(
        estop: bool, nav_active: bool,
        actuator_fresh: bool, cmd_vel_fresh: bool) -> str:
    """Return ``stop``, ``actuator``, or ``cmd_vel`` for this control tick."""
    if estop:
        return 'stop'
    if nav_active:
        return 'cmd_vel' if cmd_vel_fresh else 'stop'
    if actuator_fresh:
        return 'actuator'
    if cmd_vel_fresh:
        return 'cmd_vel'
    return 'stop'
