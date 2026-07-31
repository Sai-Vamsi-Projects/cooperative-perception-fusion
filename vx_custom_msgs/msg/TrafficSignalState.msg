# TrafficSignalState.msg
# Unified MAPEM + SPATEM message for traffic light decision logic.

# Intersection name (string type, used to match intersections between MAPEM and SPATEM)
uint8 intersection_id           # Intersection ID 

# Lane number (uint8 type, used to represent lane number within the intersection)
uint8 lane_id                      # Lane number

# Signal group ID (uint8 type, representing the signal group controlling the lane)
uint8 signal_group_id              # Signal group controlling this lane

# Lane centerline coordinates (geometry_msgs/Point[] type, representing the centerline of the lane)
geometry_msgs/Point[] lane_points  # Array of Points representing lane centerline coordinates

# Traffic light event state (uint8 type, representing the current traffic light state)
# ETSI eventState (2–4,7–9=RED; 5–6=GREEN; 0–1=UNKNOWN)
uint8 event_state                  # ETSI eventState (2–4,7–9=RED; 5–6=GREEN; 0–1=UNKNOWN)

# Timestamp when SPATEM message was received
builtin_interfaces/Time last_update  # Timestamp when SPATEM was received
