"""COCO-17 keypoint indices (in YOLOv8-pose output order)."""
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

# YOLO-pose always emits a length-17 array (occluded joints get low conf),
# so NUM_COCO is 17. Conductor shots are upper-body, so only upper-body
# keypoints (0-12) are named here; lower-body indices (13-16) are unused.
NUM_COCO = 17

# RTMPose WholeBody(133) layout: body 0-16, feet 17-22, face 23-90,
# left hand 91-111, right hand 112-132.
# Canonical joints that are also extractable via MediaPipe (Pose+Hands):
# shoulders/elbows/wrists + both hands.
CANONICAL_WHOLEBODY = [
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
] + list(range(91, 133))  # 9 upper-body (1 head + 6 arm + 2 hip) + 42 hand = 51 keypoints
