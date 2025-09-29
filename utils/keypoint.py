import cv2 as cv
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from argparse import ArgumentParser
import pickle

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def fill_missing(keypoints, frame_idx, shape):
    if len(keypoints) < frame_idx:
        keypoints.append(np.zeros(shape))


def process_pose_landmarks(image):
    pose_landmarks = pose.process(image).pose_landmarks
    if pose_landmarks:
        keypoints = np.array(
            [(lm.x, lm.y, lm.z, lm.visibility) for lm in pose_landmarks.landmark]
        )
    else:
        keypoints = None
    return keypoints


def process_hand_landmarks(image):
    results_hands = hands.process(image)
    keypoints_left = None
    keypoints_right = None
    if results_hands.multi_hand_landmarks:
        for hand_landmarks, hand_type in zip(
            results_hands.multi_hand_landmarks, results_hands.multi_handedness
        ):
            keypoints = np.array(
                [(lm.x, lm.y, lm.z, lm.visibility) for lm in hand_landmarks.landmark]
            )
            if hand_type.classification[0].label == "Left":
                keypoints_left = keypoints
            else:
                keypoints_right = keypoints
    return keypoints_left, keypoints_right


def process_face_landmarks(image):
    face_landmarks = face_mesh.process(image).multi_face_landmarks
    if face_landmarks:
        keypoints = np.array(
            [(lm.x, lm.y, lm.z, lm.visibility) for lm in face_landmarks[0].landmark]
        )
    else:
        keypoints = None
    return keypoints


def extract_from_video(path):
    keypoint_lefthand = []
    keypoint_righthand = []
    keypoint_pose = []
    frame_idx = 0
    print()
    print(f"Extracting keypoints for {path}")
    cap = cv.VideoCapture(path)
    loop = tqdm(
        range(0, int(cap.get(cv.CAP_PROP_FRAME_COUNT))), desc="Extracting keypoints: "
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        pose_keypoints = process_pose_landmarks(image)
        keypoints_left, keypoints_right = process_hand_landmarks(image)

        image.flags.writeable = True

        if pose_keypoints is not None:
            assert pose_keypoints.shape[0] == 33, (
                f"Pose keypoints shape is not 33: {pose_keypoints.shape}"
            )
            keypoint_pose.append(pose_keypoints)
        if keypoints_left is not None:
            assert keypoints_left.shape[0] == 21, (
                f"Left hand keypoints shape is not 21: {keypoints_left.shape}"
            )
            keypoint_lefthand.append(keypoints_left)

        if keypoints_right is not None:
            assert keypoints_right.shape[0] == 21, (
                f"Right hand keypoints shape is not 21: {keypoints_right.shape}"
            )
            keypoint_righthand.append(keypoints_right)

        fill_missing(keypoint_lefthand, frame_idx, (21, 4))
        fill_missing(keypoint_righthand, frame_idx, (21, 4))
        fill_missing(keypoint_pose, frame_idx, (33, 4))

        loop.update()

    if frame_idx == 0:
        return

    keypoint_lefthand = np.array(keypoint_lefthand)
    keypoint_righthand = np.array(keypoint_righthand)
    keypoint_pose = np.array(keypoint_pose)

    min_len = min(
        keypoint_lefthand.shape[0], keypoint_righthand.shape[0], keypoint_pose.shape[0]
    )
    keypoint_lefthand = keypoint_lefthand[:min_len]
    keypoint_righthand = keypoint_righthand[:min_len]
    keypoint_pose = keypoint_pose[:min_len]

    keypoints = np.concatenate(
        [keypoint_pose, keypoint_lefthand, keypoint_righthand], axis=1
    )

    return keypoints


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    keypoints = extract_from_video(args.input_path)
    with open(args.output_path, "wb") as f:
        pickle.dump(keypoints, f)
    print(f"Saved keypoints to {args.output_path}")
