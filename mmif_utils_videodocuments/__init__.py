from typing import List

import cv2
import mmif.serialize.annotation
import numpy as np
from PIL import Image
from mmif.serialize import Mmif, Annotation, Document
from mmif.vocabulary import AnnotationTypes, DocumentTypes


def open_file(mmif: Mmif, video_doc_id: str) -> float:
    video_doc = mmif.get_document_by_id(video_doc_id)
    if video_doc is None or video_doc.at_type != DocumentTypes.VideoDocument:
        raise ValueError(f'Video document with id "{video_doc_id}" does not exist.')
    for v in mmif.get_views_for_document(video_doc_id):
        for a in v.get_annotations(AnnotationTypes.Annotation):
            framerate_keys = ('fps', 'framerate')
            for k, v in a.properties.items():
                if k.lower() in framerate_keys:
                    return v
    cap = cv2.VideoCapture(video_doc.location_path())
    return cap.get(cv2.CAP_PROP_FPS)


def capture(vd: mmif.serialize.annotation.Document) -> cv2.VideoCapture:
    v = cv2.VideoCapture(vd.location_path())
    vd.add_property('fps', v.get(cv2.CAP_PROP_FPS))
    return v


def get_framerate(video_doc: Document) -> float:
    cap = cv2.VideoCapture(video_doc.location_path())
    return cap.get(cv2.CAP_PROP_FPS)


def frames_to_seconds(video_doc: Document, frames: int, sample_ratio: int) -> float:
    # Needs to take fps and sample ratio
    fps = get_framerate(video_doc)
    return frames / (fps * sample_ratio)


def frames_to_milliseconds(video_doc: Document, frames: int, sample_ratio: int) -> float:
    # Needs to take fps and sample ratio
    fps = get_framerate(video_doc)
    return frames / (fps * sample_ratio) * 1000


def seconds_to_frames(video_doc: Document, seconds: float, sample_ratio: int) -> int:
    fps = get_framerate(video_doc)
    return int(seconds * fps * sample_ratio)


def milliseconds_to_frames(video_doc: Document, milliseconds: float, sample_ratio: int) -> int:
    fps = get_framerate(video_doc)
    return int(milliseconds / 1000 * fps * sample_ratio)


def extract_frames(video_doc: Document, sample_ratio: int, frame_cutoff: int = None) -> List[np.ndarray]:
    video_frames = []
    video_filename = video_doc.location_path()

    # Open the video file
    video = cv2.VideoCapture(video_filename)
    current_frame = 0
    while video.isOpened():
        # Read the current frame
        ret, frame = video.read()

        if ret:
            video_frames.append(frame)
        else:
            break

        # Skip sampleRatio frames
        current_frame += sample_ratio
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        if frame_cutoff is not None and len(video_frames) > frame_cutoff - 1:
            break

    # Potentially print some statistics like how many frames extracted, sampleRatio, cutoff
    print(f'Extracted {len(video_frames)} frames from {video_filename}')
    return video_frames


def extract_pil_images(video_doc: Document, sample_ratio: int = 15, frame_cutoff: int = None) -> List[Image.Image]:
    video_frames = []
    video_filename = video_doc.location_path()

    # Open the video file
    video = cv2.VideoCapture(video_filename)
    current_frame = 0
    while video.isOpened():
        # Read the current frame
        ret, frame = video.read()

        if ret:
            # Convert it to a PIL image
            video_frames.append(Image.fromarray(frame[:, :, ::-1]))
        else:
            break

        # Skip sampleRatio frames
        if sample_ratio is not None:
            current_frame += sample_ratio
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        else:
            current_frame += 1
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        if frame_cutoff is not None and len(video_frames) > frame_cutoff - 1:
            break

    # Potentially print some statistics like how many frames extracted, sampleRatio, cutoff
    print(f'Extracted {len(video_frames)} frames from {video_filename}')
    return video_frames


patches = {Document: [extract_frames, frames_to_seconds, get_framerate],
           Mmif: [open_file]}
