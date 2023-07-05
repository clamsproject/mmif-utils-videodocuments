import cv2
from PIL import Image  # If we're trying to keep the imports down in utils then forget about the PIL option
from mmif.serialize import Mmif, Annotation, Document
from mmif.vocabulary import AnnotationTypes, DocumentTypes


def open_file(video_doc: Document):
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


def convert_to_seconds(annotation: Annotation):
    # Needs to take fps and sample ratio
    pass


def extract_frames(video_doc: Document, sampleRatio: int, pil: bool = False, frame_cutoff: int = None):
    video_frames = []
    video_filename = video_doc.location_path()

    # Open the video file
    video = cv2.VideoCapture(video_filename)
    current_frame = 0
    while video.isOpened():
        # Read the current frame
        ret, frame = video.read()

        if ret:
            # Convert it to a PIL image if required
            if pil:
                video_frames.append(Image.fromarray(frame[:, :, ::-1]))
            else:
                video_frames.append(frame)
        else:
            break

        # Skip sampleRatio frames
        current_frame += sampleRatio
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        if frame_cutoff is not None and len(video_frames) > frame_cutoff:
            break

    # Potentially print some statistics like how many frames extracted, sampleRatio, cutoff
    return video_frames


def extract_pil_images():
    # Potentially make this its own function instead of an option for extract frames
    pass


patches = {Mmif: [get_framerate]}
