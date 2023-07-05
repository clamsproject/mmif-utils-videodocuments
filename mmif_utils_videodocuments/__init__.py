import cv2
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
    pass


patches = {Mmif: [get_framerate]}

