from functools import partial
import numpy as np
import menpo.io as mio

def bbox_overlap_area(a, b):
    max_overlap = np.min([a.max(axis=0), b.max(axis=0)], axis=0)
    min_overlap = np.max([a.min(axis=0), b.min(axis=0)], axis=0)
    overlap_size = max_overlap - min_overlap
    if np.any(overlap_size < 0):
        return 0
    else:
        return overlap_size.prod()


def bbox_proportion_overlap(a, b):
    overlap = bbox_overlap_area(a, b)
    return overlap / bbox_area(a)


def bbox_area(b):
    return (b.max(axis=0) - b.min(axis=0)).prod()


def bbox_area_ratio(a, b):
    return bbox_area(a) / bbox_area(b)


def bbox_overlap_acceptable(gt, d):
    return (bbox_proportion_overlap(gt, d) > 0.5 and
            bbox_area_ratio(gt, d) > 0.5)


def load_dlib_detector():
    from menpodetect import load_dlib_frontal_face_detector
    detector = load_dlib_frontal_face_detector()
    return partial(detector, greyscale=False)

detector = load_dlib_detector()

def load_opencv_detector():
    from menpodetect import load_opencv_frontal_face_detector
    detector = load_opencv_frontal_face_detector()
    return partial(detector, greyscale=False)


def load_pico_detector():
    from menpodetect import load_pico_frontal_face_detector
    detector = load_pico_frontal_face_detector()
    return partial(detector, greyscale=False)


def detect_and_check(img, det=None, group=None):
    if det is None:
        det = detector

    gt = img.landmarks[group].lms.bounding_box()
    bad_fits = []
    for detection in detector(img):
        if bbox_overlap_acceptable(gt.points, detection.points):
            return detection

    return None

def normalize(gt):
    from menpo.transform import Translation, NonUniformScale
    t = Translation(gt.centre()).pseudoinverse()
    s = NonUniformScale(gt.range()).pseudoinverse()
    return t.compose_before(s)


def random_instance(pca):
    weights = np.random.multivariate_normal(np.zeros_like(pca.eigenvalues),
                                            np.diag(pca.eigenvalues))
    return pca.instance(weights)


_DETECTORS = {
    'dlib': load_dlib_detector,
    'pico': load_pico_detector,
    'opencv': load_opencv_detector
}

def synthesize_detection(pca_model, lms):
    """Synthesizes a bounding box for a particular detector.

    Args:
      pca_model: A menpo PCAModel instance.
      im: A menpo image.
    Returns:
      A
    """
    gt_bb = lms.bounding_box()

    instance = random_instance(pca_model)

    return normalize(gt_bb).pseudoinverse().apply(instance)

def create_generator(shapes, detections):
    import menpo.io as mio
    from menpo.landmark import LandmarkGroup
    from menpo.model import PCAModel

    # normalize these to size [1, 1], centred on origin
    normed_detections = [
      normalize(lms.bounding_box()).apply(det)
      for lms, det in zip(shapes, detections)
    ]

    # build a PCA model from good detections
    return PCAModel(normed_detections)

def load_n_create_generator(pattern, detector_name,
        group=None, overwrite=False):
    import menpo.io as mio
    from menpo.landmark import LandmarkGroup
    from menpo.model import PCAModel
    try:
        detector = _DETECTORS[detector_name]()
    except KeyError:
        detector_list = ', '.join(list(_DETECTORS.keys()))
        raise ValueError('Valid detector types are: {}'.format(detector_list))
    print('Running {} detector on {}'.format(detector_name, pattern))
    bboxes = [(img, detect_and_check(img, detector, group=group))
              for img in mio.import_images(pattern, normalise=False,
                                           verbose=True)]

    # find all the detections that did not fail
    detections = filter(lambda x: x[1] is not None, bboxes)

    print('Creating a model out of {} detections.'.format(len(detections)))
    # normalize these to size [1, 1], centred on origin
    normed_detections = [
      normalize(im.landmarks[group].lms.bounding_box()).apply(det)
      for im, det in detections
    ]

    # build a PCA model from good detections
    pca = PCAModel(normed_detections)

    mio.export_pickle(pca, '{}_gen.pkl'.format(detector_name), overwrite=overwrite)

if __name__ == '__main__':
    path = '/Users/gtrigeo/db/lfpw/trainset/*.png'
    create_generator(path, 'dlib', group='PTS')
