"""
Take white bckgrnd solo logo and generate a mask; both will be simultaneously transformed in augmentation step
"""
import cv2
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import json
import os

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


# TODO predict_stages.py uses cv2.imread then adjusted x = (np.float32(x) - 127.5) / 127.5
def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def _preprocess_numpy_input(x, data_format, mode, **kwargs):
    """Preprocesses a Numpy array encoding a batch of images.

    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed Numpy array.
    """
    backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if x.ndim == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x


def get_mask(input_img):
    """
    :param input_img: BGR image
    :return: green mask on black bg, np.int32 to satisfy requirement for
    imgaug segmentation_maps https://github.com/aleju/imgaug
    """
    # TODO see if you can find a way to verify BGR
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    w, h = img.shape[0], img.shape[1]
    maskout = np.zeros((w, h, 3), dtype=np.int32)
    # TODO give it color options to support multiple labels for now just bw

    for col in range(w):
        for row in range(h):
            if img[col][row] < 255:  # TODO revisit thresh for white in original png logo or will be transparent/actual border so great news
                maskout[col][row][1] = 255

    return maskout


def augmentation_seq():
    """
    Note imgaug expects RGB not BGR
    :return:
    """
    # TODO this is a reference of transforms that were helpful in keras so use them in imgaug
    # rescale=None,
    # shear_range=0.2,
    # rotation_range=0.2,
    # width_shift_range=0.3,
    # height_shift_range=0.3,
    # zoom_range=[.85, 1.2],
    # fill_mode='nearest',
    # horizontal_flip=False,
    # vertical_flip=False,

    seq_simple = iaa.Sequential([
        iaa.Affine(rotate = (-45, 45))
        # iaa.Crop(px=(1, 16), keep_size=False),
        # iaa.Fliplr(0.5),
        # iaa.GaussianBlur(sigma=(0, 3.0))
    ])

    _sometimes = lambda aug: iaa.Sometimes(0.85, aug)
    seq_complex = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            # _sometimes(iaa.CropAndPad(
            #     percent=(-0.05, 0.1),
            #     pad_mode=ia.ALL,
            #     pad_cval=(0, 255)
            # )),

            _sometimes(
                iaa.Resize((0.5, 3))
            ),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((3, 5),
                       [
                           # _sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(2, 7)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 11)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                           # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # # search either for all edges or for directed edges,
                           # # blend the result with the original image using a blobby mask
                           # iaa.SimplexNoiseAlpha(iaa.OneOf([
                           #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           # ])),
                           # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           # iaa.OneOf([
                           #     iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           # ]),
                           # iaa.Invert(0.05, per_channel=True),  # invert color channels
                           # iaa.Add((-10, 10), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-5, 5)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           # iaa.OneOf([
                           #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           #     iaa.FrequencyNoiseAlpha(
                           #         exponent=(-1, 0),
                           #         first=iaa.Multiply((0.85, 1.15), per_channel=True),
                           #         second=iaa.LinearContrast((0.8, 1.2))
                           #     )
                           # ]),
                           # iaa.LinearContrast((0.75, 1.25), per_channel=0.2),  # improve or worsen the contrast
                           # iaa.Grayscale(alpha=(0.0, 1.0)),
                           # _sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # # move pixels locally around (with random strengths)
                           # _sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # # sometimes move parts of the image around
                           # _sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       ),
            _sometimes(
                iaa.Affine(
                    # scale={"x": (0.5, 2.5), "y": (0.5, 2.5)},
                    # scale images to 80-120% of their size, individually per axis
                    # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                    # shear=(-16, 16),  # shear by -16 to +16 degrees
                    # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    # TODO add this back cval=(0, 100),  # if mode is constant, use a cval between 0 and 255
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                ),
            ),
        ],
        random_order=False
    )
    return seq_complex


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        if len(segmentation) > 0:
            segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,  # both are coming from the k enumerate
        'category_id': category_id,
        'id': image_id,  # both are coming from the k enumerate TODO this fixed an issue for what cocovis_custom.py is expecting and that's native coco tools so the original poster was wrong (was using custom display function now makes sense why author just didn't kow about coco tools)
        'bbox': bbox,
        'area': area
    }

    return annotation


def file_meta(fname, idx, height=0, width=0):
    """
    Generate metadata for coco annotations images section
    :return:
    """
    return {
      "license": 1,
      "file_name": fname,
      "coco_url": "",
      "height": height,
      "width": width,
      "date_captured": "2020-01-01 00:00:01",
      "flickr_url": "",
      "id": idx
    }


def main():
    # https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

    logo_id, unused_id = [1, 2]

    color = '(0, 255, 0)'
    category_ids = {
        1: {
            color: logo_id,
        },
    }

    img = cv2.imread('./solo.png')  # cv2.IMREAD_GRAYSCALE

    imask = get_mask(img)
    imgcolor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    seq = augmentation_seq()

    images = [imgcolor]
    segmentation_maps = [imask]

    images_aug, seg_aug = [], []
    for _ in range(100):
        newimg, newmask = seq(images=images, segmentation_maps=segmentation_maps)
        images_aug.extend(newimg)
        seg_aug.extend(newmask)

    annotation_id = 1  # the selection from category_ids in this case 0, 255, 0: logo_id
    is_crowd = 0

    with open('./solo_template.json', 'r') as f:
        ann_template = json.load(f)

    # TODO MAJOR SOMETIMES HAS A -1 DUE TO SOME AUGMETATION STRIP OUT SOME OF THE IRRELEVANT AUGMENTATIONS
    # TODO MAJOR SOMETIMES THERE'S BLACK IN THE AUGMETNATION SO THE MASK ROUTINE ADDS THOSE SPOTS TO THE MASK
    annotations = []
    imginfo = []

    for k, i in enumerate(images_aug):
        # TODO MAJOR whiten borders image
        fname = 'logo{}.jpg'.format(k)
        cv2.imwrite(os.path.join('./traindata/', fname), cv2.cvtColor(i, cv2.COLOR_RGB2BGR))
        imginfo.append(file_meta(fname, k, height=i.shape[1], width=i.shape[0]))  # TODO test this should be cols(w) rows(h) for now ok because it's square

        # TODO was deriving cat id from the mask files category_id = category_ids[image_id][color]
        category_id = 1  # see json
        mask_img = seg_aug[k][:, :, 1]
        image_id = k  # TODO VERIFY will populate id and image_id make sure this is appropriate as it diverges from the example, which was using category_id or something
        annotation = create_sub_mask_annotation(mask_img, image_id, category_id, annotation_id, is_crowd)
        annotations.append(annotation)
        cv2.imwrite('./traindata/masks/mask{}.jpg'.format(k), seg_aug[k])

    ann_template['annotations'] = annotations
    ann_template['images'] = imginfo
    with open('./solo.json', 'w') as f:
        f.write(json.dumps(ann_template))


if __name__ == '__main__':
    main()
