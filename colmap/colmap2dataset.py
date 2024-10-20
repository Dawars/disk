from pathlib import Path

import h5py, argparse, os, json
import numpy as np
from tqdm import tqdm

from colmap.read_model import read_model
from colmap.read_dense import read_array


def convert_depth(name, src_path, dst_path):
    '''
    Converts depth from the COLMAP format of .bin files to h5py HDF files
    '''
    fname, ext = os.path.splitext(name)

    depth_src_name = f'{name}.geometric.bin'
    depth = read_array(os.path.join(src_path, depth_src_name))

    depth_dst_name = f'{fname}.h5'

    with h5py.File(os.path.join(dst_path, depth_dst_name), 'w') as dst_file:
        dst_file.create_dataset('depth', data=depth.astype(np.float16))

def camera_to_K(camera):
    '''
    Assembles the camera params (given as an unstructured list by COLMAP) into
    an intrinsics matrix
    '''
    assert camera.model == 'PINHOLE'

    fx, fy, cx, cy = camera.params

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float32)

def create_calibration(image, camera, prefix):
    '''
    Saves camera intrinsics and extrinsics to a HDF file
    '''
    #.replace("/0/pictures", "")  # todo replace when combining dataset
    image_path = Path(image.name)
    subdirs = image_path.parent
    (Path(prefix) / subdirs).mkdir(exist_ok=True, parents=True)
    path = os.path.join(prefix, subdirs, f'{image_path.name}.h5')

    with h5py.File(path, 'w') as dst_file:
        dst_file.create_dataset('R', data=image.qvec2rotmat())
        dst_file.create_dataset('T', data=image.tvec)
        dst_file.create_dataset('K', data=camera_to_K(camera))

def covisible_pairs(images, low=0.5, high=0.8):
    '''
    Considers all pairs of images and for each, it computes the ratio between
    3d landmarks co-visible in both images and the minimum of the number
    of 3d landmarks in either of the two images. This serves as a proxy for
    image covisiblity and pairs with the ratio between `low` and `high` are
    returned.
    '''
    images = list(images.values())

    idxs = []
    for image in tqdm(images):
        image_idxs = image.point3D_ids
        idxs.append(frozenset(image_idxs[image_idxs != -1].tolist()))

    pairs = []

    for i in range(len(images)):
        idxs_i = idxs[i]
        for j in range(i+1, len(images)):
            idxs_j = idxs[j]

            inter = len(idxs_i & idxs_j)
            ratio = inter / min(len(idxs_i), len(idxs_j))

            if low <= ratio <= high:
                pairs.append((images[i].name, images[j].name)) 

    return pairs

def encode_pairs(pairs):
    '''
    "Encodes" the list of pairs, coming as a list of file names such as
    [
        ['file_name_1.jpg', 'file_name_5.jpg'],
        ['file_name_1.jpg', 'file_name_13.jpg'],
        ...
    ]
    by assigning each filename a unique numeric ID and returning
    1) a list `id2name` such that id2name[i] is the filename of the i-th image
    2) a list of lists of the same structure as the input argument, except
       using file IDs instead of full (string) names
    '''
    curr_id = 0
    name2id = {}
    id2name = []

    for pair in pairs:
        for name in pair:
            if name in name2id:
                continue
            else:
                name2id[name] = curr_id
                id2name.append(name)
                curr_id += 1

    pairs_as_ixs = []
    for pair in pairs:
        pair = [name2id[n] for n in pair]
        pairs_as_ixs.append(pair)

    return id2name, pairs_as_ixs

def process_scene(scene_path, data_root, save_root, no_depth=True):
    scene_root = data_root / scene_path
    # image_root = data_root / "images"
    scene_subdir = scene_root.parent.parent.relative_to(data_root)
    # print(data_root)
    print(str(scene_subdir).replace("/", "_"))
    scene_name = str(scene_subdir).replace("/", "_")
    image_path = os.path.join(scene_root, "undistort", 'images')
    sparse_path = os.path.join(scene_root, "undistort", 'sparse')
    calib_path = save_root / scene_subdir / 'calibration'
    json_path = save_root / scene_subdir / 'dataset.json'

    if not no_depth:
        depth_src_path = os.path.join(sparse_path, "..", 'stereo', 'depth_maps')
        depth_dst_path = os.path.join(save_root, scene_subdir, 'depth')
    else:
        depth_src_path = ''
        depth_dst_path = ''

    cameras, images, points3D = read_model(sparse_path, ext='.bin')

    os.makedirs(calib_path, exist_ok=True)

    print('Creating calibration files...')
    for image in tqdm(images.values()):
        create_calibration(image, cameras[image.camera_id], calib_path)

    if not no_depth:
        os.makedirs(depth_dst_path, exist_ok=True)
        print('Converting depth...')
        for image in tqdm(images.values()):
            convert_depth(image.name, depth_src_path, depth_dst_path)
    else:
        print('Skipping depth maps...')

    images, tuples = encode_pairs(covisible_pairs(images))

    dataset = {
        scene_name: {
            'images': images,
            'tuples': tuples,
            'calib_path': str(calib_path),
            'depth_path': str(depth_dst_path),
            'image_path': str(image_path),
        },
    }

    with open(json_path, 'w') as json_file:
        json.dump(dataset, json_file)


if __name__ == '__main__':
    save_root = Path("/vast/ro38seb/datasets/MegaScenes/disk")
    scene_list = Path("scene_list.txt").read_text().split()
    for recon_path in tqdm(scene_list):
        process_scene(Path(recon_path),
                      Path("/vast/ro38seb/datasets/MegaScenes/"),
                      save_root)
