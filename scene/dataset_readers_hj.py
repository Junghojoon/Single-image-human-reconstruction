import re
import os
import sys
import glob
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
# from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm
from smpl.smpl_numpy import SMPL

import cv2

try:
    import pyexr
except Exception as e:
    print(e)
    # raise e
    pyexr = None


class CameraInfo(NamedTuple):
    # uid: int
    # R: np.array
    # T: np.array
    # image: np.array
    # image_path: str
    # image_name: str
    # width: int
    # height: int
    # FovY: np.array = None
    # FovX: np.array = None
    # fx: np.array = None # f = focal length
    # fy: np.array = None
    # cx: np.array = None # c = principle point
    # cy: np.array = None
    # normal: np.array = None
    # hdr: bool = False
    # depth: np.array = None
    # image_mask: np.array = None
    
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    bkgd_mask: np.array
    bound_mask: np.array
    width: int
    height: int
    smpl_param: dict
    world_vertex: np.array
    world_bound: np.array
    big_pose_smpl_param: dict
    big_pose_world_vertex: np.array
    big_pose_world_bound: np.array
    
    fx: np.array = None # f = focal length
    fy: np.array = None
    cx: np.array = None # c = principle point
    cy: np.array = None


class SceneInfo(NamedTuple):
    # point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    # ply_path: str


def load_img(path):
    if not "." in os.path.basename(path):
        files = glob.glob(path + '.*')
        assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
        path = files[0]
    if path.endswith(".exr"):
        if pyexr is not None:
            exr_file = pyexr.open(path)
            # print(exr_file.channels)
            all_data = exr_file.get()
            img = all_data[..., 0:3]
            if "A" in exr_file.channels:
                mask = np.clip(all_data[..., 3:4], 0, 1)
                img = img * mask
        else:
            img = imageio.imread(path)
            import pdb;
            pdb.set_trace()
        img = np.nan_to_num(img)
        hdr = True
    else:  # LDR image
        img = imageio.imread(path)
        img = img / 255
        # img[..., 0:3] = srgb_to_rgb_np(img[..., 0:3])
        hdr = False
    return img, hdr


def load_pfm(file: str):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(br'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = data[::-1, ...]  # cv2.flip(data, 0)

    return np.ascontiguousarray(data)


def load_depth(tiff_path):
    return imageio.imread(tiff_path)


def load_mask(mask_file):
    mask = imageio.imread(mask_file, mode='L')
    mask = mask.astype(np.float32)
    mask[mask > 0.5] = 1.0

    return mask


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, debug=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            ppx = intr.params[1]
            ppy = intr.params[2]

            Fovx = focal2fov(focal_length_x, width)
            FovY = focal2fov(focal_length_y, height)

        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            ppx = intr.params[2]
            ppy = intr.params[3]

            Fovx = focal2fov(focal_length_x, width)
            FovY = focal2fov(focal_length_y, height)

        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image, is_hdr = load_img(os.path.join(images_folder, image_name))

        mask_path = os.path.join(os.path.dirname(images_folder), "masks", os.path.basename(extr.name))
        mask = 1 - np.array(Image.open(mask_path)) / 255
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovX=Fovx, FovY=FovY, fx=focal_length_x, fy=focal_length_y, cx=ppx,
                              cy=ppy, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, hdr=is_hdr,
                              image_mask=mask)
        cam_infos.append(cam_info)

        if debug and idx >= 5:
            break
    sys.stdout.write('\n')
    return cam_infos

###################################   ZJUMoCapRefine   ###################################################################
def get_camera_extrinsics_zju_mocap_refine(view_index, val=False, camera_view_num=36):
    def norm_np_arr(arr):
        return arr / np.linalg.norm(arr)

    def lookat(eye, at, up):
        zaxis = norm_np_arr(at - eye)
        xaxis = norm_np_arr(np.cross(zaxis, up))
        yaxis = np.cross(xaxis, zaxis)
        _viewMatrix = np.array([
            [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, eye)],
            [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, eye)],
            [-zaxis[0], -zaxis[1], -zaxis[2], np.dot(zaxis, eye)],
            [0       , 0       , 0       , 1     ]
        ])
        return _viewMatrix
    
    def fix_eye(phi, theta):
        camera_distance = 3
        return np.array([
            camera_distance * np.sin(theta) * np.cos(phi),
            camera_distance * np.sin(theta) * np.sin(phi),
            camera_distance * np.cos(theta)
        ])

    if val:
        eye = fix_eye(np.pi + 2 * np.pi * view_index / camera_view_num + 1e-6, np.pi/2 + np.pi/12 + 1e-6).astype(np.float32) + np.array([0, 0, -0.8]).astype(np.float32)
        at = np.array([0, 0, -0.8]).astype(np.float32)

        extrinsics = lookat(eye, at, np.array([0, 0, -1])).astype(np.float32)
    return extrinsics
######################################################################################################################
def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy
######################################################################################################################
def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 4]]], 1) # 4,5,7,6,4
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask
######################################################################################################################
def readCamerasZJUMoCapRefine(path, output_view, white_background, image_scaling=0.5, split='train', novel_view_vis=False):
    cam_infos = []

    # pose_start = 0
    # if split == 'train':
    #     pose_interval = 5
    #     pose_num = 100
    # elif split == 'test':
    #     pose_start = 0
    #     pose_interval = 30
    #     pose_num = 17
        
    pose_start = 0
    # if split == 'train':
    #     pose_interval = 5
    #     pose_num = 100
    # elif split == 'test':
    #     pose_start = 0
    #     pose_interval = 30
    #     pose_num = 17
        
    if split == 'train' or split == 'test':
        pose_interval = 1
        pose_num = 653


    ann_file = os.path.join(path, 'annots.npy')
    annots = np.load(ann_file, allow_pickle=True).item()
    cams = annots['cams']
    ims = np.array([
        np.array(ims_data['ims'])[output_view]
        for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval][::pose_interval]
    ])

    cam_inds = np.array([
        np.arange(len(ims_data['ims']))[output_view]
        for ims_data in annots['ims'][pose_start:pose_start + pose_num * pose_interval][::pose_interval]
    ])

    if 'CoreView_313' in path or 'CoreView_315' in path:
        for i in range(ims.shape[0]):
            ims[i] = [x.split('/')[0] + '/' + x.split('/')[1].split('_')[4] + '.jpg' for x in ims[i]]


    smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')

    # SMPL in canonical space
    big_pose_smpl_param = {}
    big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
    big_pose_smpl_param['Th'] = np.zeros((1,3)).astype(np.float32)
    big_pose_smpl_param['shapes'] = np.zeros((1,10)).astype(np.float32)
    big_pose_smpl_param['poses'] = np.zeros((1,72)).astype(np.float32)
    big_pose_smpl_param['poses'][0, 5] = 45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 8] = -45/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 23] = -30/180*np.array(np.pi)
    big_pose_smpl_param['poses'][0, 26] = 30/180*np.array(np.pi)

    big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
    big_pose_xyz = (np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(np.float32)

    # obtain the original bounds for point sampling
    big_pose_min_xyz = np.min(big_pose_xyz, axis=0)
    big_pose_max_xyz = np.max(big_pose_xyz, axis=0)
    big_pose_min_xyz -= 0.05
    big_pose_max_xyz += 0.05
    big_pose_world_bound = np.stack([big_pose_min_xyz, big_pose_max_xyz], axis=0)

    idx = 0
    for pose_index in range(pose_num):
        for view_index in range(len(output_view)):

            if novel_view_vis:
                view_index_look_at = view_index
                view_index = 0

            # Load image, mask, K, D, R, T
            # print(f"ims: {ims}")
            # print(f"pose_index: {pose_index}, view_index: {view_index}")

            image_path = os.path.join(path, ims[pose_index][view_index].replace('\\', '/'))
            image_name = ims[pose_index][view_index].split('.')[0]
            image = np.array(imageio.imread(image_path).astype(np.float32)/255.)

            msk_path = image_path.replace('images', 'mask').replace('jpg', 'png')
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

            if not novel_view_vis:
                cam_ind = cam_inds[pose_index][view_index]
                K = np.array(cams['K'][cam_ind])
                D = np.array(cams['D'][cam_ind])
                R = np.array(cams['R'][cam_ind])
                T = np.array(cams['T'][cam_ind]) / 1000.

                image = cv2.undistort(image, K, D)
                msk = cv2.undistort(msk, K, D)
            else:
                pose = np.matmul(np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]]), get_camera_extrinsics_zju_mocap_refine(view_index_look_at, val=True))
                R = pose[:3,:3]
                T = pose[:3, 3].reshape(-1, 1)
                cam_ind = cam_inds[pose_index][view_index]
                K = np.array(cams['K'][cam_ind])

            image[msk == 0] = 1 if white_background else 0

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3:4] = T

            # get the world-to-camera transform and set R, T
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Reduce the image resolution by ratio, then remove the back ground
            ratio = image_scaling
            if ratio != 1.:
                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                K[:2] = K[:2] * ratio

            image = Image.fromarray(np.array(image*255.0, dtype=np.byte), "RGB")

            focalX = K[0,0]
            focalY = K[1,1]
            FovX = focal2fov(focalX, image.size[0])
            FovY = focal2fov(focalY, image.size[1])
            
            principle_x = K[0,2]
            principle_y = K[1,2]

            # load smpl data
            i = int(os.path.basename(image_path)[:-4])
            vertices_path = os.path.join(path, 'smpl_vertices', '{}.npy'.format(i))
            # xyz = np.load(vertices_path).astype(np.float32)
            data = np.load(vertices_path, allow_pickle=True)
            
            # print(type(data))  # 데이터의 타입 확인
            # print(data)  # 실제 데이터 내용 출력

            xyz = np.load(vertices_path).astype(np.float32)
            # xyz = np.load(vertices_path, allow_pickle=True).astype(np.float32)
            
            # data = np.load(vertices_path, allow_pickle=True)

            # # 데이터가 numpy 배열이고 첫 번째 요소가 딕셔너리인지 확인
            # if isinstance(data, np.ndarray) and isinstance(data[0], dict):
            #     annots = data[0]
                
            #     # 'keypoints3d' 값을 추출하여 xyz에 저장 (필요시 다른 키 사용 가능)
            #     if 'keypoints3d' in annots:
            #         xyz = np.array(annots['keypoints3d'], dtype=np.float32)
            #     else:
            #         raise KeyError("'keypoints3d' 키가 데이터에 없습니다.")
            # else:
            #     raise ValueError("로드된 데이터가 예상한 딕셔너리 형식이 아닙니다.")


            smpl_param_path = os.path.join(path, "smpl_params", '{}.npy'.format(i))
            smpl_param = np.load(smpl_param_path, allow_pickle=True).item()
            
            Rh = smpl_param['Rh']
            smpl_param['R'] = cv2.Rodrigues(Rh)[0].astype(np.float32)
            smpl_param['Th'] = smpl_param['Th'].astype(np.float32)
            smpl_param['shapes'] = smpl_param['shapes'].astype(np.float32)
            smpl_param['poses'] = smpl_param['poses'].astype(np.float32)
            
            # print("RH = ", Rh)
            # print("R = ", smpl_param['R'])
            # print("TH = ", smpl_param['Th'])
            # print("shapes = ", smpl_param['shapes'])
            # print("poses = ", smpl_param['poses'])
            # input()

            # obtain the original bounds for point sampling
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= 0.05
            max_xyz += 0.05
            world_bound = np.stack([min_xyz, max_xyz], axis=0)

            # get bounding mask and bcakground mask
            bound_mask = get_bound_2d_mask(world_bound, K, w2c[:3], image.size[1], image.size[0])
            bound_mask = Image.fromarray(np.array(bound_mask*255.0, dtype=np.byte))

            bkgd_mask = Image.fromarray(np.array(msk*255.0, dtype=np.byte))

            cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            bound_mask=bound_mask, width=image.size[0], height=image.size[1], 
                            smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            big_pose_world_bound=big_pose_world_bound, fx=focalX, fy=focalY, cx=principle_x, cy=principle_y))

            idx += 1
            
    return cam_infos
######################################################################################################################
def readZJUMoCapRefineInfo(path, white_background, eval): #, output_path):
    train_view = [0]
    # test_view = [i for i in range(0, 23)]
    # test_view = [4]
    # test_view.remove(train_view[0])
    test_view = train_view.copy()
    print("Reading Training Transforms")
    train_cam_infos = readCamerasZJUMoCapRefine(path, train_view, white_background, split='train')
    # print("train_cam", train_cam_infos)
    
    print("Reading Test Transforms")
    test_cam_infos = readCamerasZJUMoCapRefine(path, test_view, white_background, split='test', novel_view_vis=False)
    # print("test_cam", test_cam_infos)
    # input()
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    # ply_path = os.path.join('output', output_path, "points3d.ply")
    # if not os.path.exists(ply_path):
    #     # Since this data set has no colmap data, we start with random points
    #     num_pts = 100_000 # 6890
    #     print(f"Generating random point cloud ({num_pts})...")
        
    #     # We create random points inside the bounds of the synthetic Blender scenes
    #     xyz = train_cam_infos[0].big_pose_world_vertex

    #     shs = np.random.random((num_pts, 3)) / 255.0
    #     pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #     storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(#point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization)
                           #ply_path=ply_path)
    # print(scene_info.train_cameras[0])
    # input()
    return scene_info
######################################################################################################################

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T

    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32)
        colors /= 255.0

    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    if np.all(normals == 0):
        print("random init normal")
        normals = np.random.random(normals.shape)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# def readColmapSceneInfo(path, images, eval, llffhold=8, debug=False):
#     try:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
#         cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#     except:
#         cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
#         cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
#         cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#         cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

#     reading_dir = "images" if images is None else images
#     cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
#                                            images_folder=os.path.join(path, reading_dir),
#                                            debug=debug)
#     cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
#     print(cam_infos)
#     input()

#     if "DTU" in path and not debug:
#         test_indexes = [2, 12, 17, 30, 34]
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in test_indexes]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_indexes]
#     elif eval and not debug:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, "sparse/0/points3D.ply")
#     bin_path = os.path.join(path, "sparse/0/points3D.bin")
#     txt_path = os.path.join(path, "sparse/0/points3D.txt")
#     if not os.path.exists(ply_path):
#         print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#         try:
#             xyz, rgb, _ = read_points3D_binary(bin_path)
#         except:
#             xyz, rgb, _ = read_points3D_text(txt_path)
#         storePly(ply_path, xyz, rgb)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization) # ,
#                            #ply_path=ply_path)
#     return scene_info

## -----------------------------------------------------------
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []

    read_mvs = False
    mvs_dir = f"{path}/extra"
    if os.path.exists(mvs_dir) and "train" not in transformsfile:
        print("Loading mvs as geometry constraint.")
        read_mvs = True

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image, is_hdr = load_img(image_path)

            # 이미지 크기를 (width, height) 튜플로 변환
            if len(image.shape) == 3:
                height, width, channels = image.shape
            else:
                height, width = image.shape  # 만약 흑백 이미지라면 채널이 없으므로 예외 처리

            image_size = (width, height)  # (width, height) 형태로 튜플 생성

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            image_mask = np.ones_like(image[..., 0])
            if image.shape[-1] == 4:
                image_mask = image[:, :, 3]
                image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])

            # read depth and mask
            depth = None
            normal = None
            if read_mvs:
                depth_path = os.path.join(mvs_dir + "/depths/", os.path.basename(frame["file_path"]) + ".tiff")
                normal_path = os.path.join(mvs_dir + "/normals/", os.path.basename(frame["file_path"]) + ".pfm")

                depth = load_depth(depth_path)
                normal = load_pfm(normal_path)

                depth = depth * image_mask
                normal = normal * image_mask[..., np.newaxis]

            # fov 계산 시 image_size 사용
            fovy = focal2fov(fov2focal(fovx, image_size[0]), image_size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, image_mask=image_mask,
                                        image_path=image_path, depth=depth, normal=normal, image_name=image_name,
                                        width=image_size[0], height=image_size[1], hdr=is_hdr))

            if debug and idx >= 5:
                break

    return cam_infos


## -----------------------------------------------------------
# def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", debug=False):
#     cam_infos = []

#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         fovx = contents["camera_angle_x"]
        
#         read_mvs = False
#         mvs_dir = f"{path}/extra"
#         if os.path.exists(mvs_dir) and "train" not in transformsfile:
#             print("Loading mvs as geometry constraint.")
#             read_mvs = True

#         frames = contents["frames"]
#         for idx, frame in enumerate(frames[:20]):
#             cam_name = os.path.join(path, frame["file_path"] + extension)

#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])
#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)
#             R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             image_path = os.path.join(path, cam_name)
#             image_name = Path(cam_name).stem
#             image, is_hdr = load_img(image_path)
#             # image = Image.open(image_path)
#             image = image

#             # im_data = np.array(image.convert("RGBA"))

#             # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

#             # norm_data = im_data / 255.0
#             # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
#             # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            
#             # 이미지가 정상적으로 로드되는지 확인
#             print(image.size)  # (width, height) 형태의 튜플이 출력되어야 함

#             # 정수형이 아닌지 확인
#             if isinstance(image.size, int):
#                 raise ValueError("image.size는 정수형이어서는 안 됩니다. 이미지 로드 과정에서 문제가 발생한 것 같습니다.")

#             fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#             FovY = fovy 
#             FovX = fovx
            
#             bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            
#             image_mask = np.ones_like(image[..., 0])
#             if image.shape[-1] == 4: 
#                 image_mask = image[:, :, 3]
#                 image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])
                
#             # read depth and mask
#             depth = None
#             normal = None
#             if read_mvs:
#                 depth_path = os.path.join(mvs_dir + "/depths/", os.path.basename(frame["file_path"]) + ".tiff")
#                 normal_path = os.path.join(mvs_dir + "/normals/", os.path.basename(frame["file_path"]) + ".pfm")

#                 depth = load_depth(depth_path)
#                 normal = load_pfm(normal_path)

#                 depth = depth * image_mask
#                 normal = normal * image_mask[..., np.newaxis]

#             # cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#             #                 image_path=image_path, image_name=image_name, bkgd_mask=None, bound_mask=None, width=image.size[0], height=image.size[1]))
#             cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, 
#                             image_mask=image_mask, image_path=image_path, depth=depth, normal=normal, 
#                             image_name=image_name, width=image.shape[1], height=image.shape[0], hdr=is_hdr))
            
#             if debug and idx >= 5:
#                 break

#     return cam_infos
## -----------------------------------------------------------
# def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", debug=False):
#     cam_infos = []

#     read_mvs = False
#     mvs_dir = f"{path}/extra"
#     if os.path.exists(mvs_dir) and "train" not in transformsfile:
#         print("Loading mvs as geometry constraint.")
#         read_mvs = True

#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         fovx = contents["camera_angle_x"]

#         frames = contents["frames"]
#         for idx, frame in enumerate(tqdm(frames, leave=False)):
#             image_path = os.path.join(path, frame["file_path"] + extension)
#             image_name = Path(image_path).stem

#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])
#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)
#             R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             image, is_hdr = load_img(image_path)

#             bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

#             image_mask = np.ones_like(image[..., 0])
#             if image.shape[-1] == 4:
#                 image_mask = image[:, :, 3]
#                 image = image[:, :, :3] * image[:, :, 3:4] + bg * (1 - image[:, :, 3:4])

#             # read depth and mask
#             depth = None
#             normal = None
#             if read_mvs:
#                 depth_path = os.path.join(mvs_dir + "/depths/", os.path.basename(frame["file_path"]) + ".tiff")
#                 normal_path = os.path.join(mvs_dir + "/normals/", os.path.basename(frame["file_path"]) + ".pfm")

#                 depth = load_depth(depth_path)
#                 normal = load_pfm(normal_path)

#                 depth = depth * image_mask
#                 normal = normal * image_mask[..., np.newaxis]

#             fovy = focal2fov(fov2focal(fovx, image.shape[0]), image.shape[1])
#             cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image, image_mask=image_mask,
#                                         image_path=image_path, depth=depth, normal=normal, image_name=image_name,
#                                         width=image.shape[1], height=image.shape[0], hdr=is_hdr))

#             if debug and idx >= 5:
#                 break

#     return cam_infos
## -----------------------------------------------------------
def readNerfSyntheticInfo(path, white_background, eval, extension=".png", debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, debug=debug)

    # Test 데이터도 읽어들이는 경우
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, debug=debug)
    else:
        test_cam_infos = []  # 평가하지 않으면 test 카메라는 비어있음

    # NeRF Normalization 계산
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # point cloud 경로 설정
    ply_path = os.path.join(path, "points3d.ply")

    # points3d.ply 파일이 없을 경우 생성
    if not os.path.exists(ply_path):
        num_pts = 100_000  # 생성할 무작위 포인트 수
        print(f"Generating random point cloud ({num_pts})...")

        # 랜덤 포인트 생성 (synthetic Blender scene의 범위 내)
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3  # [-1.3, 1.3] 범위의 무작위 값
        shs = np.random.random((num_pts, 3)) / 255.0  # SH 값 설정

        # 노멀 벡터 무작위로 생성 및 정규화
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        # 생성한 포인트와 노멀을 PLY 파일로 저장
        storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)
    
    # PLY 파일 로드
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # Scene 정보 반환
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info
## -----------------------------------------------------------
# def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
#     print("Reading Training Transforms")
#     train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
#     print("Reading Test Transforms")
#     test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
#     if not eval:
#         train_cam_infos.extend(test_cam_infos)
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, "points3d.ply")
#     if not os.path.exists(ply_path):
#         # Since this data set has no colmap data, we start with random points
#         num_pts = 100_000
#         print(f"Generating random point cloud ({num_pts})...")
        
#         # We create random points inside the bounds of the synthetic Blender scenes
#         xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
#         shs = np.random.random((num_pts, 3)) / 255.0
#         pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

#         storePly(ply_path, xyz, SH2RGB(shs) * 255)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info
## -----------------------------------------------------------
# def readNerfSyntheticInfo(path, white_background, eval, extension=".png", debug=False):
#     print("Reading Training Transforms")
#     train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, debug=debug)
#
#     if eval:
#         print("Reading Test Transforms")
#         test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,
#                                                    debug=debug)
#     else:
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, "points3d.ply")
#     if not os.path.exists(ply_path):
#         # Since this data set has no colmap data, we start with random points
#         num_pts = 100_000
#         print(f"Generating random point cloud ({num_pts})...")

#         # We create random points inside the bounds of the synthetic Blender scenes
#         xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
#         shs = np.random.random((num_pts, 3)) / 255.0
#         normals = np.random.randn(*xyz.shape)
#         normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

#         storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)

#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)

#     return scene_info

def loadCamsFromScene(path, valid_list, white_background, debug):
    with open(f'{path}/sfm_scene.json') as f:
        sfm_scene = json.load(f)

    # load bbox transform
    bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)
    bbox_transform = bbox_transform.copy()
    bbox_transform[[0, 1, 2], [0, 1, 2]] = bbox_transform[[0, 1, 2], [0, 1, 2]].max() / 2
    bbox_inv = np.linalg.inv(bbox_transform)

    # meta info
    image_list = sfm_scene['image_path']['file_paths']

    # camera parameters
    train_cam_infos = []
    test_cam_infos = []
    camera_info_list = sfm_scene['camera_track_map']['images']
    for i, (index, camera_info) in enumerate(camera_info_list.items()):
        # flg == 2 stands for valid camera 
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            intrinsic[2, 2] = intrinsic[3, 3] = 1

            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)
            c2w = np.linalg.inv(extrinsic)
            c2w[:3, 3] = (c2w[:4, 3] @ bbox_inv.T)[:3]
            extrinsic = np.linalg.inv(c2w)

            R = np.transpose(extrinsic[:3, :3])
            T = extrinsic[:3, 3]

            focal_length_x = camera_info['camera']['intrinsic']['focal'][0]
            focal_length_y = camera_info['camera']['intrinsic']['focal'][1]
            ppx = camera_info['camera']['intrinsic']['ppt'][0]
            ppy = camera_info['camera']['intrinsic']['ppt'][1]

            image_path = os.path.join(path, image_list[index])
            image_name = Path(image_path).stem

            image, is_hdr = load_img(image_path)

            depth_path = os.path.join(path + "/depths/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".tiff"))

            if os.path.exists(depth_path):
                depth = load_depth(depth_path)
                depth *= bbox_inv[0, 0]
            else:
                print("No depth map for test view.")
                depth = None

            normal_path = os.path.join(path + "/normals/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".pfm"))
            if os.path.exists(normal_path):
                normal = load_pfm(normal_path)
            else:
                print("No normal map for test view.")
                normal = None

            mask_path = os.path.join(path + "/pmasks/", os.path.basename(
                image_list[index]).replace(os.path.splitext(image_list[index])[-1], ".png"))
            if os.path.exists(mask_path):
                img_mask = (imageio.imread(mask_path, pilmode='L') > 0.1).astype(np.float32)
                # if pmask is available, mask the image for PSNR
                image *= img_mask[..., np.newaxis]
            else:
                img_mask = np.ones_like(image[:, :, 0])

            fovx = focal2fov(focal_length_x, image.shape[1])
            fovy = focal2fov(focal_length_y, image.shape[0])
            if int(index) in valid_list:
                image *= img_mask[..., np.newaxis]
                test_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, fx=focal_length_x,
                                                 fy=focal_length_y, cx=ppx, cy=ppy, image=image,
                                                 image_path=image_path, image_name=image_name,
                                                 depth=depth, image_mask=img_mask, normal=normal,
                                                 width=image.shape[1], height=image.shape[0], hdr=is_hdr))
            else:
                image *= img_mask[..., np.newaxis]
                if depth is not None:
                    depth *= img_mask
                if normal is not None:
                    normal *= img_mask[..., np.newaxis]

                train_cam_infos.append(CameraInfo(uid=index, R=R, T=T, FovY=fovy, FovX=fovx, fx=focal_length_x,
                                                  fy=focal_length_y, cx=ppx, cy=ppy, image=image,
                                                  image_path=image_path, image_name=image_name,
                                                  depth=depth, image_mask=img_mask, normal=normal,
                                                  width=image.shape[1], height=image.shape[0], hdr=is_hdr))
                
                            # cam_infos.append(CameraInfo(uid=idx, pose_id=pose_index, R=R, T=T, K=K, FovY=FovY, FovX=FovX, image=image,
                            # image_path=image_path, image_name=image_name, bkgd_mask=bkgd_mask, 
                            # bound_mask=bound_mask, width=image.size[0], height=image.size[1], 
                            # smpl_param=smpl_param, world_vertex=xyz, world_bound=world_bound, 
                            # big_pose_smpl_param=big_pose_smpl_param, big_pose_world_vertex=big_pose_xyz, 
                            # big_pose_world_bound=big_pose_world_bound))
        if debug and i >= 5:
            break

    return train_cam_infos, test_cam_infos, bbox_transform

def readNeILFInfo(path, white_background, eval, debug=False):
    validation_indexes = []
    if eval:
        if "data_dtu" in path:

            # val_names_path = "datasets/data_dtu_valnames/dtu_scan24/val_names.txt"
            val_names_path = path.replace("data_dtu", "data_dtu_valnames")
            val_names_path = val_names_path.replace("DTU", "dtu") + "/val_names.txt"
            if not os.path.exists(val_names_path):
                # print("File does not exist")
                validation_indexes = [2, 12, 17, 30, 34]
            else:
                with open(val_names_path, "r") as file:
                    lines = file.readlines()
                validation_indexes = []
                for line in lines:
                    number = int(line.strip().split('.')[0])
                    validation_indexes.append(number)

            # validation_indexes = [2, 12, 17, 30, 34]
        else:
            raise NotImplementedError

    print("Reading Training transforms")
    if eval:
        print("Reading Test transforms")

    train_cam_infos, test_cam_infos, bbx_trans = loadCamsFromScene(
        f'{path}/inputs', validation_indexes, white_background, debug)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = f'{path}/inputs/model/sparse_bbx_scale.ply'
    if not os.path.exists(ply_path):
        org_ply_path = f'{path}/inputs/model/sparse.ply'

        # scale sparse.ply
        pcd = fetchPly(org_ply_path)
        inv_scale_mat = np.linalg.inv(bbx_trans)  # [4, 4]
        points = pcd.points
        xyz = (np.concatenate([points, np.ones_like(points[:, :1])], axis=-1) @ inv_scale_mat.T)[:, :3]
        normals = pcd.normals
        colors = pcd.colors

        storePly(ply_path, xyz, colors * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    # "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    ###########################################
    "ZJU_MoCap_refine" : readZJUMoCapRefineInfo,
    ###########################################
    "NeILF": readNeILFInfo,
}
