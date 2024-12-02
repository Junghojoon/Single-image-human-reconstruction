
import os
import cv2
import numpy as np
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov
# from smpl.smpl_numpy import SMPL
# import imageio.v2 as imageio

class CameraInfo(NamedTuple):
    uid: int
    pose_id: int
    R: np.array
    T: np.array
    K: np.array
    fx: float
    fy: float
    cx: float  # 수정된 부분
    cy: float  # 수정된 부분
    FovY: float
    FovX: float
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

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict

# # ---------------- original ------------------
def getNerfppNorm(cam_info):
    if not cam_info:
        raise ValueError("No camera information provided for normalization")

    def get_center_and_diag(cam_centers):
        if not cam_centers:
            raise ValueError("No camera centers available")
        
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

    if not cam_centers:
        raise ValueError("Failed to extract camera centers")

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
# # ------------------------------------------------------

# def getNerfppNorm(cam_info):
#     cam_centers = [cam.T.reshape(3, 1) for cam in cam_info]
#     cam_centers = np.hstack(cam_centers)
#     center = np.mean(cam_centers, axis=1, keepdims=True)
#     dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
#     diagonal = np.max(dist)
#     radius = diagonal * 1.1
#     translate = -center.flatten()
#     return {"translate": translate, "radius": radius}

# def getNerfppNorm(cam_info):
#     cam_centers = [cam.T.reshape(3, 1) for cam in cam_info]
#     cam_centers = np.hstack(cam_centers)
#     center = np.mean(cam_centers, axis=1, keepdims=True)
#     dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
#     diagonal = np.max(dist)
#     radius = diagonal * 1.1
#     translate = -center.flatten()
#     return {"translate": translate, "radius": radius}

# def getNerfppNorm(cam_info):
#     cam_centers = []
#     for cam in cam_info:
#         R = cam.R
#         T = cam.T
#         # 카메라 센터 C는 R * C + T = 0을 만족
#         C = -np.dot(R.T, T)
#         cam_centers.append(C.reshape(3, 1))
#     cam_centers = np.hstack(cam_centers)
#     center = np.mean(cam_centers, axis=1, keepdims=True)
#     dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
#     diagonal = np.max(dist)
#     radius = diagonal * 1.1
#     translate = -center.flatten()
#     return {"translate": translate, "radius": radius}


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

# # --------------------------------------
# def readCamerasZJUMoCapRefine(path, output_views, white_background, image_scaling=0.5, split='train', novel_view_vis=False):
#     cam_infos = []
#     ann_file = os.path.join(path, 'annots.npy')
#     annots = np.load(ann_file, allow_pickle=True).item()
#     cams = annots['cams']
#     # pose_num = len(annots['ims'])

#     ims = [
#         [im for im in ims_data['ims'] if '000000.jpg' in im and any(folder in im for folder in output_views)]
#         for ims_data in annots['ims']
#     ]
#     print('ims', ims)
#     cam_inds = [
#         [view for view in range(len(ims_data['ims'])) if '000000.jpg' in ims_data['ims'][view] and any(folder in ims_data['ims'][view] for folder in output_views)]
#         for ims_data in annots['ims']
#     ]
#     print('cam_inds', cam_inds)

#     for pose_index, pose_images in enumerate(ims):
#         for view_index, image_path in enumerate(pose_images):
#             # folder_name = image_path.split('/')[1]
#             cam_ind = cam_inds[pose_index][view_index] if not novel_view_vis else 0
#             image_full_path = os.path.join(path, image_path)

#             print(f"Pose Index: {pose_index}, View Index: {view_index}")
#             print(f"Image Path: {image_full_path}")
#             print(f"Camera Index: {cam_ind}")
#             print(f"K (Intrinsic):\n{cams['K'][cam_ind]}")
#             print(f"D (Distortion): {cams['D'][cam_ind]}")
#             print(f"R (Rotation):\n{cams['R'][cam_ind]}")
#             print(f"T (Translation):\n{cams['T'][cam_ind]}")
#             print("-" * 50)

#             image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             K = np.array(cams['K'][cam_ind])
#             D = np.array(cams['D'][cam_ind])
#             R = np.array(cams['R'][cam_ind])
#             T = np.array(cams['T'][cam_ind]).flatten() / 1000.0

#             w2c = np.eye(4)
#             w2c[:3, :3] = R
#             w2c[:3, 3] = T

#             R = np.transpose(w2c[:3, :3])
#             T = w2c[:3, 3]

#             image = cv2.undistort(image, K, D)
#             H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
#             image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
#             K[:2] *= image_scaling

#             image = Image.fromarray(image)

#             cam_infos.append(CameraInfo(
#                 uid=pose_index, pose_id=pose_index, R=R, T=T, K=K, fx=K[0, 0], fy=K[1, 1],
#                 cx=K[0, 2], cy=K[1, 2], FovY=focal2fov(K[1, 1], H), FovX=focal2fov(K[0, 0], W),
#                 image=image, image_path=image_full_path, image_name=os.path.basename(image_full_path),
#                 bkgd_mask=None, bound_mask=None, width=W, height=H, smpl_param={}, 
#                 world_vertex=None, world_bound=None, big_pose_smpl_param={}, 
#                 big_pose_world_vertex=None, big_pose_world_bound=None
#             ))

#     return cam_infos

# # ---------------------------------------

# # ---------------------------------------
# def readCamerasZJUMoCapRefine(path, output_views, white_background, image_scaling=0.5, split='train', novel_view_vis=False):
# # def readCamerasZJUMoCapRefine(path, output_views, image_scaling=0.5, novel_view_vis=False):
#     cam_infos = []
#     ann_file = os.path.join(path, 'annots.npy')
#     annots = np.load(ann_file, allow_pickle=True).item()
#     cams = annots['cams']
#     # pose_num = len(annots['ims'])

#     # 모든 ims 데이터를 가져오도록 수정 (100개의 뷰를 포함)
#     ims = [
#         [im for im in ims_data['ims'] if any(folder in im for folder in output_views)]
#         for ims_data in annots['ims']
#     ]

#     print('ims', ims)
#     # input()
    
#     cam_inds = [
#         [view for view in range(len(ims_data['ims'])) if any(folder in ims_data['ims'][view] for folder in output_views)]
#         for ims_data in annots['ims']
#     ]
#     print('cam_inds', cam_inds)
#     # input()

#     # 각 포즈와 뷰에 대해 카메라 정보를 로드
#     for pose_index, pose_images in enumerate(ims):
#         for view_index, image_path in enumerate(pose_images):
#             # folder_name = image_path.split('/')[1]
#             cam_ind = cam_inds[pose_index][view_index] if not novel_view_vis else 0
#             image_full_path = os.path.join(path, image_path)

#             print(f"Pose Index: {pose_index}, View Index: {view_index}")
#             print(f"Image Path: {image_full_path}")
#             print(f"Camera Index: {cam_ind}")
#             print(f"K (Intrinsic):\n{cams['K'][cam_ind]}")
#             print(f"D (Distortion): {cams['D'][cam_ind]}")
#             print(f"R (Rotation):\n{cams['R'][cam_ind]}")
#             print(f"T (Translation):\n{cams['T'][cam_ind]}")
#             print("-" * 50)

#             image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             K = np.array(cams['K'][cam_ind])
#             D = np.array(cams['D'][cam_ind])
#             R = np.array(cams['R'][cam_ind])
#             T = np.array(cams['T'][cam_ind]).flatten() / 1000.0

#             w2c = np.eye(4)
#             w2c[:3, :3] = R
#             w2c[:3, 3] = T

#             R = np.transpose(w2c[:3, :3])
#             T = w2c[:3, 3]

#             image = cv2.undistort(image, K, D)
#             H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
#             image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
#             K[:2] *= image_scaling

#             image = Image.fromarray(image)

#             cam_infos.append(CameraInfo(
#                 uid=pose_index, pose_id=pose_index, R=R, T=T, K=K, fx=K[0, 0], fy=K[1, 1],
#                 cx=K[0, 2], cy=K[1, 2], FovY=focal2fov(K[1, 1], H), FovX=focal2fov(K[0, 0], W),
#                 image=image, image_path=image_full_path, image_name=os.path.basename(image_full_path),
#                 bkgd_mask=None, bound_mask=None, width=W, height=H, smpl_param={}, 
#                 world_vertex=None, world_bound=None, big_pose_smpl_param={}, 
#                 big_pose_world_vertex=None, big_pose_world_bound=None
#             ))
            
#     print(cam_infos)
#     # input()

    # return cam_infos

# # ------------------------------------------------------------

def readCamerasZJUMoCapRefine(path, output_views, white_background, image_scaling=0.5, split='train', novel_view_vis=False):
    cam_infos = []
    ann_file = os.path.join(path, 'annots.npy')
    annots = np.load(ann_file, allow_pickle=True).item()
    cams = annots['cams']
    
    if not annots['ims'] or not annots['ims'][0]['ims']:
        raise ValueError("No image paths found in annotations")
    
    # 이미지 경로 가져오기
    image_paths = annots['ims'][0]['ims']
    
    print(f"Processing {len(image_paths)} images")
    print(f"Number of cameras: K={len(cams['K'])}, R={len(cams['R'])}, T={len(cams['T'])}")

    # 각 이미지와 카메라 정보에 대해 처리
    for i in range(len(image_paths)):
        image_path = os.path.join(path, image_paths[i])
        
        # 파일 존재 확인
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found - {image_path}")
            continue

        try:
            # 이미지 로드 및 처리
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read image - {image_path}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 카메라 파라미터 가져오기
            K = np.array(cams['K'][i], dtype=np.float32)
            D = np.array(cams['D'][i], dtype=np.float32)
            R = np.array(cams['R'][i], dtype=np.float32)
            T = np.array(cams['T'][i], dtype=np.float32)

            # World to camera transformation matrix
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = R
            w2c[:3, 3] = T.reshape(3)

            # Camera to world transformation
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            # 이미지 처리 (undistortion 및 resize)
            image = cv2.undistort(image, K, D)
            H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            K[:2] *= image_scaling

            # PIL Image로 변환
            image = Image.fromarray(image)

            print(f"Processed camera {i}:")
            print(f"Image Path: {image_path}")
            print(f"K shape: {K.shape}, R shape: {R.shape}, T shape: {T.shape}")

            # CameraInfo 객체 생성 및 추가
            cam_info = CameraInfo(
                uid=i,
                pose_id=i,
                R=R,
                T=T,
                K=K,
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
                FovY=focal2fov(K[1, 1], H),
                FovX=focal2fov(K[0, 0], W),
                image=image,
                image_path=image_path,
                image_name=os.path.basename(image_path),
                bkgd_mask=None,
                bound_mask=None,
                width=W,
                height=H,
                smpl_param={},
                world_vertex=None,
                world_bound=None,
                big_pose_smpl_param={},
                big_pose_world_vertex=None,
                big_pose_world_bound=None
            )
            cam_infos.append(cam_info)

        except Exception as e:
            print(f"Error processing camera {i}: {str(e)}")
            continue

    if not cam_infos:
        raise ValueError("No valid cameras were loaded")

    print(f"Successfully loaded {len(cam_infos)} cameras")
    return cam_infos

def readZJUMoCapRefineInfo(path, white_background, eval):
    # 모든 뷰를 사용하도록 설정
    all_views = [f'{i:02}' for i in range(100)]  # 100개의 뷰
    train_view = test_view = all_views
    
    print("Reading MoCap Training Transforms")
    train_cam_infos = readCamerasZJUMoCapRefine(path, train_view, white_background, split='train')
    
    if not train_cam_infos:
        raise ValueError("No training cameras were loaded")
    
    print("Reading Test Transforms")
    test_cam_infos = readCamerasZJUMoCapRefine(path, test_view, white_background, split='test', novel_view_vis=False)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    # getNerfppNorm에 전달하기 전에 검증
    if not train_cam_infos:
        raise ValueError("No valid cameras available for normalization")

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(train_view) == 1:
        nerf_normalization['radius'] = 1

    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization
    )
    return scene_info

sceneLoadTypeCallbacks = {
    ###########################################
    "ZJU_MoCap_refine" : readZJUMoCapRefineInfo,
    ###########################################
}