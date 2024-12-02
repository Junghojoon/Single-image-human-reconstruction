import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from utils.graphics_utils import fov2focal
from utils.general_utils import PILtoTorch



WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 3200:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    
    scale_cx = cam_info.cx
    scale_cy = cam_info.cy
    scale_fx = cam_info.fx
    scale_fy = cam_info.fy

    gt_image = resized_image_rgb[:3, ...]
    # loaded_mask = None

    # if resized_image_rgb.shape[1] == 4:
    #     loaded_mask = resized_image_rgb[3:4, ...]

    if cam_info.bound_mask is not None:
        resized_bound_mask = PILtoTorch(cam_info.bound_mask, resolution)
    else:
        resized_bound_mask = None

    # if cam_info.bkgd_mask is not None:
    #     resized_bkgd_mask = PILtoTorch(cam_info.bkgd_mask, resolution)
    # else:
    #     resized_bkgd_mask = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, image_path=cam_info.image_path,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  fx=scale_fx, fy=scale_fy, cx=scale_cx, cy=scale_cy,
                  image=gt_image,
                  image_name=cam_info.image_name, uid=id, 
                  image_mask=resized_bound_mask, 
                  data_device=args.data_device)

    # return Camera(colmap_id=cam_info.uid, pose_id=cam_info.pose_id, R=cam_info.R, T=cam_info.T, K=cam_info.K, 
    #               FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
    #               image=gt_image, gt_alpha_mask=loaded_mask,
    #               image_name=cam_info.image_name, uid=id, bkgd_mask=resized_bkgd_mask, 
    #               bound_mask=resized_bound_mask, smpl_param=cam_info.smpl_param, 
    #               world_vertex=cam_info.world_vertex, world_bound=cam_info.world_bound, 
    #               big_pose_smpl_param=cam_info.big_pose_smpl_param, 
    #               big_pose_world_vertex=cam_info.big_pose_world_vertex, 
    #               big_pose_world_bound=cam_info.big_pose_world_bound, 
    #               data_device=args.data_device)
    
####################################################################################################################

# def loadCam(args, id, cam_info, resolution_scale):
#     orig_h, orig_w = cam_info.image.shape[:2]

#     if args.resolution in [1, 2, 4, 8]:
#         scale = resolution_scale * args.resolution
#     else:  # should be a type that converts to float
#         if args.resolution == -1:
#             if orig_w > 1600:
#                 global WARNED
#                 if not WARNED:
#                     print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
#                           "If this is not desired, please explicitly specify '--resolution/-r' as 1")
#                     WARNED = True
#                 global_down = orig_w / 1600
#             else:
#                 global_down = 1
#         else:
#             global_down = orig_w / args.resolution

#         scale = global_down * resolution_scale
#     resolution = (int(orig_h / scale), int(orig_w / scale))

#     image = torch.from_numpy(cam_info.image).float().permute(2, 0, 1)
#     if scale == 1:
#         resized_image_rgb = image
#     else:
#         resized_image_rgb = torchvision.transforms.Resize(resolution, antialias=True)(image)
#     gt_image = resized_image_rgb

#     resized_depth = None
#     if cam_info.depth is not None:
#         depth = torch.from_numpy(cam_info.depth).float().unsqueeze(0)
#         resized_depth = torchvision.transforms.Resize(
#             resolution, interpolation=InterpolationMode.NEAREST)(depth)

#     resized_normal = None
#     if cam_info.normal is not None:
#         normal = torch.from_numpy(cam_info.normal).float().permute(2, 0, 1)
#         resized_normal = torchvision.transforms.Resize(
#             resolution, interpolation=InterpolationMode.NEAREST)(normal)

#     resized_image_mask = None
#     if cam_info.image_mask is not None:
#         image_mask = torch.from_numpy(cam_info.image_mask).float().unsqueeze(0)
#         resized_image_mask = torchvision.transforms.Resize(
#             resolution, interpolation=InterpolationMode.NEAREST)(image_mask)

#     # change the fx and fy
#     scale_cx = cam_info.cx
#     scale_cy = cam_info.cy
#     scale_fx = cam_info.fx
#     scale_fy = cam_info.fy
#     if cam_info.cx is not None and cam_info.cy is not None:
#         scale_cx /= scale
#         scale_cy /= scale
#         scale_fx /= scale
#         scale_fy /= scale

#     return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
#                   FoVx=cam_info.FovX, FoVy=cam_info.FovY, fx=scale_fx, fy=scale_fy, cx=scale_cx, cy=scale_cy,
#                   image=gt_image, depth=resized_depth, normal=resized_normal, image_mask=resized_image_mask,
#                   image_name=cam_info.image_name, uid=id, data_device=args.data_device, hdr=cam_info.hdr)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos, desc="resolution scale: {}".format(resolution_scale), leave=False)):
        # for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

# def camera_to_JSON(id, camera):
#     Rt = np.zeros((4, 4))
#     Rt[:3, :3] = camera.R.transpose()
#     Rt[:3, 3] = camera.T
#     Rt[3, 3] = 1.0

#     W2C = np.linalg.inv(Rt)
#     pos = W2C[:3, 3]
#     rot = W2C[:3, :3]
#     serializable_array_2d = [x.tolist() for x in rot]

#     if camera.cx is None:
#         camera_entry = {
#             'id': id,
#             'img_name': camera.image_name,
#             'width': camera.width,
#             'height': camera.height,
#             'position': pos.tolist(),
#             'rotation': serializable_array_2d,
#             'FoVx': camera.FovX,
#             'FoVy': camera.FovY,
#         }
#     else:
#         camera_entry = {
#             'id': id,
#             'img_name': camera.image_name,
#             'width': camera.width,
#             'height': camera.height,
#             'position': pos.tolist(),
#             'rotation': serializable_array_2d,
#             'fx': camera.fx,
#             'fy': camera.fy,
#             'cx': camera.cx,
#             'cy': camera.cy,
#         }
#     return camera_entry


def JSON_to_camera(json_cam):
    rot = np.array(json_cam['rotation'])
    pos = np.array(json_cam['position'])
    W2C = np.zeros((4, 4))
    W2C[:3, :3] = rot
    W2C[:3, 3] = pos
    W2C[3, 3] = 1
    Rt = np.linalg.inv(W2C)
    R = Rt[:3, :3].transpose()
    T = Rt[:3, 3]
    H, W = json_cam['height'], json_cam['width']
    if 'cx' not in json_cam:
        if 'fx' in json_cam:
            FovX = focal2fov(json_cam["fx"], W)
            FovY = focal2fov(json_cam["fy"], H)
        else:
            FovX = json_cam["FoVx"]
            FovY = json_cam["FoVy"]
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=FovX, FoVy=FovY, fx=None, fy=None, cx=None, cy=None,
                        image=None, image_name=json_cam['img_name'], uid=json_cam['id'],
                        data_device='cuda', height=H, width=W)
    else:
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=None, FoVy=None, fx=json_cam["fx"], fy=json_cam["fy"],
                        cx=json_cam["cx"], cy=json_cam["cy"], image=None, image_name=json_cam['img_name'],
                        uid=json_cam['id'], data_device='cuda', height=H, width=W)
    return camera
