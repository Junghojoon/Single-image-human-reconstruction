
# This is script for 3D Gaussian Splatting rendering

import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim
from utils.image_utils import psnr
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer
### add
import trimesh
import numpy as np
from PIL import Image
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, TexturesVertex, RasterizationSettings, MeshRasterizer

import os
import torch
import trimesh
from PIL import Image
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, TexturesVertex, RasterizationSettings, MeshRasterizer
from pytorch3d.ops import interpolate_face_attributes

def tensor_to_image(tensor, file_name):
    tensor = tensor.detach().cpu()

    # Check if the tensor is already in (H, W) format
    if len(tensor.shape) == 3:  # Assumes 3D tensor, check the shape
        if tensor.shape[0] in [1, 3]:  # If the first dimension is a color channel
            tensor = tensor.permute(1, 2, 0)  # Convert to (H, W, C)

    # Convert tensor to numpy before scaling and changing data type
    tensor = tensor.numpy()

    # Normalize the tensor values to [0, 255]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).astype(np.uint8)  # Convert to uint8 type for saving as image

    # Convert numpy array to image using PIL and save
    img = Image.fromarray(tensor)
    img.save(file_name)

def obj_to_mesh(device, obj_path):
    # obj 파일을 읽어서 mesh로 변환
    obj_mesh = trimesh.load_mesh(obj_path)
    verts = torch.Tensor(obj_mesh.vertices).to(device)
    faces = torch.Tensor(obj_mesh.faces).to(device)

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Create a Meshes object for the obj file. Here we have only one mesh in the batch.
    mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)], 
        textures=textures
    )
    return mesh

def get_vis_faceset(device, obj_path):
    # obj 파일을 읽어서 mesh로 변환
    mesh = obj_to_mesh(device, obj_path)
    
    # Select the viewpoint using spherical angles  
    distance = 1   # distance from camera to the object
    elevation = 0.0   # angle of elevation in degrees
    azimuth = 0.0  # No rotation so the camera is positioned on the +Z axis. 

    # Get the position of the camera based on the spherical angles
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=800, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )

    fragments = rasterizer(mesh)

    # (V, 3) 각 픽셀들에 대해서 대응하는 face의 normal 계산 
    packed_normals = mesh.verts_normals_packed()
    faces_normals = packed_normals[mesh.faces_packed()].to(device)
    
    # 각 픽셀에 대한 보간된 노말 계산
    pixel_normals = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_normals) # (batch, h, w, pixel, (x, y, z))
    pixel_normals = pixel_normals.squeeze(dim=3).squeeze(dim=0).permute(2, 0, 1)

    return pixel_normals

# Define your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the path to your .obj file
obj_file_path = "./output/NeRF_Syn/my_386/241004_results/0_000000.obj"
# Define where to save the output normal map image
# output_normal_map_path = "/output/NeRF_Syn/my_386/241004_mesh_normal/mesh_normal_map.png"

# Call the function to generate the pixel normals from the .obj file
pixel_normals = get_vis_faceset(device, obj_file_path)

# Save the generated normal map as an image
# tensor_to_image(pixel_normals, output_normal_map_path)

# print(f"Normal map saved at {output_normal_map_path}")

# def process_obj_files_in_folder(folder_path, device):
#     # 폴더 내의 모든 obj 파일을 가져오기
#     obj_files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]
#     print(f"폴더 경로: {folder_path}")
#     print(f"발견된 .obj 파일: {obj_files}")
    
#     if not obj_files:
#         raise ValueError("폴더 내에 .obj 파일이 없습니다.")
    
#     # obj 파일이 없거나 처리 실패 시 None으로 초기화
#     pixel_normals = None
    
#     # 각 obj 파일을 처리
#     for obj_file in obj_files:
#         obj_path = os.path.join(folder_path, obj_file)
#         print(f"Processing: {obj_path}")
        
#         # get_vis_faceset 호출 및 노말 처리
#         pixel_normals = get_vis_faceset(device, obj_path)
        
#         if pixel_normals is None:
#             print(f"{obj_file}의 노말 계산이 실패했습니다. 계속 진행합니다.")
#             continue  # 다음 파일로 넘어감
        
#         # 저장 예시
#         output_file = f"{os.path.splitext(obj_file)[0]}_normals.png"
#         tensor_to_image(pixel_normals, output_file)
#         print(f"{obj_file}의 노말 이미지를 {output_file}로 저장했습니다.")

#     if pixel_normals is None:
#         raise ValueError("obj 파일 처리가 실패했습니다.")
    
#     return pixel_normals

# # CUDA 디바이스 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# folder_path = "/data/3D_data/mocap_hj/my_386/0_000000.obj"
# pixel_normals = process_obj_files_in_folder(folder_path, device)

### add

def render_view(camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
                scaling_modifier, override_color, computer_pseudo_normal=True):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    intrinsic = camera.intrinsics
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=computer_pseudo_normal,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    ### add alpha blend
    # Rasterize visible Gaussians to alpha mask image.
    raster_settings_alpha = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        computer_pseudo_normal=computer_pseudo_normal,
        backward_geometry=True,
        prefiltered=False,
        debug=False
    )
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
    alpha = torch.ones_like(pc.get_xyz)
    out_extras = {}
    out_extras["alpha"] =  rasterizer_alpha(
        means3D = pc.get_xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = alpha,
        opacities = pc.get_opacity,
        scales = pc.get_scaling,
        rotations = pc.get_rotation,
        cov3D_precomp = None)[0]
    ### add alpha blend

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    # cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - camera.camera_center.repeat(pc.get_shs.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    features = pc.get_normal
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer( ### add alpha
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )
     
    rendered_normal = rendered_feature
    
    # rendered_normal과 pseudo_normal을 수직으로 반전시키는 코드 추가
    # rendered_normal = torch.flip(rendered_normal, dims=[1])  # Y축 기준 반전
    # rendered_pseudo_normal = torch.flip(rendered_pseudo_normal, dims=[1])  # Y축 기준 반전

    # mesh_normal, mesh_mask = pc.get_mesh_normal(camera.world_view_transform, camera.full_proj_transform)
    # print(mesh_mask.min(), mesh_mask.max(), mesh_mask)
    # mesh_normal = mesh_normal * 0.5 + 0.5
    # mask = mesh_mask.expand(3, -1, -1) < 0.95
    # print(mask)
    # mesh_normal[mask] == 0
    mesh_normal = None

    # from torchvision.utils import save_image
    # save_image(rendered_image, "./debug_rendered_images.png")
    results = {"render": rendered_image,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "mesh_normal": mesh_normal,
               "num_contrib": num_contrib,
               ### add
               "alpha" : out_extras["alpha"]}
                ### add
    return results

### add losses
def zero_one_loss(img):
    zero_epsilon = 1e-3
    
    # img가 텐서가 아닌 경우 텐서로 변환
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def predicted_normal_loss(normal, normal_ref):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (B, 3, H, W), normal_ref: (B, 3, H, W)

    # 만약 배치 차원이 있으면 배치 차원을 제거
    if normal_ref.dim() == 4:
        normal_ref = normal_ref.squeeze(0)
    if normal.dim() == 4:
        normal = normal.squeeze(0)

    # 크기 확인
    if normal.shape != normal_ref.shape:
        # normal_ref를 normal의 크기에 맞추기
        normal_ref = F.interpolate(normal_ref.unsqueeze(0), size=normal.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

    # Reshape normals for loss computation
    n = normal_ref.permute(1, 2, 0).reshape(-1, 3).detach()  # Reference normals
    n_pred = normal.permute(1, 2, 0).reshape(-1, 3)  # Predicted normals

    # Compute loss
    loss = (1.0 - torch.sum(n * n_pred, axis=-1)).mean()

    return loss

# def predicted_normal_loss(normal, normal_ref):
#     """Computes the predicted normal supervision loss defined in ref-NeRF."""
#     # normal: (B, 3, H, W), normal_ref: (B, 3, H, W)

#     # 만약 배치 차원이 있으면 배치 차원을 제거
#     if normal_ref.dim() == 4:
#         normal_ref = normal_ref.squeeze(0)
#     if normal.dim() == 4:
#         normal = normal.squeeze(0)

#     # Reshape normals for loss computation
#     n = normal_ref.permute(1, 2, 0).reshape(-1, 3).detach()  # Reference normals
#     n_pred = normal.permute(1, 2, 0).reshape(-1, 3)  # Predicted normals

#     # Compute loss
#     loss = (1.0 - torch.sum(n * n_pred, axis=-1)).mean()

#     return loss

# def delta_normal_loss(delta_normal_norm): 
#     # delta_normal_norm: (B, 3, H, W) 또는 (3, H, W)일 수 있음

#     # 배치 차원이 있을 경우 제거
#     if delta_normal_norm.dim() == 4:
#         delta_normal_norm = delta_normal_norm.squeeze(0)

#     # Create weight tensor with all ones (delta_normal_norm과 같은 크기)
#     weight = torch.ones_like(delta_normal_norm)

#     # Reshape the tensors for loss computation
#     w = weight.permute(1, 2, 0).reshape(-1, 3)[..., 0].detach()  # Weight is all ones
#     l = delta_normal_norm.permute(1, 2, 0).reshape(-1, 3)[..., 0]  # Delta normal norm

#     # Compute the loss
#     loss = (w * l).mean()

#     return loss
### add losses

def calculate_loss(viewpoint_camera, pc, render_pkg, opt):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    # rendered_opacity = render_pkg["opacity"]
    
    ### add
    # rendered_depth = render_pkg["depth"]
    rendered_normal = render_pkg["normal"]
    pseudo_normal = pixel_normals
    alpha = render_pkg['alpha']
    # occ = render_pkg["occ"]
    ### add
    
    gt_image = viewpoint_camera.original_image.cuda()
    # image_mask = viewpoint_camera.image_mask.cuda()
    
    ####add
    # # 텐서가 2D일 경우 (H, W) -> (1, 1, H, W)로 변환
    # if len(rendered_normal.shape) == 2:
    #     rendered_normal = rendered_normal.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    # elif len(rendered_normal.shape) == 3:
    #     rendered_normal = rendered_normal.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

    # pseudo_normal도 동일한 방식으로 확인 및 변환
    # if len(pseudo_normal.shape) == 2:
    #     pseudo_normal = pseudo_normal.unsqueeze(0).unsqueeze(0)
    # elif len(pseudo_normal.shape) == 3:
    #     pseudo_normal = pseudo_normal.unsqueeze(0)

    # rendered_normal_resized = F.interpolate(rendered_normal, size=pseudo_normal.shape[-2:], mode='bilinear', align_corners=False)

    ###add
    # 텐서 크기를 확인합니다.
    print(f"rendered_normal.shape: {rendered_normal.shape}")
    print(f"pseudo_normal.shape: {pseudo_normal.shape}")

    # rendered_normal이 1D나 3D일 경우 4D로 변환
    if len(rendered_normal.shape) == 2:  # (H, W) -> (1, 1, H, W)
        rendered_normal = rendered_normal.unsqueeze(0).unsqueeze(0)
    elif len(rendered_normal.shape) == 3:  # (C, H, W) -> (1, C, H, W)
        rendered_normal = rendered_normal.unsqueeze(0)

    # pseudo_normal과 크기를 맞추기 위해 interpolate 사용
    rendered_normal_resized = F.interpolate(rendered_normal, size=pseudo_normal.shape[-2:], mode='bilinear', align_corners=False)

    # Loss 계산
    Ll1 = F.l1_loss(rendered_normal_resized, pseudo_normal)
    ###add
    
    # Ll1 = F.l1_loss(rendered_normal, pseudo_normal) # rendered_image, gt_image
    ssim_val = ssim(rendered_image, gt_image)
    
    ## add losses -->
    normal_loss = predicted_normal_loss(rendered_normal, pseudo_normal)
    # delta_n_loss = delta_normal_loss(rendered_normal)
    zeroone_loss = zero_one_loss(alpha)
    ## <-- add losses
    
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    
    ## add losses -->
    tb_dict["normal_loss"] = normal_loss.item()
    # tb_dict["delta_n_loss"] = delta_n_loss.item()
    tb_dict["zeroone_loss"] = zeroone_loss.item()
    
    # lambda_zero_one = 1e-3
    # lambda_predicted_normal = 2e-1
    # lambda_delta_reg = 1e-3
    ## <-- add losses
    
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val) + (normal_loss * 2e-1) + (zeroone_loss * 1e-3)
    # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val) + normal_loss + delta_n_loss + zeroone_loss
    
    ## add losses print ----->
    print('Ll1', Ll1)
    print('ssim_val', ssim_val)
    print('normal_loss', normal_loss)
    # print('delta_n_loss', delta_n_loss)
    print('zeroone_loss', zeroone_loss)
    # print("alpha min:", alpha.min(), "alpha max:", alpha.max())
    print('loss', loss)
    ## <----- add losses print
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def render_bind(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
           scaling_modifier=1.0,override_color=None, opt: OptimizationParams = None, 
           is_training=False, dict_params=None):
    
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    
    results = render_view(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color,
                          computer_pseudo_normal=True if opt is not None and opt.lambda_normal_render_depth>0 else False)

    results["hdr"] = viewpoint_camera.hdr

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt)
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    
    return results