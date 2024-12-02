
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

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

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
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
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
               "num_contrib": num_contrib}
    
    return results

def calculate_loss(viewpoint_camera, pc, render_pkg, opt):
    # 입력 이미지로부터 마스크 생성
    input_image = viewpoint_camera.original_image.cuda()
    # 완전히 검은색(0,0,0)인 부분만 배경으로 처리
    object_mask = (input_image.sum(dim=0) > 0.0).float()

    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["opacity"]
    rendered_depth = render_pkg["depth"]
    rendered_normal = render_pkg["normal"]
    gt_image = viewpoint_camera.original_image.cuda()
    
    # Mask를 적용한 loss 계산
    loss = 0
    
    # RGB Loss with mask
    Ll1 = F.l1_loss(rendered_image * object_mask, gt_image * object_mask)
    ssim_val = ssim(rendered_image * object_mask, gt_image * object_mask)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    # Depth Loss with mask
    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        depth_mask = gt_depth > 0
        # 객체 영역과 depth가 있는 영역의 교집합에만 적용
        valid_mask = object_mask * depth_mask
        loss_depth = F.l1_loss(rendered_depth * valid_mask, gt_depth * valid_mask)
        loss = loss + opt.lambda_depth * loss_depth

    # Opacity Loss
    if opt.lambda_mask_entropy > 0:
        # 객체 영역은 높은 opacity, 배경은 낮은 opacity로 제약
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(object_mask * torch.log(o) + (1-object_mask) * torch.log(1 - o)).mean()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    # Normal Loss with mask
    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = render_pkg['pseudo_normal']
        loss_normal = F.mse_loss(
            rendered_normal * object_mask, 
            normal_pseudo.detach() * object_mask
        )
        loss = loss + opt.lambda_normal_render_depth * loss_normal

    # 추가적인 배경 제약
    background_opacity_reg = (rendered_opacity * (1 - object_mask)).mean()
    loss = loss + 0.1 * background_opacity_reg  # 배경의 opacity를 0에 가깝게

    return loss, {"loss": loss.item()}

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