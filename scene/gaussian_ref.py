
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import rotation_to_quaternion, quaternion_multiply, quaternion_to_rotation_matrix
from utils.sh_utils import RGB2SH, eval_sh
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from arguments import OptimizationParams
from tqdm import tqdm
from mesh import Mesh, safe_normalize, dot
from mesh_utils import decimate_mesh, clean_mesh, poisson_mesh_reconstruction
import mcubes
import trimesh
import nvdiffrast.torch as dr
from torchvision.utils import save_image, make_grid

# ----------------------------------------------------
from scene.models import flip_align_view
import open3d as o3d
# ----------------------------------------------------

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        if self.use_pbr:
            self.base_color_activation = torch.sigmoid
            self.roughness_activation = torch.sigmoid
            self.metallic_activation = torch.sigmoid

    def __init__(self, sh_degree: int, render_type='render'):
        self.render_type = render_type
        self.use_pbr = render_type in ['neilf']
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        
        self._normal = torch.empty(0) # normal 1 원래는 그냥 normal 하나 만 있음
        self._normal2 = torch.empty(0) # normal 2
        
        self._shs_dc = torch.empty(0) # output radiance
        self._shs_rest = torch.empty(0) # output radiance
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.normal_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()
        self.transform = {}
        if self.use_pbr:
            self._base_color = torch.empty(0)
            self._roughness = torch.empty(0)
            self._metallic = torch.empty(0)
            self._incidents_dc = torch.empty(0)
            self._incidents_rest = torch.empty(0)
            self._visibility_dc = torch.empty(0)
            self._visibility_rest = torch.empty(0)

        # load mesh.ply
        if self.render_type in ["normal", "bind"]:
            # if self.opt.gui:
            #     self.glctx = dr.RasterizeCudaContext() # support at most 2048 resolution.
            # else:
            #     self.glctx = dr.RasterizeGLContext(output_db=False) # will crash if using GUI...
            self.glctx = dr.RasterizeCudaContext() # support at most 2048 resolution.
            self.vertices = torch.empty(0)
            self.triangles = torch.empty(0)

            self._vertices_offsets = torch.empty(0)
        
        self.use_offset = False
        self.use_anchor_field = False
        self.fix_gaus = False
        self.use_precompute_global = False
        if self.render_type in ["bind"]:
            self.triangles_points = torch.empty(0)
     
    
    def get_normal(self, dir_pp_normalized=None, return_delta=False):
        normal_axis = self.get_minimum_axis
        normal_axis = normal_axis
        normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
        delta_normal1 = self._normal  # (N, 3) 
        delta_normal2 = self._normal2 # (N, 3) 
        delta_normal = torch.stack([delta_normal1, delta_normal2], dim=-1) # (N, 3, 2)
        idx = torch.where(positive, 0, 1).long()[:,None,:].repeat(1, 3, 1) # (N, 3, 1)
        delta_normal = torch.gather(delta_normal, index=idx, dim=-1).squeeze(-1) # (N, 3)
        normal = delta_normal + normal_axis 
        normal = normal/normal.norm(dim=1, keepdim=True) # (N, 3)
        if return_delta:
            return normal, delta_normal
        else:
            return normal
        
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if not self.brdf:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        elif (self.brdf_mode=="envmap" and self.brdf_dim==0):
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features = torch.zeros((fused_color.shape[0], self.brdf_dim + 3)).float().cuda()
            features[:, :3 ] = fused_color
            features[:, 3: ] = 0.0
        elif self.brdf_mode=="envmap" and self.brdf_dim>0:
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3)).float().cuda()
            features[:, :3 ] = fused_color
            features[:, 3: ] = 0.0
            features_rest = torch.zeros((fused_color.shape[0], 3, (self.brdf_dim + 1) ** 2)).float().cuda()
        else:
            raise NotImplementedError

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        if not self.brdf:
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(features[:,:3].contiguous().requires_grad_(True))
            if (self.brdf_mode=="envmap" and self.brdf_dim==0):
                self._features_rest = nn.Parameter(features[:,3:].contiguous().requires_grad_(True))
            elif self.brdf_mode=="envmap":
                self._features_rest = nn.Parameter(features_rest.contiguous().requires_grad_(True))

            normals = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
            normals2 = np.copy(normals)

            self._normal = nn.Parameter(torch.from_numpy(normals).to(self._xyz.device).requires_grad_(True))
            specular_len = 3 
            self._specular = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], specular_len), device="cuda").requires_grad_(True))
            self._roughness = nn.Parameter(self.default_roughness*torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
            self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(self._xyz.device).requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def training_setup(self, training_args):
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.brdf:
            self._normal.requires_grad_(requires_grad=False)
            l.extend([
                {'params': list(self.brdf_mlp.parameters()), 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"},
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            ])
            self._normal2.requires_grad_(requires_grad=False)
            l.extend([
                {'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"},
            ])
            
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        self._update_learning_rate(iteration, "xyz")
        if self.brdf and not self.fix_brdf_lr:
            for param in ["brdf_mlp","roughness","specular","normal","f_dc", "f_rest"]:
                lr = self._update_learning_rate(iteration, param)
    
    def save_ply(self, path, viewer_fmt=False):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz) if not self.brdf else self._normal.detach().cpu().numpy()
        normals2 = self._normal2.detach().cpu().numpy() if (self.brdf) else np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() if not self.brdf else self._features_dc.detach().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() if not ((self.brdf and self.brdf_mode=="envmap" and self.brdf_dim==0)) else self._features_rest.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        roughness = None if not self.brdf else self._roughness.detach().cpu().numpy()
        specular = None if not self.brdf else self._specular.detach().cpu().numpy()
        
        if viewer_fmt:
            f_dc = 0.5 + (0.5*normals)
            f_rest = np.zeros((f_rest.shape[0], 45))
            normals = np.zeros_like(normals)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(viewer_fmt)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.brdf and not viewer_fmt:
            attributes = np.concatenate((xyz, normals, normals2, f_dc, f_rest, opacities, scale, rotation, roughness, specular), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz)
        pcd_o3d.colors = o3d.utility.Vector3dVector(f_dc)
        return pcd_o3d
        
    def load_ply(self, path, og_number_points=-1):
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if not self.brdf:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        else:
            features_dc = np.zeros((xyz.shape[0], 3))
            features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        if not self.brdf:
            if not self.brdf:
                assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        elif self.brdf_mode=="envmap":
            features_extra = np.zeros((xyz.shape[0], 3*(self.brdf_dim + 1) ** 2 ))
            if len(extra_f_names)==3*(self.brdf_dim + 1) ** 2:
                for idx, attr_name in enumerate(extra_f_names):
                    features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
                features_extra = features_extra.reshape((features_extra.shape[0], (self.brdf_dim + 1) ** 2, 3))
                features_extra = features_extra.swapaxes(1,2)
            else:
                print(f"NO INITIAL SH FEATURES FOUND!!! USE ZERO SH AS INITIALIZE.")
                features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.brdf_dim + 1) ** 2))
        else:
            assert len(extra_f_names)==self.brdf_dim
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        if self.brdf:
            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

            specular_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("specular")]
            specular = np.zeros((xyz.shape[0], len(specular_names)))
            for idx, attr_name in enumerate(specular_names):
                specular[:, idx] = np.asarray(plydata.elements[0][attr_name])

            normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
            normal2 = np.stack((np.asarray(plydata.elements[0]["nx2"]),
                            np.asarray(plydata.elements[0]["ny2"]),
                            np.asarray(plydata.elements[0]["nz2"])),  axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)) if not self.brdf else nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)) if not ((self.brdf and self.brdf_mode=="envmap")) else nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.brdf:
            self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
            self._specular = nn.Parameter(torch.tensor(specular, dtype=torch.float, device="cuda").requires_grad_(True))
            self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
            self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True))
        
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.brdf:
            self._roughness = optimizable_tensors["roughness"]
            self._specular = optimizable_tensors["specular"]
            self._normal = optimizable_tensors["normal"]
            self._normal2 = optimizable_tensors["normal2"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, \
                              new_roughness, new_specular, new_normal, new_normal2):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.brdf:
            d.update({
                "roughness": new_roughness,
                "specular" : new_specular,
                "normal" : new_normal,
                "normal2" : new_normal2,
            })

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.brdf:
            self._roughness = optimizable_tensors["roughness"]
            self._specular = optimizable_tensors["specular"]
            self._normal = optimizable_tensors["normal"]
            self._normal2 = optimizable_tensors["normal2"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if torch.sum(selected_pts_mask) == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1) if not self.brdf else self._features_dc[selected_pts_mask].repeat(N,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1) if not ((self.brdf and self.brdf_mode=="envmap" and self.brdf_dim==0)) else self._features_rest[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1) if self.brdf else None
        new_specular = self._specular[selected_pts_mask].repeat(N,1) if self.brdf else None
        new_normal = self._normal[selected_pts_mask].repeat(N,1) if self.brdf else None
        new_normal2 = self._normal2[selected_pts_mask].repeat(N,1) if (self.brdf) else None
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, 
                                   new_roughness, new_specular, new_normal, new_normal2)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
        
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if torch.sum(selected_pts_mask) == 0:
            return
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask] if self.brdf else None
        new_specular = self._specular[selected_pts_mask] if self.brdf else None
        new_normal = self._normal[selected_pts_mask] if self.brdf else None
        new_normal2 = self._normal2[selected_pts_mask] if (self.brdf) else None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
                                   new_roughness, new_specular, new_normal, new_normal2)
        