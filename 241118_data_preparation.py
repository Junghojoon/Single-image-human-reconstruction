import os
import torch
import numpy as np
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer import (
    PerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    HardPhongShader,
    TexturesUV,
    BlendParams,
)
import json
import cv2

def pytorch3d_to_opencv_transform(R, T):
    """
    수정된 좌표계 변환
    """
    # 카메라 외부 파라미터 계산
    R_cv = R.copy()  # 회전 행렬은 그대로 사용
    T_cv = T.reshape(3, 1)  # 변환 벡터 reshape
    
    # 4x4 변환 행렬 생성
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_cv
    transform_matrix[:3, 3] = T_cv.reshape(-1)
    
    return R_cv, T_cv, transform_matrix

def calculate_camera_intrinsics(view_idx, num_views, image_size=512):
    """각 뷰에 대한 카메라 내부 파라미터 계산"""
    # FOV를 고정값으로 설정
    fov = 50.0  # 변동 없이 고정
    
    # 초점 거리 계산
    focal_length = (image_size/2) / np.tan(np.deg2rad(fov/2))
    
    # 주점을 이미지 중심으로 고정
    cx = image_size/2.0
    cy = image_size/2.0
    
    # 내부 파라미터 행렬
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 왜곡 계수는 0으로 설정
    D = np.zeros((1, 5), dtype=np.float32)
    
    return K, D

def render_and_save_fixed_with_mesh(i, renderer, mesh, device, output_dir, num_views, fixed_distance=2.5):
    with torch.no_grad():
        # 카메라 설정
        image_size = 512
        K, D = calculate_camera_intrinsics(i, num_views, image_size)
        
        # 각도 계산 수정 (-180도부터 시작)
        angle = (360.0 * i / num_views) - 180.0
        R, T = look_at_view_transform(
            dist=fixed_distance, 
            elev=0.0,
            azim=angle,
            device=device
        )
        R, T = R[0].cpu().numpy(), T[0].cpu().numpy()
        
        # OpenCV 변환
        R_cv, T_cv, transform_matrix = pytorch3d_to_opencv_transform(R, T)
        
        # 카메라 생성
        cameras = PerspectiveCameras(
            focal_length=((K[0,0]/image_size * 2,),),  # 정규화된 초점거리
            principal_point=((0.0, 0.0),),  # NDC 좌표계에서는 (0,0)이 중심
            R=torch.tensor(R).unsqueeze(0).to(device),
            T=torch.tensor(T).unsqueeze(0).to(device),
            device=device,
            in_ndc=True  # NDC 좌표계 사용
        )
        
        # 렌더링 설정 업데이트
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True
        )
        
        # 이미지 렌더링 (RGBA 포맷으로)
        images = renderer(mesh, cameras=cameras)
        image = images[0].cpu().numpy()  # RGBA 채널 모두 유지
        
        # RGBA 이미지로 변환 (0-255 범위로)
        image = (image * 255).astype(np.uint8)
        
        # 검은색 배경으로 변환
        alpha_mask = image[:, :, 3] > 0  # 알파 채널이 0보다 큰 부분이 객체
        black_background = np.zeros_like(image[:, :, :3])  # RGB 검은색 배경
        black_background[alpha_mask] = image[alpha_mask][:, :3]  # 객체 부분만 복사
        
        # 검은색 배경의 이미지만 저장
        image_path = os.path.join(output_dir, f'render_{i:03d}.png')
        cv2.imwrite(image_path, cv2.cvtColor(black_background, cv2.COLOR_RGB2BGR))
        
        # 카메라 파라미터 저장
        camera_params = {
            'K': K.tolist(),
            'D': D.tolist(),
            'R': R_cv.tolist(),
            'T': T_cv.reshape(-1).tolist(),
            'world2cam': transform_matrix.tolist(),
            'cam2world': np.linalg.inv(transform_matrix).tolist(),
            'width': image_size,
            'height': image_size,
            'camera_angle_x': np.deg2rad(50.0),  # FOV in radians
            'camera_angle_y': np.deg2rad(50.0)
        }
        
        camera_info_path = os.path.join(output_dir, f'camera_{i:03d}.json')
        with open(camera_info_path, 'w') as f:
            json.dump({'cams': camera_params}, f, indent=4)

        # Debug information
        print(f"\nCamera {i} parameters:")
        print(f"K matrix:\n{K}")
        print(f"D coefficients:\n{D}")
        print(f"Original R:\n{R}")
        print(f"Original T:\n{T}")
        print(f"Converted R_cv:\n{R_cv}")
        print(f"Converted T_cv:\n{T_cv}")
        
        # 메쉬 저장 (OpenCV 좌표계) - 첫 번째 뷰에서만 실행
        if i == 0:  # 첫 번째 뷰에서만 메시 저장
            verts = mesh.verts_padded()[0]
            faces = mesh.faces_padded()[0]
            
            # 정면을 보도록 변환 (x축 반전 추가)
            transform = torch.tensor([
                [-1,  0,  0],    # x축 반전 (좌우 반전을 위해 -1로 변경)
                [0,  1,  0],     # y축 유지
                [0,  0, -1]      # z축 반전
            ], dtype=torch.float32, device=device)
            
            transformed_verts = (torch.matmul(transform, verts.T).T)
            
            # x축 중심으로 180도 회전
            rotation_x_180 = torch.tensor([
                [1,  0,  0],     # x축 유지
                [0, -1,  0],     # y축 반전
                [0,  0, -1]      # z축 반전
            ], dtype=torch.float32, device=device)
            
            transformed_verts = torch.matmul(rotation_x_180, transformed_verts.T).T
            
            mesh_output_path = os.path.join(output_dir, 'mesh.obj')
            save_obj(mesh_output_path, transformed_verts.cpu(), faces.cpu())

        print(f"Rendered view {i + 1}/{num_views}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 파라미터 설정
    num_views = 100
    output_dir = "/data/3D_data/mocap_hj/ours/241128_1_images"
    obj_filename = "/data/3D_data/mocap_hj/ours/000000_texture.obj"
    texture_image_path = "/data/3D_data/mocap_hj/ours/000000_texture_albedo.png"
    
    os.makedirs(output_dir, exist_ok=True)

    # OBJ 파일 로드 및 메쉬 설정
    print("Loading mesh...")
    verts, faces, aux = load_obj(obj_filename, load_textures=True)
    
    # 메쉬를 회전시켜 정면이 카메라를 향하도록 수정
    rotation_matrix = torch.tensor([
        [-1, 0, 0],    # x축 반전
        [0, 1, 0],     # y축 유지
        [0, 0, -1]     # z축 반전
    ], dtype=torch.float32, device=device)
    
    # 버텍스들을 회전
    verts = torch.matmul(verts.to(device), rotation_matrix.t())
    
    texture_image = cv2.imread(texture_image_path)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
    texture_image = torch.from_numpy(texture_image.astype(np.float32) / 255.0).to(device)
    
    textures = TexturesUV(
        maps=texture_image.unsqueeze(0),
        faces_uvs=[faces.textures_idx.to(device)],
        verts_uvs=[aux.verts_uvs.to(device)]
    )
    
    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx.to(device)],
        textures=textures
    )

    # 렌더링 설정
    print("Setting up renderer...")
    lights = PointLights(device=device, location=[[0.0, 0.0, 5.0]])
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1
    )
    
    # 렌더러 설정 수정 - 투명한 배경 설정
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device, 
            lights=lights,
            blend_params=BlendParams(background_color=(0, 0, 0, 0))  # RGBA에서 A=0은 완전 투명
        )
    )

    print("Starting rendering...")
    # 각 뷰에 대해 렌더링 실행
    for i in range(num_views):
        render_and_save_fixed_with_mesh(i, renderer, mesh, device, output_dir, num_views, 
                                      fixed_distance=2.5)

if __name__ == "__main__":
    main()