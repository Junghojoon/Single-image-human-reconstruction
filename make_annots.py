# import os
# import numpy as np
# import json

# # JSON 파일과 이미지 파일이 저장된 폴더 경로
# data_dir = "/data/3D_data/mocap_hj/ours/images"  # 실제 경로로 변경하세요
# output_dir = "/data/3D_data/mocap_hj/ours"
# os.makedirs(output_dir, exist_ok=True)

# # annots 파일에 포함될 데이터 구조 초기화
# annotations = {
#     "cams": {
#         "K": [],  # Intrinsic matrix
#         "D": [],  # Distortion coefficients
#         "R": [],  # Rotation matrix
#         "T": []   # Translation vector
#     },
#     "ims": []  # Image paths with multiple views
# }

# # 폴더 내 모든 JSON 및 PNG 파일 검색
# json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
# image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])

# # 각 JSON 파일에 대해 대응하는 이미지 파일을 찾아 annots 생성
# for json_file in json_files:
#     # 카메라 JSON 파일 경로
#     json_path = os.path.join(data_dir, json_file)
    
#     # JSON 파일 로드
#     with open(json_path, 'r') as f:
#         camera_params = json.load(f)
    
#     # JSON 파일 이름에서 인덱스 추출
#     try:
#         index = json_file.split('_')[1].split('.')[0]
#     except IndexError:
#         print(f"잘못된 파일 이름 형식: {json_file}")
#         continue

#     # 이미지 경로 설정 (해당 인덱스에 맞는 100개의 뷰 추가)
#     image_paths = []
#     for view_index in range(100):  # 100개의 뷰 사용
#         image_name = f"render_{view_index}.png"
#         if image_name in image_files:
#             image_paths.append(os.path.join(data_dir, image_name))
#         else:
#             print(f"{image_name}에 해당하는 이미지 파일을 찾을 수 없습니다.")
#             continue

#     if not image_paths:
#         print(f"뷰를 찾을 수 없습니다: {json_file}")
#         continue

#     # 카메라 정보 추가 (JSON에서 동적으로 가져옴)
#     try:
#         annotations["cams"]["K"].append(camera_params["cams"]["K"])
#         annotations["cams"]["D"].append(camera_params["cams"]["D"])
#         annotations["cams"]["R"].append(camera_params["cams"]["R"])
#         annotations["cams"]["T"].append(camera_params["cams"]["T"])
#     except KeyError as e:
#         print(f"카메라 파라미터에서 누락된 키: {e}")
#         continue

#     # 이미지 정보 추가 (`ims`에 100개의 이미지 경로 추가)
#     annotations["ims"].append({"ims": image_paths})

# # annots.npy 파일로 저장
# annots_path = os.path.join(output_dir, "annots.npy")
# np.save(annots_path, annotations)

# print(f"'annots.npy' 파일이 '{output_dir}' 디렉토리에 생성되었습니다.")

import os
import numpy as np
import json
from collections import defaultdict

def create_annots(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # annots 초기화
    annotations = {
        "cams": defaultdict(list),
        "ims": []
    }

    # 카메라 JSON 파일과 이미지 파일 찾기
    camera_files = sorted([f for f in os.listdir(data_dir) if f.startswith('camera_') and f.endswith('.json')])
    image_files = sorted([f for f in os.listdir(data_dir) if f.startswith('render_') and f.endswith('.png')])

    # 파일 개수 검증
    if len(camera_files) != len(image_files):
        raise ValueError(f"Number of camera files ({len(camera_files)}) does not match number of image files ({len(image_files)})")

    print(f"Found {len(camera_files)} camera files and {len(image_files)} image files")

    # 카메라 파라미터 수집
    for camera_file in camera_files:
        try:
            # 카메라 인덱스 추출
            camera_idx = int(camera_file.split('_')[1].split('.')[0])
            camera_path = os.path.join(data_dir, camera_file)

            # 해당하는 이미지 파일 확인 (인덱스 형식 맞춤)
            expected_image = f"render_{camera_idx:03d}.png"  # 3자리 숫자 형식으로 수정
            if not os.path.exists(os.path.join(data_dir, expected_image)):
                raise ValueError(f"Missing corresponding image file: {expected_image}")

            # 카메라 파라미터 로드
            with open(camera_path, 'r') as f:
                camera_params = json.load(f)['cams']

            # 필수 키 확인
            required_keys = ['K', 'D', 'R', 'T']
            if not all(key in camera_params for key in required_keys):
                raise KeyError(f"Missing required camera parameters in {camera_file}")

            # 파라미터 추가
            for key in required_keys:
                annotations["cams"][key].append(camera_params[key])

        except Exception as e:
            print(f"Error processing camera file {camera_file}: {str(e)}")
            raise

    # 이미지 경로 구성 (3자리 숫자 형식 사용)
    image_paths = [os.path.join("images", f"render_{i:03d}.png") for i in range(len(camera_files))]
    
    # 모든 이미지 존재 확인
    for img_path in image_paths:
        full_path = os.path.join(data_dir, os.path.basename(img_path))
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

    # 이미지 정보 추가
    annotations["ims"].append({"ims": image_paths})

    # defaultdict를 일반 dict로 변환
    annotations["cams"] = dict(annotations["cams"])

    # 데이터 검증
    n_cameras = len(annotations["cams"]["K"])
    n_images = len(annotations["ims"][0]["ims"])
    
    print(f"\nVerification before saving:")
    print(f"Number of K matrices: {len(annotations['cams']['K'])}")
    print(f"Number of D matrices: {len(annotations['cams']['D'])}")
    print(f"Number of R matrices: {len(annotations['cams']['R'])}")
    print(f"Number of T vectors: {len(annotations['cams']['T'])}")
    print(f"Number of images: {n_images}")

    if n_cameras != n_images:
        raise ValueError(f"Number of cameras ({n_cameras}) does not match number of images ({n_images})")

    # annots.npy 저장
    annots_path = os.path.join(output_dir, "annots.npy")
    np.save(annots_path, annotations)

    # 저장된 파일 검증
    loaded_annots = np.load(annots_path, allow_pickle=True).item()
    
    print("\nValidation Results:")
    print(f"Number of cameras: {len(loaded_annots['cams']['K'])}")
    print(f"Number of image sets: {len(loaded_annots['ims'])}")
    print(f"Number of images in set: {len(loaded_annots['ims'][0]['ims'])}")
    print(f"First camera K matrix shape: {np.array(loaded_annots['cams']['K'][0]).shape}")
    print(f"First camera R matrix shape: {np.array(loaded_annots['cams']['R'][0]).shape}")
    
    # 첫 번째 이미지 경로 출력
    print(f"\nFirst image path: {loaded_annots['ims'][0]['ims'][0]}")
    
    print(f"\nSuccessfully cr veated annots.npy at {annots_path}")

    return loaded_annots

if __name__ == "__main__":
    # 경로 설정
    data_dir = "/data/3D_data/mocap_hj/ours/images"
    output_dir = "/data/3D_data/mocap_hj/ours"
    
    try:
        annots = create_annots(data_dir, output_dir)
    except Exception as e:
        print(f"Error creating annotations: {str(e)}")