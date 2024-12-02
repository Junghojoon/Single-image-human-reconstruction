import trimesh

# OBJ 파일 경로
obj_file_path = '/data/3D_data/mocap_hj/ours/images/mesh.obj'
# /n/output/NeRF_Syn/0_000000.obj
# PLY 파일로 저장할 경로
ply_file_path = './output/1016_mocap/ours/mesh_neus_decimate.ply'

# OBJ 파일 로드
mesh = trimesh.load(obj_file_path)

# PLY 파일로 저장
mesh.export(ply_file_path)

print(f"OBJ 파일이 {ply_file_path} 경로로 변환되었습니다.")