#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
from arguments import ModelParams

from scene.gaussian_model_hj import GaussianModel
from scene.dataset_readers_hj_241021 import sceneLoadTypeCallbacks

from utils.system_utils import searchForMaxIteration
from utils.camera_utils_hj import cameraList_from_camInfos

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        print(os.path.join(args.source_path, "inputs/sfm_scene.json"))
        print(args.source_path)

#######################################################################################################################################
        # elif 'zju_mocap_refine' in args.source_path: #os.path.exists(os.path.join(args.source_path, "annots.npy")):
        if os.path.exists(os.path.join(args.source_path, "annots.npy")):
            print("Found annots.json file, assuming ZJU_MoCap_refine data set!")
            scene_info = sceneLoadTypeCallbacks["ZJU_MoCap_refine"](args.source_path, args.white_background, args.exp_name)
#######################################################################################################################################

        else:
            assert False, "Could not recognize scene type!"

        # if shuffle:
        #     random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        self.scene_info = scene_info

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
