# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import List, Union, Optional


from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.imagesuper_utils import Image_Super_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, subfolder_name):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = 'fast'

        self.pipe_dict = {'hunyuan3d-paint-v2-0': 'hunyuanpaint', 'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo'}
        self.pipe_name = self.pipe_dict[subfolder_name]


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, subfolder='hunyuan3d-paint-v2-0-turbo'):
        original_model_path = model_path
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)

            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    # download from huggingface
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=["hunyuan3d-delight-v2-0/*"]
                    )
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=[f'{subfolder}/*']
                    )
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, subfolder)
                    return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
            else:
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))
        else:
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder))
            
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models()

    def load_models(self):
        # empty cude cache
        torch.cuda.empty_cache()
        # Load model
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)
        # self.models['super_model'] = Image_Super_Net(self.config)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        self.models['delight_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.models['multiview_model'].pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.no_grad()
    def __call__(self, mesh, image):
        import time
        profiling = {}
        total_start = time.time()

        if not isinstance(image, List):
            image = [image]

        # 1. 이미지 전처리
        step_start = time.time()
        print("    → [1/11] 이미지 중앙 정렬 중...")
        images_prompt = []
        for i in range(len(image)):
            if isinstance(image[i], str):
                image_prompt = Image.open(image[i])
            else:
                image_prompt = image[i]
            images_prompt.append(image_prompt)
            
        images_prompt = [self.recenter_image(image_prompt) for image_prompt in images_prompt]
        profiling['1_image_recenter'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['1_image_recenter']:.2f}초")

        # 2. Delight 모델 (그림자/하이라이트 제거)
        step_start = time.time()
        print("    → [2/11] Delight 모델 실행 중 (그림자/하이라이트 제거)...")
        images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]
        profiling['2_delight_model'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['2_delight_model']:.2f}초")

        # 3. UV Wrapping
        step_start = time.time()
        print("    → [3/11] UV Wrapping 중...")
        mesh = mesh_uv_wrap(mesh)
        profiling['3_uv_wrap'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['3_uv_wrap']:.2f}초")

        # 4. 메쉬 로드
        step_start = time.time()
        print("    → [4/11] 메쉬 로드 중...")
        self.render.load_mesh(mesh)
        profiling['4_mesh_load'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['4_mesh_load']:.2f}초")

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        # 5. Normal 맵 렌더링
        step_start = time.time()
        print(f"    → [5/11] Normal 맵 렌더링 중 ({len(selected_camera_elevs)}개 뷰)...")
        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        profiling['5_render_normal'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['5_render_normal']:.2f}초")

        # 6. Position 맵 렌더링
        step_start = time.time()
        print(f"    → [6/11] Position 맵 렌더링 중 ({len(selected_camera_elevs)}개 뷰)...")
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)
        profiling['6_render_position'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['6_render_position']:.2f}초")

        # 7. Multiview 생성 (가장 시간 많이 걸림)
        step_start = time.time()
        print("    → [7/11] Multiview 모델 실행 중 (멀티뷰 이미지 생성)...")
        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiviews = self.models['multiview_model'](images_prompt, normal_maps + position_maps, camera_info)
        profiling['7_multiview_model'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['7_multiview_model']:.2f}초")

        # 8. 이미지 리사이즈
        step_start = time.time()
        print("    → [8/11] 이미지 리사이즈 중...")
        for i in range(len(multiviews)):
            # multiviews[i] = self.models['super_model'](multiviews[i])
            multiviews[i] = multiviews[i].resize(
                (self.config.render_size, self.config.render_size))
        profiling['8_image_resize'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['8_image_resize']:.2f}초")

        # 9. 텍스처 베이킹
        step_start = time.time()
        print("    → [9/11] 텍스처 베이킹 중...")
        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)
        profiling['9_texture_bake'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['9_texture_bake']:.2f}초")

        # 10. 텍스처 인페인팅
        step_start = time.time()
        print("    → [10/11] 텍스처 인페인팅 중...")
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = self.texture_inpaint(texture, mask_np)
        profiling['10_texture_inpaint'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['10_texture_inpaint']:.2f}초")

        # 11. 메쉬 저장
        step_start = time.time()
        print("    → [11/11] 최종 메쉬 저장 중...")
        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()
        profiling['11_mesh_save'] = time.time() - step_start
        print(f"    ✓ 완료: {profiling['11_mesh_save']:.2f}초")

        profiling['TOTAL'] = time.time() - total_start

        # 프로파일링 결과 출력
        print("\n" + "="*60)
        print("🔍 텍스처 생성 단계별 시간 분석")
        print("="*60)
        for step, elapsed in profiling.items():
            if step != 'TOTAL':
                percentage = (elapsed / profiling['TOTAL']) * 100
                print(f"  {step:25s}: {elapsed:6.2f}초 ({percentage:5.1f}%)")
        print("-"*60)
        print(f"  {'TOTAL':25s}: {profiling['TOTAL']:6.2f}초")
        print("="*60 + "\n")

        return textured_mesh
