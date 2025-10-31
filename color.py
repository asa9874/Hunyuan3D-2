import torch
import os
import time
from datetime import datetime
from PIL import Image
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

# ==== 설정 가능한 매개변수 ====
# 경로 설정
INPUT_IMAGE = 'my/input/shirts.jpg'     # 입력 이미지 경로

# 배경 제거 설정
REMOVE_BACKGROUND = True            # 배경 제거 활성화 (True/False)

# 형상 생성 품질 설정
NUM_INFERENCE_STEPS = 5             # 형상 생성 추론 단계 (기본: 5, 범위: 3-10)
OCTREE_RESOLUTION = 192             # Octree 해상도 (128=빠름/낮은품질, 192=균형, 256=느림/높은품질)
GUIDANCE_SCALE = 5                  # 가이던스 스케일 (기본: 5, 범위: 3-10)

# 텍스처 생성 품질 설정
DELIGHT_INFERENCE_STEPS = 6         # 그림자 제거 추론 단계 (기본: 6, 범위: 4-10)
MULTIVIEW_INFERENCE_STEPS = 6       # 멀티뷰 생성 추론 단계 (기본: 6, 범위: 4-10)

# 카메라 뷰 설정 (속도 최적화)
CAMERA_VIEWS = 'standard'           # 'standard'(6뷰,느림), 'fast'(4뷰,빠름), 'minimal'(3뷰,매우빠름)

# 렌더링 설정
RENDER_SIZE = 2048                  # 렌더 해상도 (1024=빠름, 2048=기본, 4096=고품질/느림)
TEXTURE_SIZE = 2048                 # 텍스처 해상도 (1024, 2048, 4096)
# ============================

# 출력 파일명을 입력 파일명과 동일하게 설정 (확장자만 .glb로 변경)
input_filename = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
OUTPUT_PATH = f'my/output/{input_filename}.glb'

# 로그 파일 경로 설정
log_dir = 'my/log'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = os.path.join(log_dir, f'{timestamp}_{input_filename}.txt')

# 시간 측정을 위한 변수
time_records = {}
start_total = time.time()  # 전체 시작 시간

# 0. 이미지 로드 및 배경 제거 (필요시)
print("=" * 60)
print("입력 이미지를 처리합니다...")
print(f"  - 입력 이미지: {INPUT_IMAGE}")
start_time = time.time()

# 원본 이미지 모드 확인 (변환 전)
image = Image.open(INPUT_IMAGE)
original_mode = image.mode
print(f"  - 원본 이미지 모드: {original_mode}")

# 배경 제거 수행 여부 결정
if REMOVE_BACKGROUND and original_mode in ['RGB', 'L']:  # RGB 또는 그레이스케일인 경우
    print("  - 배경이 있는 이미지입니다. 배경 제거를 수행합니다...")
    if original_mode == 'L':
        image = image.convert('RGB')
    rembg = BackgroundRemover()
    image = rembg(image)
    print("  - 배경 제거 완료! (RGBA로 변환됨)")
elif REMOVE_BACKGROUND and original_mode == 'RGBA':
    print("  - 이미지에 이미 알파 채널(투명도)이 있어 배경 제거를 건너뜁니다.")
elif not REMOVE_BACKGROUND:
    print("  - 배경 제거가 비활성화되어 있습니다.")
    image = image.convert("RGBA")
else:
    image = image.convert("RGBA")

time_records['이미지 전처리'] = time.time() - start_time
print(f"완료! (소요 시간: {time_records['이미지 전처리']:.2f}초)")
print("=" * 60)

# 1. 형상(Shape) 생성 파이프라인 로드
print("=" * 60)
print("형상 생성 파이프라인을 로드합니다...")
start_time = time.time()
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-dit-v2-0-turbo',
    torch_dtype=torch.float16
)
shape_pipeline.to('cuda')
time_records['파이프라인 로드'] = time.time() - start_time
print(f"완료! (소요 시간: {time_records['파이프라인 로드']:.2f}초)")
print("=" * 60)

print("=" * 60)
print(f"이미지로부터 3D 모델 형상을 생성합니다...")
print(f"  - 추론 단계: {NUM_INFERENCE_STEPS}")
print(f"  - 해상도: {OCTREE_RESOLUTION}")
print(f"  - 가이던스 스케일: {GUIDANCE_SCALE}")
start_time = time.time()
# shape_pipeline의 결과는 리스트이므로, [0]을 붙여 첫 번째 모델 객체를 꺼냅니다.
mesh = shape_pipeline(
    image=image,  # 전처리된 이미지 사용
    num_inference_steps=NUM_INFERENCE_STEPS,
    octree_resolution=OCTREE_RESOLUTION,
    guidance_scale=GUIDANCE_SCALE
)[0]
time_records['형상 생성'] = time.time() - start_time
print(f"형상 생성이 완료되었습니다! (소요 시간: {time_records['형상 생성']:.2f}초)")
print("=" * 60)

# 2. 텍스처(Texture) 생성 파이프라인 로드
print("\n텍스처 생성 파이프라인을 로드합니다...")
start_time = time.time()

# 카메라 뷰 설정
if CAMERA_VIEWS == 'minimal':
    camera_azims = [0, 120, 240]  # 3뷰: 정면, 좌측 120도, 우측 120도
    camera_elevs = [0, 0, 0]
    view_weights = [1, 0.3, 0.3]
    print(f"  - 카메라 뷰: 최소 (3뷰, 매우 빠름)")
elif CAMERA_VIEWS == 'fast':
    camera_azims = [0, 90, 180, 270]  # 4뷰: 전후좌우
    camera_elevs = [0, 0, 0, 0]
    view_weights = [1, 0.1, 0.5, 0.1]
    print(f"  - 카메라 뷰: 빠름 (4뷰)")
else:  # standard
    camera_azims = [0, 90, 180, 270, 0, 180]  # 6뷰: 전후좌우 + 위아래
    camera_elevs = [0, 0, 0, 0, 90, -90]
    view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
    print(f"  - 카메라 뷰: 표준 (6뷰)")

# 파이프라인 설정을 위한 커스텀 config 클래스
class CustomTexGenConfig:
    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, subfolder_name):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path
        self.candidate_camera_azims = camera_azims
        self.candidate_camera_elevs = camera_elevs
        self.candidate_view_weights = view_weights
        self.render_size = RENDER_SIZE
        self.texture_size = TEXTURE_SIZE
        self.delight_inference_steps = DELIGHT_INFERENCE_STEPS  # 추가
        self.multiview_inference_steps = MULTIVIEW_INFERENCE_STEPS  # 추가
        self.bake_exp = 4
        self.merge_method = 'fast'
        self.pipe_dict = {'hunyuan3d-paint-v2-0': 'hunyuanpaint', 'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo'}
        self.pipe_name = self.pipe_dict[subfolder_name]

# 원래 config 클래스를 임시로 교체
import hy3dgen.texgen.pipelines as texgen_module
original_config = texgen_module.Hunyuan3DTexGenConfig
texgen_module.Hunyuan3DTexGenConfig = CustomTexGenConfig

paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-paint-v2-0-turbo'
)

# 원래 config 복원
texgen_module.Hunyuan3DTexGenConfig = original_config

# ⚡ 중요: 텍스처 생성 모델들을 GPU로 이동 (성능 향상)
print("  - 모델을 GPU로 이동 중...")
paint_pipeline.models['delight_model'].pipeline.to('cuda')
paint_pipeline.models['multiview_model'].pipeline.to('cuda')
paint_pipeline.render.device = torch.device('cuda')

# GPU 상태 확인
gpu_status_lines = []
gpu_status_lines.append("📊 GPU 상태 확인:")
try:
    # Diffusers 파이프라인의 내부 모델 확인
    delight_device = paint_pipeline.models['delight_model'].pipeline.unet.device
    multiview_device = paint_pipeline.models['multiview_model'].pipeline.unet.device
    gpu_status_lines.append(f"    - delight_model 디바이스: {delight_device}")
    gpu_status_lines.append(f"    - multiview_model 디바이스: {multiview_device}")
except:
    gpu_status_lines.append(f"    - 디바이스 확인 실패 (하지만 .to('cuda') 호출은 완료됨)")
gpu_status_lines.append(f"    - render 디바이스: {paint_pipeline.render.device}")
if torch.cuda.is_available():
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
    gpu_status_lines.append(f"    - GPU 메모리 사용: {gpu_mem:.2f} GB")
    gpu_status_lines.append(f"    - GPU 메모리 예약: {gpu_mem_reserved:.2f} GB")

# 콘솔에 출력
print("\n  " + "\n  ".join(gpu_status_lines))

time_records['텍스처 파이프라인 로드'] = time.time() - start_time
print(f"\n완료! (소요 시간: {time_records['텍스처 파이프라인 로드']:.2f}초)")

print("=" * 60)
print("모델에 텍스처를 입힙니다...")
if torch.cuda.is_available():
    gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
    print(f"  - 시작 전 GPU 메모리: {gpu_mem_before:.2f} GB")
start_time = time.time()
mesh_textured = paint_pipeline(mesh, image=image)  # 전처리된 이미지 사용
time_records['텍스처 생성'] = time.time() - start_time

# 텍스처 생성 프로파일링 결과 저장
texture_profiling = getattr(paint_pipeline, 'profiling', {})

if torch.cuda.is_available():
    gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
    print(f"  - 완료 후 GPU 메모리: {gpu_mem_after:.2f} GB (변화: {gpu_mem_after - gpu_mem_before:+.2f} GB)")
print(f"텍스처 생성이 완료되었습니다! (소요 시간: {time_records['텍스처 생성']:.2f}초)")
print("=" * 60)

# 3. 최종 결과물 저장
print("\n최종 결과물을 저장합니다...")
start_time = time.time()
mesh_textured.export(OUTPUT_PATH)
time_records['파일 저장'] = time.time() - start_time
print(f"완료! (소요 시간: {time_records['파일 저장']:.2f}초)")

# 4. 전체 소요 시간 출력 및 로그 저장
print("\n" + "=" * 60)
print("📊 처리 시간 요약")
print("=" * 60)
total_time = time.time() - start_total
for step, elapsed in time_records.items():
    percentage = (elapsed / total_time) * 100
    print(f"  {step:20s}: {elapsed:6.2f}초 ({percentage:5.1f}%)")
print("-" * 60)
print(f"  {'총 소요 시간':20s}: {total_time:6.2f}초")
print("=" * 60)

# 5. 사용된 설정값 출력
print("\n" + "=" * 60)
print("⚙️  사용된 설정값")
print("=" * 60)
print(f"  입력 이미지         : {INPUT_IMAGE}")
print(f"  출력 파일          : {OUTPUT_PATH}")
print(f"  배경 제거          : {'활성화' if REMOVE_BACKGROUND else '비활성화'}")
print("-" * 60)
print("  [형상 생성]")
print(f"    추론 단계         : {NUM_INFERENCE_STEPS}")
print(f"    Octree 해상도     : {OCTREE_RESOLUTION}")
print(f"    가이던스 스케일    : {GUIDANCE_SCALE}")
print("-" * 60)
print("  [텍스처 생성]")
print(f"    Delight 추론 단계 : {DELIGHT_INFERENCE_STEPS}")
print(f"    Multiview 추론 단계: {MULTIVIEW_INFERENCE_STEPS}")
print(f"    카메라 뷰 모드     : {CAMERA_VIEWS}")
print(f"    렌더 해상도       : {RENDER_SIZE} x {RENDER_SIZE}")
print(f"    텍스처 해상도      : {TEXTURE_SIZE} x {TEXTURE_SIZE}")
print("=" * 60)
print(f"\n✅ 텍스처가 적용된 모델이 '{OUTPUT_PATH}'에 저장되었습니다.")
print("=" * 60)

# 6. 로그 파일 저장
log_content = []
log_content.append("=" * 60)
log_content.append("Hunyuan3D-2 3D 모델 생성 로그")
log_content.append("=" * 60)
log_content.append(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_content.append("")

# GPU 상태 정보 추가
log_content.append("=" * 60)
log_content.extend(gpu_status_lines)
log_content.append("=" * 60)
log_content.append("")

# 텍스처 생성 단계별 상세 정보 추가
if texture_profiling:
    log_content.append("=" * 60)
    log_content.append("🔍 텍스처 생성 단계별 시간 분석")
    log_content.append("=" * 60)
    for step_name, step_time in texture_profiling.items():
        if step_name != 'TOTAL':
            percentage = (step_time / texture_profiling.get('TOTAL', 1)) * 100
            log_content.append(f"  {step_name:25s}: {step_time:7.2f}초 ({percentage:5.1f}%)")
    log_content.append("-" * 60)
    log_content.append(f"  {'TOTAL':25s}: {texture_profiling.get('TOTAL', 0):7.2f}초")
    log_content.append("=" * 60)
    log_content.append("")

log_content.append("=" * 60)
log_content.append("📊 처리 시간 요약")
log_content.append("=" * 60)
for step, elapsed in time_records.items():
    percentage = (elapsed / total_time) * 100
    log_content.append(f"  {step:20s}: {elapsed:6.2f}초 ({percentage:5.1f}%)")
log_content.append("-" * 60)
log_content.append(f"  {'총 소요 시간':20s}: {total_time:6.2f}초")
log_content.append("=" * 60)
log_content.append("")
log_content.append("=" * 60)
log_content.append("⚙️  사용된 설정값")
log_content.append("=" * 60)
log_content.append(f"  입력 이미지         : {INPUT_IMAGE}")
log_content.append(f"  출력 파일          : {OUTPUT_PATH}")
log_content.append(f"  배경 제거          : {'활성화' if REMOVE_BACKGROUND else '비활성화'}")
log_content.append("-" * 60)
log_content.append("  [형상 생성]")
log_content.append(f"    추론 단계         : {NUM_INFERENCE_STEPS}")
log_content.append(f"    Octree 해상도     : {OCTREE_RESOLUTION}")
log_content.append(f"    가이던스 스케일    : {GUIDANCE_SCALE}")
log_content.append("-" * 60)
log_content.append("  [텍스처 생성]")
log_content.append(f"    Delight 추론 단계 : {DELIGHT_INFERENCE_STEPS}")
log_content.append(f"    Multiview 추론 단계: {MULTIVIEW_INFERENCE_STEPS}")
log_content.append(f"    카메라 뷰 모드     : {CAMERA_VIEWS}")
log_content.append(f"    렌더 해상도       : {RENDER_SIZE} x {RENDER_SIZE}")
log_content.append(f"    텍스처 해상도      : {TEXTURE_SIZE} x {TEXTURE_SIZE}")
log_content.append("=" * 60)

with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_content))

print(f"\n📝 로그 파일이 '{LOG_PATH}'에 저장되었습니다.")