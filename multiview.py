import torch
import os
import time
from datetime import datetime
from PIL import Image
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

# CUDA/cuDNN 초기화 및 환경 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # CUDA 동기화 모드
torch.backends.cudnn.enabled = False  # cuDNN 비활성화 (초기화 오류 방지)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

# CUDA 장치 확인 및 초기화
if torch.cuda.is_available():
    torch.cuda.init()
    print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"cuDNN 활성화: {torch.backends.cudnn.enabled}")
    # 초기 GPU 메모리 정리
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
else:
    print("⚠️ CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    exit(1)

# ==== 설정 가능한 매개변수 ====
# 경로 설정 - 멀티뷰 이미지 (최소 2개, 권장 3-4개)
INPUT_IMAGES = {
    "front": "mymv/input/front.png",      # 필수: 정면
    #"left": "mymv/input/left.png",        # 선택: 좌측
    "back": "mymv/input/back.png",        # 선택: 후면
    # "right": "mymv/input/right.png",    # 선택: 우측
}

# 배경 제거 설정
REMOVE_BACKGROUND = True            # 배경 제거 활성화 (True/False)

# 형상 생성 품질 설정 (MultiView 모델)
NUM_INFERENCE_STEPS = 5             # 형상 생성 추론 단계 (기본: 5, 범위: 3-10)
OCTREE_RESOLUTION = 380             # Octree 해상도 (256=빠름, 380=균형, 512=느림/고품질)
NUM_CHUNKS = 20000                  # 청크 수 (메모리 관리, 기본: 20000)
GUIDANCE_SCALE = 5                  # 가이던스 스케일 (기본: 5, 범위: 3-10)
USE_FLASH_VDM = True                # FlashVDM 최적화 (속도 향상)

# 텍스처 생성 품질 설정
DELIGHT_INFERENCE_STEPS = 6         # 그림자 제거 추론 단계 (기본: 6, 범위: 4-10)
MULTIVIEW_INFERENCE_STEPS = 6       # 멀티뷰 생성 추론 단계 (기본: 6, 범위: 4-10)

# 카메라 뷰 설정 (텍스처 생성 시)
CAMERA_VIEWS = 'standard'           # 'standard'(6뷰,느림), 'fast'(4뷰,빠름), 'minimal'(3뷰,매우빠름)

# 렌더링 설정
RENDER_SIZE = 2048                  # 렌더 해상도 (1024=빠름, 2048=기본, 4096=고품질/느림)
TEXTURE_SIZE = 2048                 # 텍스처 해상도 (1024, 2048, 4096)
# ============================

# 출력 파일명 설정
output_basename = 'multiview_model'
OUTPUT_PATH = f'mymv/output/{output_basename}.glb'

# 로그 파일 경로 설정
log_dir = 'mymv/log'
os.makedirs(log_dir, exist_ok=True)
os.makedirs('mymv/output', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = os.path.join(log_dir, f'{timestamp}_{output_basename}.txt')

# 시간 측정을 위한 변수
time_records = {}
start_total = time.time()  # 전체 시작 시간

# 0. 멀티뷰 이미지 로드 및 배경 제거
print("=" * 60)
print("멀티뷰 이미지를 처리합니다...")
print(f"  - 입력 이미지 개수: {len(INPUT_IMAGES)}개")
start_time = time.time()

processed_images = {}
for view_name, image_path in INPUT_IMAGES.items():
    if not os.path.exists(image_path):
        print(f"  ⚠️  경고: '{image_path}' 파일이 없습니다. 이 뷰는 건너뜁니다.")
        continue
    
    print(f"  - 처리 중: {view_name} ({os.path.basename(image_path)})")
    image = Image.open(image_path)
    original_mode = image.mode
    print(f"    원본 모드: {original_mode}")
    
    # 배경 제거 수행 여부 결정
    if REMOVE_BACKGROUND and original_mode in ['RGB', 'L']:
        print(f"    배경 제거 수행 중...")
        if original_mode == 'L':
            image = image.convert('RGB')
        rembg = BackgroundRemover()
        image = rembg(image)
        print(f"    배경 제거 완료!")
    elif REMOVE_BACKGROUND and original_mode == 'RGBA':
        print(f"    이미 알파 채널 존재")
    else:
        image = image.convert("RGBA")
    
    processed_images[view_name] = image

if len(processed_images) < 2:
    print("\n❌ 오류: 최소 2개 이상의 이미지가 필요합니다.")
    print("   (권장: front, left, back 3개)")
    exit(1)

print(f"\n✓ {len(processed_images)}개 뷰 처리 완료: {list(processed_images.keys())}")
time_records['이미지 전처리'] = time.time() - start_time
print(f"완료! (소요 시간: {time_records['이미지 전처리']:.2f}초)")
print("=" * 60)

# 1. 멀티뷰 형상(Shape) 생성 파이프라인 로드
print("\n" + "=" * 60)
print("멀티뷰 형상 생성 파이프라인을 로드합니다...")
print("  📦 모델: Hunyuan3D-DiT-v2-mv-Turbo")
start_time = time.time()

try:
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mv',
        subfolder='hunyuan3d-dit-v2-mv-turbo',
        torch_dtype=torch.float16,
        variant='fp16'
    )
    
    # GPU로 명시적으로 이동
    print("  - 모델을 GPU로 이동 중...")
    shape_pipeline.to('cuda')
    
    # FlashVDM 최적화 활성화
    if USE_FLASH_VDM:
        print("  ⚡ FlashVDM 최적화 활성화 중...")
        shape_pipeline.enable_flashvdm()
        print("  ✓ FlashVDM 활성화 완료 (속도 향상)")
    
    # GPU 메모리 상태 확인
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # GPU 캐시 정리
        torch.cuda.synchronize()
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  - GPU 메모리 사용: {gpu_mem:.2f} GB")
    
    time_records['파이프라인 로드'] = time.time() - start_time
    print(f"완료! (소요 시간: {time_records['파이프라인 로드']:.2f}초)")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 파이프라인 로드 중 오류 발생: {e}")
    print("\n💡 해결 방법:")
    print("  1. CUDA 드라이버 업데이트")
    print("  2. PyTorch 재설치: pip install torch --upgrade --force-reinstall")
    print("  3. cuDNN 재설치")
    print("  4. 시스템 재부팅 후 재시도")
    exit(1)

# 2. 멀티뷰 이미지로부터 3D 형상 생성
print("\n" + "=" * 60)
print(f"멀티뷰 이미지로부터 3D 모델 형상을 생성합니다...")
print(f"  - 입력 뷰: {list(processed_images.keys())}")
print(f"  - 추론 단계: {NUM_INFERENCE_STEPS}")
print(f"  - Octree 해상도: {OCTREE_RESOLUTION}")
print(f"  - 청크 수: {NUM_CHUNKS}")
print(f"  - 가이던스 스케일: {GUIDANCE_SCALE}")

# GPU 메모리 정리 및 상태 확인
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gpu_mem_before_shape = torch.cuda.memory_allocated() / 1024**3
    print(f"  - 시작 전 GPU 메모리: {gpu_mem_before_shape:.2f} GB")

start_time = time.time()

try:
    mesh = shape_pipeline(
        image=processed_images,
        num_inference_steps=NUM_INFERENCE_STEPS,
        octree_resolution=OCTREE_RESOLUTION,
        num_chunks=NUM_CHUNKS,
        guidance_scale=GUIDANCE_SCALE,
        generator=torch.manual_seed(42),  # 재현성을 위한 시드
        output_type='trimesh'
    )[0]
    
    time_records['형상 생성'] = time.time() - start_time
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"형상 생성이 완료되었습니다! (소요 시간: {time_records['형상 생성']:.2f}초)")
    print(f"  - 생성된 메쉬: {len(mesh.vertices):,}개 정점, {len(mesh.faces):,}개 면")
    
except RuntimeError as e:
    print(f"\n❌ 형상 생성 중 오류 발생: {e}")
    if "cuDNN" in str(e):
        print("\n💡 cuDNN 오류 해결 방법:")
        print("  1. 다른 프로그램에서 GPU를 사용 중이면 종료")
        print("  2. 설정 낮추기:")
        print("     OCTREE_RESOLUTION = 256")
        print("     NUM_CHUNKS = 10000")
        print("  3. 시스템 재부팅")
        print("  4. CUDA Toolkit 및 cuDNN 재설치")
    exit(1)

# 형상 생성 파이프라인 메모리 해제 (텍스처 생성을 위한 공간 확보)
print("  - Shape 파이프라인 메모리 해제 중...")
del shape_pipeline
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
print("  ✓ 메모리 해제 완료")
print("=" * 60)

# 3. 텍스처(Texture) 생성 파이프라인 로드
print("\n텍스처 생성 파이프라인을 로드합니다...")
start_time = time.time()

# 카메라 뷰 설정
if CAMERA_VIEWS == 'minimal':
    camera_azims = [0, 120, 240]
    camera_elevs = [0, 0, 0]
    view_weights = [1, 0.3, 0.3]
    print(f"  - 카메라 뷰: 최소 (3뷰, 매우 빠름)")
elif CAMERA_VIEWS == 'fast':
    camera_azims = [0, 90, 180, 270]
    camera_elevs = [0, 0, 0, 0]
    view_weights = [1, 0.1, 0.5, 0.1]
    print(f"  - 카메라 뷰: 빠름 (4뷰)")
else:  # standard
    camera_azims = [0, 90, 180, 270, 0, 180]
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
        self.delight_inference_steps = DELIGHT_INFERENCE_STEPS
        self.multiview_inference_steps = MULTIVIEW_INFERENCE_STEPS
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

# 4. 텍스처 생성 (정면 이미지 사용)
print("=" * 60)
print("모델에 텍스처를 입힙니다...")
print("  - 텍스처 기준 이미지: front (정면)")
if torch.cuda.is_available():
    gpu_mem_before = torch.cuda.memory_allocated() / 1024**3
    print(f"  - 시작 전 GPU 메모리: {gpu_mem_before:.2f} GB")

start_time = time.time()
# 정면 이미지를 텍스처 생성의 기준으로 사용
reference_image = processed_images.get('front') or list(processed_images.values())[0]
mesh_textured = paint_pipeline(mesh, image=reference_image)
time_records['텍스처 생성'] = time.time() - start_time

# 텍스처 생성 프로파일링 결과 저장
texture_profiling = getattr(paint_pipeline, 'profiling', {})

if torch.cuda.is_available():
    gpu_mem_after = torch.cuda.memory_allocated() / 1024**3
    print(f"  - 완료 후 GPU 메모리: {gpu_mem_after:.2f} GB (변화: {gpu_mem_after - gpu_mem_before:+.2f} GB)")
print(f"텍스처 생성이 완료되었습니다! (소요 시간: {time_records['텍스처 생성']:.2f}초)")
print("=" * 60)

# 5. 최종 결과물 저장
print("\n최종 결과물을 저장합니다...")
start_time = time.time()
mesh_textured.export(OUTPUT_PATH)
time_records['파일 저장'] = time.time() - start_time
print(f"완료! (소요 시간: {time_records['파일 저장']:.2f}초)")

# 6. 전체 소요 시간 출력
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

# 7. 사용된 설정값 출력
print("\n" + "=" * 60)
print("⚙️  사용된 설정값")
print("=" * 60)
print(f"  입력 이미지 뷰      : {list(processed_images.keys())}")
print(f"  출력 파일          : {OUTPUT_PATH}")
print(f"  배경 제거          : {'활성화' if REMOVE_BACKGROUND else '비활성화'}")
print("-" * 60)
print("  [멀티뷰 형상 생성]")
print(f"    모델              : Hunyuan3D-DiT-v2-mv-Turbo")
print(f"    FlashVDM 최적화   : {'활성화' if USE_FLASH_VDM else '비활성화'}")
print(f"    추론 단계         : {NUM_INFERENCE_STEPS}")
print(f"    Octree 해상도     : {OCTREE_RESOLUTION}")
print(f"    청크 수           : {NUM_CHUNKS}")
print(f"    가이던스 스케일    : {GUIDANCE_SCALE}")
print("-" * 60)
print("  [텍스처 생성]")
print(f"    Delight 추론 단계 : {DELIGHT_INFERENCE_STEPS}")
print(f"    Multiview 추론 단계: {MULTIVIEW_INFERENCE_STEPS}")
print(f"    카메라 뷰 모드     : {CAMERA_VIEWS}")
print(f"    렌더 해상도       : {RENDER_SIZE} x {RENDER_SIZE}")
print(f"    텍스처 해상도      : {TEXTURE_SIZE} x {TEXTURE_SIZE}")
print("=" * 60)
print(f"\n✅ 텍스처가 적용된 멀티뷰 3D 모델이 '{OUTPUT_PATH}'에 저장되었습니다.")
print("=" * 60)

# 8. 로그 파일 저장
log_content = []
log_content.append("=" * 60)
log_content.append("Hunyuan3D-2 멀티뷰 3D 모델 생성 로그")
log_content.append("=" * 60)
log_content.append(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_content.append("")

# 입력 정보
log_content.append("=" * 60)
log_content.append("📷 입력 멀티뷰 이미지")
log_content.append("=" * 60)
for view_name, image_path in INPUT_IMAGES.items():
    if view_name in processed_images:
        log_content.append(f"  ✓ {view_name:10s}: {image_path}")
    else:
        log_content.append(f"  ✗ {view_name:10s}: {image_path} (건너뜀)")
log_content.append("=" * 60)
log_content.append("")

# GPU 상태 정보
log_content.append("=" * 60)
log_content.extend(gpu_status_lines)
log_content.append("=" * 60)
log_content.append("")

# 텍스처 생성 단계별 상세 정보
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

# 전체 처리 시간
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

# 설정값
log_content.append("=" * 60)
log_content.append("⚙️  사용된 설정값")
log_content.append("=" * 60)
log_content.append(f"  입력 이미지 뷰      : {list(processed_images.keys())}")
log_content.append(f"  출력 파일          : {OUTPUT_PATH}")
log_content.append(f"  배경 제거          : {'활성화' if REMOVE_BACKGROUND else '비활성화'}")
log_content.append("-" * 60)
log_content.append("  [멀티뷰 형상 생성]")
log_content.append(f"    모델              : Hunyuan3D-DiT-v2-mv-Turbo")
log_content.append(f"    FlashVDM 최적화   : {'활성화' if USE_FLASH_VDM else '비활성화'}")
log_content.append(f"    추론 단계         : {NUM_INFERENCE_STEPS}")
log_content.append(f"    Octree 해상도     : {OCTREE_RESOLUTION}")
log_content.append(f"    청크 수           : {NUM_CHUNKS}")
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
