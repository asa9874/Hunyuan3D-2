import torch
import os
import time
from PIL import Image
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

# ==== 설정 가능한 매개변수 ====
# 경로 설정
INPUT_IMAGE = 'my/input/bag.jpg'     # 입력 이미지 경로

# 배경 제거 설정
REMOVE_BACKGROUND = True            # 배경 제거 활성화 (True/False)

# 생성 품질 설정
NUM_INFERENCE_STEPS = 5        # 추론 단계 수 (기본값: 5, 더 높을수록 품질 향상, 시간 증가) 50했을때 짱느림;
OCTREE_RESOLUTION = 128         # Octree 해상도 (기본값: 256, 더 높을수록 세밀함) - 256은 UV wrapping이 매우 느림
GUIDANCE_SCALE = 5            # 가이던스 스케일 (기본값: 5, 입력 이미지 충실도)
# ============================

# 출력 파일명을 입력 파일명과 동일하게 설정 (확장자만 .glb로 변경)
input_filename = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
OUTPUT_PATH = f'my/output/{input_filename}.glb'

# 시간 측정을 위한 변수
time_records = {}

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
paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-paint-v2-0-turbo'
)

# ⚡ 중요: 텍스처 생성 모델들을 GPU로 이동 (성능 향상)
print("  - 모델을 GPU로 이동 중...")
paint_pipeline.models['delight_model'].pipeline.to('cuda')
paint_pipeline.models['multiview_model'].pipeline.to('cuda')
paint_pipeline.render.device = torch.device('cuda')

# GPU 상태 확인
print("\n  📊 GPU 상태 확인:")
try:
    # Diffusers 파이프라인의 내부 모델 확인
    delight_device = paint_pipeline.models['delight_model'].pipeline.unet.device
    multiview_device = paint_pipeline.models['multiview_model'].pipeline.unet.device
    print(f"    - delight_model 디바이스: {delight_device}")
    print(f"    - multiview_model 디바이스: {multiview_device}")
except:
    print(f"    - 디바이스 확인 실패 (하지만 .to('cuda') 호출은 완료됨)")
print(f"    - render 디바이스: {paint_pipeline.render.device}")
if torch.cuda.is_available():
    print(f"    - GPU 메모리 사용: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"    - GPU 메모리 예약: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

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

# 4. 전체 소요 시간 출력
print("\n" + "=" * 60)
print("📊 처리 시간 요약")
print("=" * 60)
total_time = sum(time_records.values())
for step, elapsed in time_records.items():
    percentage = (elapsed / total_time) * 100
    print(f"  {step:20s}: {elapsed:6.2f}초 ({percentage:5.1f}%)")
print("-" * 60)
print(f"  {'총 소요 시간':20s}: {total_time:6.2f}초")
print("=" * 60)
print(f"\n✅ 텍스처가 적용된 모델이 '{OUTPUT_PATH}'에 저장되었습니다.")
print("=" * 60)