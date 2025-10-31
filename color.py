import torch
import os
import time
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# ==== 설정 가능한 매개변수 ====
# 경로 설정
INPUT_IMAGE = 'my/input/bag.png'     # 입력 이미지 경로

# 생성 품질 설정
NUM_INFERENCE_STEPS = 5        # 추론 단계 수 (기본값: 5, 더 높을수록 품질 향상, 시간 증가) 50했을때 짱느림;
OCTREE_RESOLUTION = 256         # Octree 해상도 (기본값: 256, 더 높을수록 세밀함)
GUIDANCE_SCALE = 5            # 가이던스 스케일 (기본값: 5, 입력 이미지 충실도)
# ============================

# 출력 파일명을 입력 파일명과 동일하게 설정 (확장자만 .glb로 변경)
input_filename = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
OUTPUT_PATH = f'my/output/{input_filename}.glb'

# 시간 측정을 위한 변수
time_records = {}

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
print(f"  - 입력 이미지: {INPUT_IMAGE}")
print(f"  - 추론 단계: {NUM_INFERENCE_STEPS}")
print(f"  - 해상도: {OCTREE_RESOLUTION}")
print(f"  - 가이던스 스케일: {GUIDANCE_SCALE}")
start_time = time.time()
# shape_pipeline의 결과는 리스트이므로, [0]을 붙여 첫 번째 모델 객체를 꺼냅니다.
mesh = shape_pipeline(
    image=INPUT_IMAGE,
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
time_records['텍스처 파이프라인 로드'] = time.time() - start_time
print(f"완료! (소요 시간: {time_records['텍스처 파이프라인 로드']:.2f}초)")

print("=" * 60)
print("모델에 텍스처를 입힙니다...")
start_time = time.time()
mesh_textured = paint_pipeline(mesh, image=INPUT_IMAGE)
time_records['텍스처 생성'] = time.time() - start_time
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