# Hunyuan3D-2 설정 매개변수 가이드

## 📋 목차
1. [형상 생성 설정](#형상-생성-설정)
2. [텍스처 생성 설정](#텍스처-생성-설정)
3. [카메라 뷰 설정](#카메라-뷰-설정)
4. [렌더링 설정](#렌더링-설정)
5. [시나리오별 권장 설정](#시나리오별-권장-설정)

---

## 형상 생성 설정

### 1. `NUM_INFERENCE_STEPS` - 추론 단계 수
**기본값:** 5 | **범위:** 3-10  
**📦 영향 모델:** `Hunyuan3D-DiT-v2-0-Turbo` (형상 생성) / `Hunyuan3D-DiT-v2-mv-Turbo` (멀티뷰)

#### 의미
- AI 모델이 2D 이미지에서 3D 형상을 생성할 때 수행하는 **반복 정제 횟수**
- Diffusion 모델이 노이즈에서 점진적으로 3D 구조를 생성하는 단계 수

#### 값의 영향
| 값 | 속도 | 품질 | 설명 |
|---|---|---|---|
| **3** | ⚡⚡⚡ 매우 빠름 | ★☆☆ 낮음 | 빠른 프리뷰, 거친 형상 |
| **5** | ⚡⚡ 빠름 | ★★☆ 균형 | 기본 권장값, 속도/품질 균형 |
| **7-10** | ⚡ 느림 | ★★★ 높음 | 세밀한 디테일, 복잡한 형상 |

#### 언제 변경하나요?
- ⬇️ **낮게 (3)**: 빠른 테스트, 간단한 형상 (구, 상자 등)
- ⬆️ **높게 (8-10)**: 복잡한 세부사항 (손가락, 얼굴 표정, 장식 등)

---

### 2. `OCTREE_RESOLUTION` - 3D 공간 해상도
**기본값:** 192 | **범위:** 128, 192, 256, 380, 512  
**📦 영향 모델:** `Hunyuan3D-DiT-v2-0-Turbo` (형상 생성) / `Hunyuan3D-DiT-v2-mv-Turbo` (멀티뷰)

#### 의미
- 3D 공간을 나누는 **격자의 세밀도** (Octree: 3차원 공간 분할 구조)
- 값이 클수록 더 많은 복셀(voxel)로 공간을 표현 → 더 정밀한 형상

#### 값의 영향
| 값 | 메모리 | 속도 | 품질 | 최적 용도 |
|---|---|---|---|---|
| **128** | 0.5GB | ⚡⚡⚡ 매우 빠름 | ★☆☆ 낮음 | 빠른 프리뷰, 저사양 GPU |
| **192** | 1.2GB | ⚡⚡ 빠름 | ★★☆ 균형 | 일반적인 사용 (권장) |
| **256** | 2.5GB | ⚡ 보통 | ★★★ 높음 | 상세한 형상 |
| **380** | 5GB | ⏱ 느림 | ★★★★ 매우 높음 | 멀티뷰 고품질 |
| **512** | 10GB+ | 🐌 매우 느림 | ★★★★★ 최고 | 최종 제작물, 고사양 GPU |

#### 예시
```python
# 간단한 형상 (컵, 공)
OCTREE_RESOLUTION = 128

# 보통 복잡도 (신발, 가구)
OCTREE_RESOLUTION = 192

# 복잡한 형상 (손가락, 옷 주름)
OCTREE_RESOLUTION = 256

# 매우 세밀한 형상 (얼굴, 조각상)
OCTREE_RESOLUTION = 380-512
```

---

### 3. `GUIDANCE_SCALE` - 입력 이미지 충실도
**기본값:** 5 | **범위:** 3-10  
**📦 영향 모델:** `Hunyuan3D-DiT-v2-0-Turbo` (형상 생성) / `Hunyuan3D-DiT-v2-mv-Turbo` (멀티뷰)

#### 의미
- 생성된 3D 모델이 **입력 이미지와 얼마나 정확히 일치**해야 하는지 조절
- Classifier-Free Guidance 강도 (높을수록 입력에 더 충실)

#### 값의 영향
| 값 | 효과 | 특징 |
|---|---|---|
| **3-4** | 창의적 | 입력 이미지를 참고하되 AI가 자유롭게 해석, 부드러운 형상 |
| **5** | 균형 | 입력 이미지 충실 + 자연스러운 3D 변환 (권장) |
| **7-10** | 엄격함 | 입력 이미지에 매우 충실, 과도하면 부자연스러울 수 있음 |

#### 언제 변경하나요?
- ⬇️ **낮게 (3-4)**: 
  - 입력 이미지가 불완전하거나 일부만 보이는 경우
  - AI가 보이지 않는 뒷면을 자연스럽게 추론하길 원할 때
  
- ⬆️ **높게 (7-9)**: 
  - 입력 이미지가 명확하고 정확한 경우
  - 정면 형상이 입력과 정확히 일치해야 할 때
  - 로고나 텍스트가 있는 경우

---

## 텍스처 생성 설정

### 4. `DELIGHT_INFERENCE_STEPS` - 그림자 제거 단계
**기본값:** 6 | **범위:** 4-10  
**📦 영향 모델:** `Hunyuan3D-Delight-v2-0` (그림자 제거 전처리 모델)

#### 의미
- 입력 이미지의 **그림자와 하이라이트를 제거**하는 AI 처리 단계 수
- 3D 텍스처는 조명이 없는 순수한 색상이어야 하므로, 사진의 그림자를 제거

#### 값의 영향
| 값 | 속도 | 효과 | 사용 시기 |
|---|---|---|---|
| **4** | ⚡ 빠름 | 기본적인 그림자 제거 | 균일한 조명의 이미지 |
| **6** | ⚡⚡ 보통 | 효과적인 그림자 제거 (권장) | 일반적인 사진 |
| **8-10** | 🐌 느림 | 강력한 그림자 제거 | 강한 그림자가 있는 이미지 |

#### 예시
```python
# 스튜디오 조명 (균일한 빛)
DELIGHT_INFERENCE_STEPS = 4

# 일반 사진 (약간의 그림자)
DELIGHT_INFERENCE_STEPS = 6

# 햇빛/강한 그림자가 있는 사진
DELIGHT_INFERENCE_STEPS = 8-10
```

---

### 5. `MULTIVIEW_INFERENCE_STEPS` - 멀티뷰 텍스처 생성 단계
**기본값:** 6 | **범위:** 4-10  
**📦 영향 모델:** `Hunyuan3D-Paint-v2-0-Turbo` (텍스처 생성 메인 모델)

#### 의미
- 3D 모델의 보이지 않는 각도(뒷면, 옆면)의 **텍스처를 AI가 생성**하는 단계 수
- 입력 이미지에 없는 부분을 예측하여 완성

#### 값의 영향
| 값 | 속도 | 품질 | 설명 |
|---|---|---|---|
| **4** | ⚡⚡ 빠름 | ★★☆ 기본 | 빠른 생성, 뒷면이 단순할 때 |
| **6** | ⚡ 보통 | ★★★ 좋음 | 권장값, 자연스러운 뒷면 |
| **8-10** | 🐌 느림 | ★★★★ 매우 좋음 | 복잡한 패턴의 뒷면 생성 |

#### 이 설정이 중요한 이유
- **입력 이미지**: 정면만 보임 (예: 티셔츠 앞면)
- **AI가 생성**: 뒷면, 옆면, 소매 안쪽 등
- 단계가 많을수록 → 더 그럴듯한 뒷면 생성

---

## 카메라 뷰 설정

### 6. `CAMERA_VIEWS` - 텍스처 렌더링 각도 수
**기본값:** 'standard' | **옵션:** 'minimal', 'fast', 'standard'  
**📦 영향 모델:** `Hunyuan3D-Paint-v2-0-Turbo` (텍스처 생성 시 렌더링 각도 제어)

#### 의미
- 3D 모델의 텍스처를 생성할 때 **몇 개의 각도에서 렌더링**할지 결정
- 각도가 많을수록 텍스처 품질 향상, 속도 느려짐

#### 옵션 상세

| 모드 | 뷰 수 | 각도 구성 | 속도 | 품질 | 권장 용도 |
|---|---|---|---|---|---|
| **minimal** | 3뷰 | 정면(0°), 좌측(120°), 우측(240°) | ⚡⚡⚡ 매우 빠름 | ★★☆ 기본 | 빠른 프리뷰, 간단한 형상 |
| **fast** | 4뷰 | 전/후/좌/우 (0°, 90°, 180°, 270°) | ⚡⚡ 빠름 | ★★★ 좋음 | 일반 사용 권장 |
| **standard** | 6뷰 | 전/후/좌/우 + 위/아래 (수평4 + 수직2) | ⚡ 느림 | ★★★★ 매우 좋음 | 최종 제작물, 복잡한 형상 |

#### 각 뷰의 가중치 (중요도)
```python
# minimal (3뷰)
- 정면(0°): 가중치 1.0  ← 가장 중요
- 좌측(120°): 가중치 0.3
- 우측(240°): 가중치 0.3

# fast (4뷰)
- 정면(0°): 가중치 1.0   ← 가장 중요
- 후면(180°): 가중치 0.5 ← 두 번째로 중요
- 좌측(90°): 가중치 0.1
- 우측(270°): 가중치 0.1

# standard (6뷰)
- 정면(0°): 가중치 1.0   ← 가장 중요
- 후면(180°): 가중치 0.5
- 좌측(90°): 가중치 0.1
- 우측(270°): 가중치 0.1
- 위(90° 상승): 가중치 0.05
- 아래(-90° 하강): 가중치 0.05
```

#### 시간 비교
```
minimal:  약 300초 (5분)
fast:     약 400초 (6.5분)  ← 권장
standard: 약 550초 (9분)
```

---

## 렌더링 설정

### 7. `RENDER_SIZE` - 렌더 해상도
**기본값:** 2048 | **옵션:** 1024, 2048, 4096  
**📦 영향 모델:** `Differentiable Renderer` (텍스처 생성 전 Normal/Position 맵 렌더링)

#### 의미
- 3D 모델을 각 각도에서 **2D 이미지로 렌더링할 때의 해상도**
- 이 렌더링 이미지들이 AI 입력으로 사용됨 (높을수록 세밀한 텍스처)

#### 값의 영향
| 값 | VRAM | 속도 | 품질 | 권장 용도 |
|---|---|---|---|---|
| **1024** | 4GB | ⚡⚡⚡ 빠름 | ★★☆ 기본 | 프리뷰, 테스트 |
| **2048** | 8GB | ⚡⚡ 보통 | ★★★ 좋음 | 일반 사용 (권장) |
| **4096** | 16GB+ | 🐌 느림 | ★★★★ 최고 | 최종 제작, 인쇄용 |

---

### 8. `TEXTURE_SIZE` - 최종 텍스처 해상도
**기본값:** 2048 | **옵션:** 1024, 2048, 4096  
**📦 영향 모델:** `UV Baker` (최종 텍스처 맵 베이킹 단계)

#### 의미
- 3D 모델에 **최종적으로 적용되는 텍스처 맵의 해상도**
- UV 맵에 베이킹되는 이미지의 픽셀 크기 (가로 × 세로)

#### 값의 영향
| 값 | 파일 크기 | 디테일 | 권장 용도 |
|---|---|---|---|
| **1024** | ~5MB | ★★☆ 기본 | 웹, 모바일, 게임 (저사양) |
| **2048** | ~15MB | ★★★ 좋음 | 일반 3D 작업, 렌더링 |
| **4096** | ~50MB | ★★★★ 최고 | 고품질 렌더링, 인쇄, 영화 |

#### RENDER_SIZE vs TEXTURE_SIZE 차이
```
RENDER_SIZE (2048)
   ↓ [AI가 각 각도에서 렌더링]
   ↓ [여러 각도의 이미지들]
   ↓ [UV 맵에 합성/베이킹]
TEXTURE_SIZE (2048)
   ↓ [최종 텍스처 맵 생성]
   ↓ [3D 모델에 적용]
```

**팁:** 일반적으로 두 값을 동일하게 설정하는 것이 좋습니다.

---

## 시나리오별 권장 설정

### 🎯 시나리오 1: 빠른 프리뷰 (속도 우선)
```python
# 형상 생성
NUM_INFERENCE_STEPS = 3
OCTREE_RESOLUTION = 128
GUIDANCE_SCALE = 5

# 텍스처 생성
DELIGHT_INFERENCE_STEPS = 4
MULTIVIEW_INFERENCE_STEPS = 4
CAMERA_VIEWS = 'minimal'

# 렌더링
RENDER_SIZE = 1024
TEXTURE_SIZE = 1024
```
**예상 시간:** 약 3-4분  
**용도:** 빠른 테스트, 형상 확인, 이터레이션

---

### ⚖️ 시나리오 2: 균형잡힌 설정 (권장)
```python
# 형상 생성
NUM_INFERENCE_STEPS = 5
OCTREE_RESOLUTION = 192
GUIDANCE_SCALE = 5

# 텍스처 생성
DELIGHT_INFERENCE_STEPS = 6
MULTIVIEW_INFERENCE_STEPS = 6
CAMERA_VIEWS = 'fast'

# 렌더링
RENDER_SIZE = 2048
TEXTURE_SIZE = 2048
```
**예상 시간:** 약 6-7분  
**용도:** 일반적인 3D 모델 생성, 대부분의 경우 추천

---

### 🏆 시나리오 3: 고품질 최종 제작물
```python
# 형상 생성
NUM_INFERENCE_STEPS = 8
OCTREE_RESOLUTION = 256
GUIDANCE_SCALE = 7

# 텍스처 생성
DELIGHT_INFERENCE_STEPS = 8
MULTIVIEW_INFERENCE_STEPS = 8
CAMERA_VIEWS = 'standard'

# 렌더링
RENDER_SIZE = 4096
TEXTURE_SIZE = 4096
```
**예상 시간:** 약 15-20분  
**필요 VRAM:** 16GB 이상  
**용도:** 최종 제작물, 포트폴리오, 인쇄용

---

### 🎮 시나리오 4: 게임 에셋 (최적화)
```python
# 형상 생성
NUM_INFERENCE_STEPS = 5
OCTREE_RESOLUTION = 192
GUIDANCE_SCALE = 5

# 텍스처 생성
DELIGHT_INFERENCE_STEPS = 6
MULTIVIEW_INFERENCE_STEPS = 6
CAMERA_VIEWS = 'fast'

# 렌더링
RENDER_SIZE = 2048
TEXTURE_SIZE = 1024  # 게임용 최적화
```
**특징:** 적절한 폴리곤 수 + 최적화된 텍스처 크기

---

### 🎨 시나리오 5: 복잡한 캐릭터/조각상
```python
# 형상 생성
NUM_INFERENCE_STEPS = 10
OCTREE_RESOLUTION = 380  # 멀티뷰라면 380, 싱글이면 256
GUIDANCE_SCALE = 6

# 텍스처 생성
DELIGHT_INFERENCE_STEPS = 8
MULTIVIEW_INFERENCE_STEPS = 10
CAMERA_VIEWS = 'standard'

# 렌더링
RENDER_SIZE = 4096
TEXTURE_SIZE = 4096
```
**특징:** 얼굴 표정, 손가락, 옷 주름 등 세밀한 디테일 포착

---

## � 모델 파이프라인 구조

전체 프로세스에서 각 설정이 영향을 주는 모델:

```
[단계 1: 형상 생성]
📦 Hunyuan3D-DiT-v2-0-Turbo (단일 이미지용)
   또는
📦 Hunyuan3D-DiT-v2-mv-Turbo (멀티뷰용)
   ├─ NUM_INFERENCE_STEPS ← 추론 반복 횟수
   ├─ OCTREE_RESOLUTION ← 3D 공간 해상도
   └─ GUIDANCE_SCALE ← 입력 이미지 충실도

        ↓ 생성된 3D 메쉬 (형상만, 텍스처 없음)

[단계 2: 텍스처 전처리 - 그림자 제거]
📦 Hunyuan3D-Delight-v2-0
   └─ DELIGHT_INFERENCE_STEPS ← 그림자 제거 정도

        ↓ 조명 제거된 깨끗한 입력 이미지

[단계 3: Normal/Position 맵 렌더링]
📦 Differentiable Renderer (Nvdiffrast 기반)
   └─ RENDER_SIZE ← 렌더링 해상도

        ↓ Normal 맵, Position 맵 (각 카메라 각도별)

[단계 4: 멀티뷰 텍스처 생성]
📦 Hunyuan3D-Paint-v2-0-Turbo
   ├─ MULTIVIEW_INFERENCE_STEPS ← 텍스처 생성 품질
   └─ CAMERA_VIEWS ← 렌더링할 카메라 각도 수

        ↓ 각 각도별 텍스처 이미지

[단계 5: UV 베이킹]
📦 UV Baker (xatlas 기반)
   └─ TEXTURE_SIZE ← 최종 텍스처 맵 크기

        ↓ 텍스처가 적용된 최종 3D 모델 (.glb)
```

### 모델별 설정 요약

| 모델 | 담당 작업 | 영향받는 설정 |
|---|---|---|
| **DiT-v2-0-Turbo** | 형상 생성 (단일) | NUM_INFERENCE_STEPS, OCTREE_RESOLUTION, GUIDANCE_SCALE |
| **DiT-v2-mv-Turbo** | 형상 생성 (멀티뷰) | NUM_INFERENCE_STEPS, OCTREE_RESOLUTION, GUIDANCE_SCALE |
| **Delight-v2-0** | 그림자 제거 | DELIGHT_INFERENCE_STEPS |
| **Paint-v2-0-Turbo** | 텍스처 생성 | MULTIVIEW_INFERENCE_STEPS, CAMERA_VIEWS |
| **Differentiable Renderer** | Normal/Position 렌더링 | RENDER_SIZE |
| **UV Baker** | 텍스처 베이킹 | TEXTURE_SIZE |

---

## �💡 최적화 팁

### GPU 메모리 부족 시
1. `OCTREE_RESOLUTION` 낮추기 (256 → 192 → 128)
2. `RENDER_SIZE`와 `TEXTURE_SIZE` 낮추기 (4096 → 2048 → 1024)
3. `CAMERA_VIEWS`를 'minimal'로 변경

### 속도 향상 필요 시
1. `NUM_INFERENCE_STEPS` 줄이기 (5 → 3)
2. `CAMERA_VIEWS`를 'minimal'로
3. `MULTIVIEW_INFERENCE_STEPS` 줄이기 (6 → 4)

### 품질 향상 필요 시
1. `OCTREE_RESOLUTION` 올리기 (192 → 256 → 380)
2. `NUM_INFERENCE_STEPS` 올리기 (5 → 8)
3. `CAMERA_VIEWS`를 'standard'로
4. `MULTIVIEW_INFERENCE_STEPS` 올리기 (6 → 8)

---

## 📊 설정 조합 치트시트

| 목표 | Inference Steps | Octree | Camera Views | Render/Texture Size | 시간 |
|---|---|---|---|---|---|
| 최고 속도 | 3 | 128 | minimal | 1024 | ~3분 |
| 빠름 | 5 | 192 | fast | 1024 | ~5분 |
| **균형 (권장)** | **5** | **192** | **fast** | **2048** | **~7분** |
| 고품질 | 7 | 256 | standard | 2048 | ~12분 |
| 최고 품질 | 10 | 380-512 | standard | 4096 | ~20분+ |

---

## ⚠️ 주의사항

1. **GPU 메모리**: `OCTREE_RESOLUTION`과 `RENDER_SIZE`가 VRAM을 가장 많이 사용
2. **처리 시간**: `CAMERA_VIEWS`와 `MULTIVIEW_INFERENCE_STEPS`가 시간에 큰 영향
3. **품질과 속도의 균형**: 프리뷰는 낮은 설정, 최종본은 높은 설정 사용
4. **입력 이미지 품질**: 설정값보다 **입력 이미지 품질**이 더 중요할 수 있음

---

## 🔗 관련 파일
- `color.py` - 단일 이미지 3D 생성
- `color_batch.py` - 배치 처리
- `multiview.py` - 멀티뷰 이미지 3D 생성
- `플로우.md` - 전체 파이프라인 설명
