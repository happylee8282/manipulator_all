# 경로 생성

기존 안경 경로 생성 알고리즘을 이 폴더에 모았습니다. 로봇 변경 시 원본 안경 경로는 공유하고 world 변환, TCP 방향, 접근 자세를 프로필에서 선택합니다.

```text
원본 PCD [안경 로컬 mm]
  → source_generation/step1: 얇은 면·경계 추출
  → source_generation/step2: 초록 중심선 생성·보정
  → CSV/PCD [안경 로컬 mm]
  → cartesian_generator: 안경 world 변환·간격·노즐 자세
  → Cartesian NPZ [m, quaternion xyzw]
  → control/common/moveit_solver: 로봇별 IK/FK·충돌 검사
```

| 위치 | 내용 |
|---|---|
| `source_generation/` | 원본 PCD에서 경로를 다시 계산하는 Step 1·2 구현 |
| `data/` | 기존에 사용한 보정 완료 초록 경로 CSV/PCD |
| `cartesian_generator.py` | 입력 검증, 재샘플링, world 변환, 접근/이탈/정착 구간, 결과 NPZ/CSV/JSON 생성 |
| `path_profile.py` | 중앙·굴곡 진입·가파른 구간별 간격과 부드러운 연결 |
| `clearance_geometry.py` | world-Z 또는 표면 법선 방향의 오프셋 계산 |

입력 CSV 필수 열은 `global_{x,y,z}_mm`, `surface_center_{x,y,z}_mm`, `outer_{x,y,z}_mm`, `inner_{x,y,z}_mm`입니다. 실제 노즐 목표 간격은 `config/tasks/glass_following.yaml`의 `cartesian_pose`에서 설정합니다. 초록 원본 경로와 노즐 목표 궤적은 서로 다른 데이터입니다.

RV-4FL에서는 동적으로 정착한 안경 snapshot을 읽은 후 Cartesian 경로를 생성합니다. UR3e의 정적 장면 변환은 명시적인 planning-only 프로필에서 사용합니다. 생성된 NPZ와 solver 검증 결과는 `runs/<robot>_<backend>/output/`에 저장하며 이전 실행 궤적을 입력 예제로 복사하지 않습니다.
