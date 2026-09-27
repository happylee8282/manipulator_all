# Mitsubishi RV-4FL

원본: `step1_modelsolution/rv-4fl_rev_B_step/rv4fl_description/urdf/rv4fl.urdf`.
링크 메시, 관절축, 관성 추정, 노즐 위치를 보존하고 패키지 메시 URI만 이 패키지로 바꿨습니다.

- 모델: `rv4fl.urdf`; TCP `nozzle_tcp`; `tool0` 기준 노즐 끝 z=0.065 m.
- 메시: `meshes/visual`, `meshes/collision`; 원본 CAD의 mm 스케일을 URDF에서 0.001로 변환.
- 초기 자세: `[0, 0, pi/2, 0, 0, 0]` rad.
- root link는 `base_link`; world→base 변환은 launch가 프로필에서 읽습니다.

관성과 effort 제한에는 원본의 추정/시뮬레이션 값이 포함됩니다. 이 모델을 복사했다고 실물 동역학 파라미터가 식별된 것은 아닙니다.
