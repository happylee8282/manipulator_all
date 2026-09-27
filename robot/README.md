# Robot

로봇 형상과 MoveIt 설정, 시뮬레이터 연결을 관리합니다. 제어 계산은 `control/`, 경로는 `path/`에서 공유합니다.

- `description/rv4fl/`: Mitsubishi RV-4FL + 65 mm 노즐의 조립 URDF와 메시.
- `description/ur3e/`: 이전 UR 프로젝트의 UR3e + 72 mm 노즐 조립 URDF와 메시.
- `description/tools/`: 두 노즐의 실제 STL 형상.
- `moveit_config/`: 로봇별 SRDF, IK plugin, 관절 제한, OMPL, action controller 설정.
- `adapters/`: Isaac 실행 명령 생성 및 기존 로봇별 USD runner. real backend는 미구현을 명확히 반환합니다.

로봇 선택은 `config/robots/{rv4fl,ur3e}.yaml`에서 합니다. UR은 여러 UR 계열을 통칭하는 프로필이 아니라 **UR3e**입니다.
URDF/메시는 이 패키지에 복사되어 있습니다. 큰 Isaac USD와 안경 STL은 기존 workspace 파일을 참조하므로 전체 시뮬레이션이 독립 배포 가능한 상태는 아닙니다.

RV-4FL에는 기존 동적 안경 정착/감시 runner를 이식했습니다. UR3e에는 기존 장면 runner와 계획 모델을 넣었으며, 동적 안경 guard가 없어 기본 `dry_run_only: true`입니다. 이번 폴더 생성 자체는 새 실기/시뮬레이션 성능 검증을 의미하지 않습니다.
