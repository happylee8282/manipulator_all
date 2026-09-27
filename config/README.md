# 설정 프로필

`configuration.py`가 다음 순서로 `settings`를 재귀 병합한다. 뒤의 프로필이 같은 값을 덮어쓴다.

```text
tasks → robots → tools → backends → controllers
    → CLI 경로/속도 선택 → runs/<robot>_<backend>/resolved.yaml
```

- `robots/`: 모델·MoveIt 파일과 관절 이름/제한/초기 자세/튜닝.
- `tools/`: 장착 도구 선택·TCP 정보. 실제 모델 수정은 URDF에도 반영한다.
- `backends/`: Isaac 및 미지원 real 등록 위치.
- `controllers/`: position과 향후 force/impedance 선택.
- `tasks/`: 경로·clearance·충돌·작업 설정.

프로필 파일은 일반 YAML이며 ROS의 `ros__parameters` 문법이 아니다. launch가 필요한
부분을 노드 파라미터로 전달한다. 상대 파일 경로는 패키지 자산 루트 기준이다.
`resources`의 `../` 경로는 `asset_workspace` 아래 기존 자산을 가리킨다.

resolved 설정에는 소스 프로필·모델·경로 입력의 SHA-256이 포함된다. 설정이 동일하면
파일을 다시 쓰지 않아 준비한 경로를 이유 없이 stale 처리하지 않는다. 활성 launch가
있는 run의 설정을 다르게 덮어쓰는 것은 거절한다.
