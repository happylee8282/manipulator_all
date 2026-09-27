# Robot descriptions

`rv4fl/rv4fl.urdf`, `ur3e/ur3e.urdf`가 launch에서 읽는 **노즐까지 조립된 모델**입니다. 모든 메시 URI는 `package://lee_manipulator_part/robot/description/...`를 사용합니다.

| 로봇 | 관절 이름 | TCP | world 연결 |
|---|---|---|---|
| RV-4FL | joint1–joint6 | nozzle_tcp | launch의 static TF, z=0.05 m |
| UR3e | shoulder_pan_joint 등 6축 | nozzle_tip | URDF의 base_joint, z=0.05 m |

URDF 수정 시 matching SRDF, `config/robots`, `config/tools`, 실제 USD 장착 위치도 확인합니다. `config/tools`의 장착/TCP 값은 현재 조립 모델 설명용 메타데이터이며, YAML 변경만으로 고정 URDF가 재조립되지는 않습니다.
그리퍼 교체는 실제 손가락 관절이 포함된 모델과 matching MoveIt 설정을 추가하는 작업입니다. 제공된 노즐에는 움직이는 손가락이 없습니다.
