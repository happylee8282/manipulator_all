# Tool meshes

`rv4fl_nozzle/nozzle.stl`과 `ur3e_nozzle/nozzle.stl`은 해당 조립 URDF가 직접 사용하는 실제 노즐 메시입니다.
장착 joint와 TCP link는 현재 각 로봇 URDF에 포함되어 있습니다. 프로필만 바꿔 임의의 공구를 즉시 끼우는 자동 조립 기능은 아닙니다.

새 그리퍼를 추가할 때에는 본체/손가락 형상, 움직이는 관절, mimic 관계, 충돌 모델과 TCP가 필요합니다. 노즐 모델을 움직이는 그리퍼로 가장하는 placeholder URDF는 제공하지 않습니다.
