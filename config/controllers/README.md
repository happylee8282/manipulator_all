# 제어 방식

`position.yaml`은 현재 구현된 5차 궤적 + 속도 feedforward + 제한된 위치 P 보정을
선택한다. 로봇별 튜닝 수치는 `../robots/*.yaml`의 `settings.controller`에 있다.
배열의 순서는 해당 로봇의 `settings.project.joint_names`와 같아야 한다.

`force.yaml`, `impedance.yaml`은 향후 제어 방식의 등록 위치다. 아직 장비 제어를
구현하지 않았으므로 선택하면 launch가 실행 전에 오류를 낸다. 수치 함수가 존재해도
실제 로봇에서 사용할 수 있다는 의미가 아니다.
