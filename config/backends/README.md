# Backend profiles

`isaac.yaml`은 시뮬레이션 설정이며 Isaac Python과 wall-time ROS bridge를 선택합니다. launch의 domain 기본값 auto는 RV-4FL=31, UR3e=32로 분리합니다.
`real.yaml`은 향후 연결 지점이고 `supported: false`입니다. 현재 실물 driver를 실행하지 않습니다.

Isaac은 RV `/isaac_joint_commands`와 UR `/isaac_joint_command`가 달라 robot profile이 command topic을 소유합니다. state topic은 둘 다 `/isaac_joint_states`입니다. 두 로봇을 동시에 띄우려면 DDS domain 또는 모든 node/service/action/topic namespace를 일관되게 분리해야 합니다.
