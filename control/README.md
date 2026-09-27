# 제어 코드

- [common/](common/README.md): 기하, IK branch 선택, 5차 trajectory, MoveIt 호출, 기본 action 서버.
- [position/](position/README.md): 현재 사용하는 필터링 위치 P 보정과 속도 feedforward.
- [force/](force/README.md): 향후 힘 제어용 수치 필터와 한계 검사.
- [impedance/](impedance/README.md): 향후 impedance 제어용 spring/damper 수치 함수.

로봇마다 같은 수치 코드를 복제하지 않고 관절 이름·제한·튜닝을 프로필로 받는다.
현재 두 로봇은 모두 6축이며 다른 자유도의 배열 계산 테스트도 포함한다.
새 자유도의 모델·IK·driver를 자동 지원한다는 의미는 아니다.

현재 활성화 가능한 방식은 `position`이다. 실제 모터 토크를 직접 계산·전송하지 않는다.
Isaac 위치/속도 목표 인터페이스와 force/torque 장비 인터페이스는 분리해서 구현한다.
