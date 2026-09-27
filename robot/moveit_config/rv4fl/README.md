# RV-4FL MoveIt

원본 `step1_modelsolution/rv-4fl_rev_B_step/rv4fl_moveit_config/config`의 SRDF·IK·OMPL·controller 설정입니다.
그룹 `manipulator`, TCP `nozzle_tcp`, action `/rv4fl_arm_controller/follow_joint_trajectory`를 사용합니다.
`rv4fl.srdf`의 world virtual joint에 대응하는 base static TF가 launch에서 필요합니다.
