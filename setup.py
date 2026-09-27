from pathlib import Path

from setuptools import find_packages, setup


PACKAGE = "lee_manipulator_part"
ROOT = Path(__file__).parent
asset_directories = ("config", "launch", "docs", "robot/description", "robot/moveit_config", "path/data")
data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE]),
    ("share/" + PACKAGE, ["package.xml", "README.md"]),
]
for directory in asset_directories:
    files_by_parent = {}
    for path in sorted((ROOT / directory).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            relative = path.relative_to(ROOT)
            files_by_parent.setdefault(str(relative.parent), []).append(str(relative))
    data_files.extend((f"share/{PACKAGE}/{parent}", paths) for parent, paths in files_by_parent.items())
data_files.append((f"share/{PACKAGE}/robot/adapters", [
    "robot/adapters/rv4fl_scene.py", "robot/adapters/ur3e_scene.py", "robot/adapters/ur3e_graph_repair.py",
]))
installed_assets = {source for _, sources in data_files for source in sources}
for directory in ("control", "execution", "gripper", "path", "robot", "safety", "scripts", "test", "resource"):
    for pattern in ("README.md", "requirements.txt"):
        for path in sorted((ROOT / directory).rglob(pattern)):
            relative = str(path.relative_to(ROOT))
            if relative not in installed_assets:
                data_files.append((f"share/{PACKAGE}/{path.relative_to(ROOT).parent}", [relative]))
                installed_assets.add(relative)

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[PACKAGE] + [f"{PACKAGE}.{name}" for name in find_packages(exclude=("test", "test.*"))],
    package_dir={PACKAGE: "."},
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="happy",
    maintainer_email="happy@todo.todo",
    description="Profile-based UR3e and RV-4FL Cartesian path planning and guarded Isaac execution",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "position_controller = lee_manipulator_part.control.position.controller:main",
        "generate_cartesian_poses = lee_manipulator_part.path.cartesian_generator:main",
        "solve_moveit_path = lee_manipulator_part.control.common.moveit_solver:main",
        "execute_solved_path = lee_manipulator_part.execution.guarded_executor:main",
        "preflight = lee_manipulator_part.safety.preflight:main",
        "diagnose_guard_failure = lee_manipulator_part.safety.diagnose_guard_failure:main",
        "task_runner = lee_manipulator_part.workflow:main",
        "source_path_generator = lee_manipulator_part.source_pipeline:main",
        "gripper_mock = lee_manipulator_part.gripper.gripper_node:main",
    ]},
)
