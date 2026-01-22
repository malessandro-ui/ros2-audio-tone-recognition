from setuptools import setup

package_name = "prosody2policy_adapter"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "tone_to_style_node = prosody2policy_adapter.tone_to_style_node:main",
        ],
    },
)
