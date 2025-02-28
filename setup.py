from setuptools import find_packages, setup

setup(
    name="omni",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'omni=omni.run:main',  # This makes the 'omni' command available
        ],
    },
    python_requires='>=3.10',  # Adjust based on your needs
    author="Wenxiao",
    description="realtime demo",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)
