from setuptools import setup, find_packages
from pathlib import Path

# load dependencies
def load_requirements():
    requirements_path = Path('requirements.txt')
    if not requirements_path.exists():
        return []
    with open(requirements_path, 'r') as f:
        # Filter out comments and empty lines
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# project setup 
setup(
    name='hair-type-classifier',
    version='0.1.0',
    description='Computer Vision project for hair type classification.',
    author='Ola/sq7alx',
    
    packages=find_packages(
        include=[
            'src', 'src.*', 
            'config', 'config.*', 
            'scripts', 'scripts.*',
            'face_parsing', 'face_parsing.*'
        ],
        exclude=[
            'tests', 'notebooks', '*.pyc', '*.egg-info'
        ]
    ),
    
    package_data={
        'config': ['config.yaml'],
    },
    
    install_requires=load_requirements(),
    
    python_requires='>=3.8',
)