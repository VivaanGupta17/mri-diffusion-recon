"""
Setup configuration for MRI-DiffRecon.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

# Core requirements only (not optional)
core_requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-image>=0.20.0",
    "h5py>=3.8.0",
    "pyyaml>=6.0",
    "matplotlib>=3.7.0",
    "tensorboard>=2.13.0",
    "tqdm>=4.65.0",
    "Pillow>=9.5.0",
]

setup(
    name="mri-diffrecon",
    version="0.1.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description=(
        "Score-Based Diffusion Models for Accelerated MRI Reconstruction"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mri-diffusion-recon",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=core_requirements,
    extras_require={
        "full": requirements,
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "perceptual": [
            "lpips>=0.1.4",
        ],
        "distributed": [
            "accelerate>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mri-train=scripts.train:main",
            "mri-reconstruct=scripts.reconstruct:main",
            "mri-evaluate=scripts.evaluate:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords=[
        "MRI", "diffusion models", "score matching",
        "image reconstruction", "accelerated MRI",
        "deep learning", "medical imaging",
        "k-space", "inverse problems",
    ],
    project_urls={
        "Paper": "https://arxiv.org",
        "Dataset": "https://fastmri.org",
        "Bug Reports": "https://github.com/yourusername/mri-diffusion-recon/issues",
    },
)
