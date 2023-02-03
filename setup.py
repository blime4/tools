from setuptools import setup, find_packages

import sys
python_min_version = (3, 6, 2)
version_range_max = max(sys.version_info[1], 8) + 1
setup(
    name="hooktools",
    version="0.1",
    author="DLframework Team",
    description="Pytorch hook tools",
    packages=find_packages(),
    python_requires='>=3.6.2',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
    ] + ['Programming Language :: Python :: 3.{}'.format(i) for i in range(python_min_version[1], version_range_max)],
    package_data={
        'hooktools':[
            "config/*.yaml",
            "config/*.yml",
        ]
    },
    license='BSD-3',
    keywords='pytorch machine learning',
)