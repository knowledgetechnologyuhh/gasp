import os
from setuptools import setup, find_packages

def get_files_in_directory(directory, extension):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

setup(
    name='GASP',
    version='2.0.0',
    packages=find_packages(),
    include_package_data=True,
    url='software.knowledge-technology.info#gasp',
    license='MIT License',
    author='Fares Abawi',
    author_email='fares.abawi@uni-hamburg.de',
    maintainer='Fares Abawi',
    maintainer_email='fares.abawi@uni-hamburg.de',
    description='Social cue integration for dynamic saliency prediction',
    extras_require={
        "RobotAudioVisualCongruency":
            ['dash==1.11.0',
             'dash-bootstrap-components==0.9.2',
             'ffpyplayer==4.3.2',
             'waitress==2.0.0',
             'pygame==2.0.3',
             'moviepy==1.0.3',
             'wrapyfi==0.4.46',
             'pexpect==4.8.0'],
    },
    install_requires=['cython==3.0.0a9',
                      'pandas==1.3.2',
                      'scikit-build==0.12.0',
                      # 'opencv-python==4.2.0.34',
                      'opencv-contrib-python==4.5.3.56',
                      'matplotlib==3.4.3',
                      'seaborn==0.11.2',
                      #'git+https://github.com/pytorch/accimage.git#egg=accimage',  # when CPU is Intel IPP compatible. This package is deprecated
                      'torch==1.7.1',
                      'torchvision==0.8.2',
                      'pytorch-lightning==1.5.10',
                      'tensorboard==2.2.0',
                      'librosa==0.8.1',
                      'cffi==1.14.6',
                      'resampy==0.2.2',
                      'sounddevice==0.4.2',
                      'numpy==1.20.3',
                      'h5py==2.10.0',
                      'scipy==1.4.1',
                      'pillow==8.3.3',
                      'urllib3==1.26.11',
                      'numba==0.49.1',
                      'pooch==1.5.1',
                      'scikit-image==0.18.3',
                      'scikit-learn==0.24.2',
                      'face_recognition==1.3.0',
                      'face-alignment==1.3.4',
                      'facenet-pytorch==2.5.2',
                      'comet_ml',
                      'tqdm==4.62.1'],
    entry_points = {
            'console_scripts': [
                'gasp_train=gazenet.bin.train:main',
                'gasp_infer=gazenet.bin.infer:main',
                'gasp_download_manager=gazenet.bin.download_manager:main',
                'gasp_scripts=gazenet.bin.scripts:main',
            ],
        },
    exclude_package_data={
        "datasets": ["*.zip", "*.7z", "*.tar.gz", "*.ptb", "*.ptb.tar", "*.npy", "*.npz", "*.hd5", "*.txt", "*.jpg", "*.png", "*.gif", "*.avi", "*.mp4", "*.wav", "*.mp3"]},
    package_data={
        "": ["datasets/processed/center_bias.jpg", "datasets/processed/center_bias_bw.jpg",
             "datasets/wtmsimgaze2020/videos/*.mp4"],
    },
    data_files=[
        ("datasets/processed/", ["datasets/processed/center_bias.jpg", "datasets/processed/center_bias_bw.jpg"]),
        ("datasets/wtmsimgaze2020/videos/", get_files_in_directory('datasets/wtmsimgaze2020/videos/', '.mp4'))
    ]
    )
