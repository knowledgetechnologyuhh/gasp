from setuptools import setup, find_packages

setup(
    name='GASP',
    version='1.0.1',
    packages=find_packages(),
    include_package_data=True,
    url='software.knowledge-technology.info#gasp',
    license='MIT License',
    author='Fares Abawi',
    author_email='fares.abawi@uni-hamburg.de',
    maintainer='Fares Abawi',
    maintainer_email='fares.abawi@uni-hamburg.de',
    description='Social cue integration for dynamic saliency prediction',
    install_requires=['cython==0.29.1',
                      'pandas==1.0.3',
                      # 'opencv-python==4.2.0.34',
                      'scikit-build==0.12.0'
                      'opencv-contrib-python==4.2.0.34',
                      'matplotlib==3.2.1',
                      'torch==1.7.1',
                      'tqdm==4.46.0',
                      'librosa==0.4.2',
                      'cffi==1.14.0',
                      'resampy==0.2.2',
                      'sounddevice==0.3.15',
                      'torchvision==0.8.2',
                      'pytorch-lightning==1.3.3',
                      'tensorboard==2.2.0',
                      'numpy==1.18.3',
                      'h5py==2.10.0',
                      'scipy==1.4.1',
                      'pillow==8.3.2',
                      'urllib3==1.25.9',
                      'numba==0.49.1',
                      'scikit-image==0.17.2',
                      'scikit-learn==0.22.2.post1',
                      'face_recognition==1.3.0',
                      'face-alignment==1.1.1',
                      'facenet-pytorch==2.5.0',
                      'comet_ml'],
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
        "": ["datasets/processed/center_bias.jpg", "datasets/processed/center_bias_bw.jpg"],
    },
    data_files=[("datasets/processed/", ["datasets/processed/center_bias.jpg", "datasets/processed/center_bias_bw.jpg"])]
    )
