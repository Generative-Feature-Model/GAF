from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='GAF',
        version='1.0',
        packages=find_packages(
            exclude=('configs', 'models', 'output', 'datasets')
        ),
        ext_modules=[
            CUDAExtension('boundary_max_pooling_cuda', [
                'GAF/prop_pooling/boundary_max_pooling_cuda.cpp',
                'GAF/prop_pooling/boundary_max_pooling_kernel.cu'
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
