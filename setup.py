from setuptools import setup

setup(
    name='Contrast-you',
    version='',
    packages=['contrastyou'],
    url='',
    license='',
    author='jizong',
    author_email='',
    description='',
    entry_points={
        "console_scripts": [
            "delete_failed_experiments=script.delete_failed_runs:_main"
        ]

    }
)
