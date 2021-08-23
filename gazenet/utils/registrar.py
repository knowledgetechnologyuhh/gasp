import os
from glob import glob

from gazenet.utils.helpers import dynamic_module_import


class RobotControllerRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        RobotControllerRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "robots", "**", "controller.py"), recursive=True)
        modules = ["gazenet.robots." + module.replace(os.path.dirname(__file__) + "/../robots/", "") for module in modules]
        dynamic_module_import(modules, globals())


# A pytorch lightning module
class ModelRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        ModelRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "models", "**", "model.py"), recursive=True)
        modules = ["gazenet.models." + module.replace(os.path.dirname(__file__) + "/../models/", "") for module in modules]
        dynamic_module_import(modules, globals())


# A pytorch lightning data module
class ModelDataRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        ModelDataRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "models", "**", "generator.py"), recursive=True)
        modules = ["gazenet.models." + module.replace(os.path.dirname(__file__) + "/../models/", "") for module in modules]
        dynamic_module_import(modules, globals())


class MetricsRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        MetricsRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "models", "**", "metrics.py"), recursive=True)
        modules = ["gazenet.models." + module.replace(os.path.dirname(__file__) + "/../models/", "") for module in modules]
        dynamic_module_import(modules, globals())


# An InferenceSampleProcessor inheriting class
class InferenceRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        InferenceRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "models", "**", "infer.py"), recursive=True)
        modules = ["gazenet.models." + module.replace(os.path.dirname(__file__) + "/../models/", "") for module in modules]
        dynamic_module_import(modules, globals())


# A SampleReader inheriting class
class ReaderRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        ReaderRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "readers", "*.py"), recursive=True)
        modules = ["gazenet.readers." + module.replace(os.path.dirname(__file__) + "/../readers/", "") for module in modules]
        dynamic_module_import(modules, globals())
        # add the data reader as well
        modules = glob(os.path.join(os.path.dirname(__file__), "dataset_processors.py"), recursive=False)
        modules = ["gazenet.utils." + modules[0].replace(os.path.dirname(__file__) + "/", "")]
        dynamic_module_import(modules, globals())


# A SampleProcessor inheriting class
class SampleRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        SampleRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "readers", "*.py"), recursive=True)
        modules = ["gazenet.readers." + module.replace(os.path.dirname(__file__) + "/../readers/", "") for module in modules]
        dynamic_module_import(modules, globals())
        # add the data sample as well
        modules = glob(os.path.join(os.path.dirname(__file__), "dataset_processors.py"), recursive=False)
        modules = ["gazenet.utils." + modules[0].replace(os.path.dirname(__file__) + "/", "")]
        dynamic_module_import(modules, globals())


class InferenceConfigRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        InferenceConfigRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "configs", "*.py"), recursive=True)
        modules = ["gazenet.configs." + module.replace(os.path.dirname(__file__) + "/../configs/", "") for module in modules]
        dynamic_module_import(modules, globals())


class TrainingConfigRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        TrainingConfigRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "..", "configs", "*.py"), recursive=True)
        modules = ["gazenet.configs." + module.replace(os.path.dirname(__file__) + "/../configs/", "") for module in modules]
        dynamic_module_import(modules, globals())


class PlotterRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        PlotterRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "annotation_plotter.py"), recursive=False)
        modules = ["gazenet.utils." + modules[0].replace(os.path.dirname(__file__) + "/", "")]
        dynamic_module_import(modules, globals())


class FaceDetectorRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        FaceDetectorRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "face_detectors.py"), recursive=False)
        modules = ["gazenet.utils." + modules[0].replace(os.path.dirname(__file__) + "/", "")]
        dynamic_module_import(modules, globals())


class AudioFeatureRegistrar(object):
    registry = {}

    @staticmethod
    def register(cls):
        AudioFeatureRegistrar.registry[cls.__name__] = cls
        return cls

    @staticmethod
    def scan():
        modules = glob(os.path.join(os.path.dirname(__file__), "audio_features.py"), recursive=False)
        modules = ["gazenet.utils." + modules[0].replace(os.path.dirname(__file__) + "/", "")]
        dynamic_module_import(modules, globals())