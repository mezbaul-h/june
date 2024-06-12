from ..settings import settings


class ModelMeta(type):
    def __call__(cls, *args, **kwargs):
        """Called when you call MyModel()"""
        obj = type.__call__(cls, kwargs)
        obj.__post_init__(kwargs)
        return obj


class ModelBase(metaclass=ModelMeta):
    def __init__(self, kwargs):
        self.device = kwargs.pop("device", None) or settings.TORCH_DEVICE
        self.generation_args = kwargs.pop("generation_args", None) or {}
        self.model_id = kwargs.pop("model")

    def __post_init__(self, kwargs):
        if kwargs:
            raise ValueError(f"Unknown model kwargs for {self.__class__.__name__}: {list(kwargs.keys())}")
