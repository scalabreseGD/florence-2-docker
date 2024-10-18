from .florence import FlorenceServe, Florence

__florence_serve = FlorenceServe()


def florence(model) -> Florence:
    return __florence_serve.get_or_load_model(model)
