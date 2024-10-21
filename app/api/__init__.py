from .florence import FlorenceServe, Florence

__florence_serve = None


def florence(model) -> Florence:
    global __florence_serve
    if __florence_serve is None:
        __florence_serve = FlorenceServe()
    return __florence_serve.get_or_load_model(model)
