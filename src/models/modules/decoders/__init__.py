import pkgutil
import importlib
import os

package_dir = os.path.dirname(__file__)
for (module_loader, name, ispkg) in pkgutil.iter_modules([package_dir]):
    importlib.import_module(f"{__name__}.{name}")
