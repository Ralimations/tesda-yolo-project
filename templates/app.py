from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_APP = Path(__file__).resolve().parents[1] / "app.py"
spec = spec_from_file_location("dataset_script_app", PROJECT_APP)
module = module_from_spec(spec)
spec.loader.exec_module(module)
app = module.app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
